# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Fine-tuning the library models for named entity recognition on CoNLL-2003. """


import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score
from torch import nn

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from jer_trainer import JERTrainer

from utils_ner import NerDataset, Split, get_labels
from torch_jer import RobertaForJointEntityRelationClassification

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    use_fast: bool = field(
        default=False, metadata={"help": "Set this flag to use fast tokenization."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    labels: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to a file containing all labels. If not specified, CoNLL-2003 labels are used."
        },
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    report_to: str = field(
        default="none", metadata={"help": "Don't log for predictions"}
    )




def compute_metrics(p: EvalPrediction) -> Dict:
    preds_list, out_label_list = align_predictions(p.predictions, p.label_ids)
    return {
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }


class PredictionModel:
    def align_predictions(self, 
        predictions: np.ndarray, label_ids: np.ndarray,
    ) -> Tuple[List[int], List[int]]:
        preds = np.argmax(predictions, axis=2)
        batch_size, seq_len = preds.shape

        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                    out_label_list[i].append(self.label_map[label_ids[i][j]])
                    preds_list[i].append(self.label_map[preds[i][j]])

        return preds_list, out_label_list

    def __init__(self, dict_args):
        parser = HfArgumentParser(
            (ModelArguments, DataTrainingArguments, TrainingArguments)
        )
        self.model_args, self.data_args, self.training_args = parser.parse_dict(
            dict_args
        )

        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO
            if self.training_args.local_rank in [-1, 0]
            else logging.WARN,
        )
        logger.info("Training/evaluation parameters %s", self.training_args)

        # Set seed
        set_seed(self.training_args.seed)

        # Prepare CONLL-2003 task
        self.labels = get_labels(self.data_args.labels)
        self.label_map: Dict[int, str] = {
            i: label for i, label in enumerate(self.labels)
        }
        self.num_labels = len(self.labels)

        # Load pretrained model and tokenizer
        #
        # Distributed training:
        # The .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.

        self.config = AutoConfig.from_pretrained(
            self.model_args.model_name_or_path,
            num_labels=self.num_labels,
            id2label=self.label_map,
            label2id={label: i for i, label in enumerate(self.labels)},
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path,
            use_fast=self.model_args.use_fast,
        )
        self.model = RobertaForJointEntityRelationClassification.from_pretrained(
            self.model_args.model_name_or_path,
            from_tf=bool(".ckpt" in self.model_args.model_name_or_path),
            config=self.config,
        )

        # Initialize our Trainer
        self.trainer = JERTrainer(
            model=self.model,
            args=self.training_args,
        )

    def do_predict(self, data_str):
        # Predict
        test_dataset = NerDataset(
            data_str=data_str,
            tokenizer=self.tokenizer,
            labels=self.labels,
            model_type=self.config.model_type,
            max_seq_length=self.data_args.max_seq_length,
            overwrite_cache=self.data_args.overwrite_cache,
            mode=Split.test,
            use_cache=False,
        )
        predictions, label_ids, metrics = self.trainer.predict(test_dataset)
        relation_preds = predictions[1]
        relation_final = np.argmax(relation_preds, axis=1)
        print(np.array(relation_final))
        actual_relations = []

        for i in test_dataset.features:
            actual_relations.append(i.relation_labels[0])
        print(np.array(actual_relations))
        print(accuracy_score(np.array(actual_relations), relation_final))
        predictions = predictions[0]
        print(predictions)
        preds_list, _ = self.align_predictions(predictions, label_ids)

        output_test_results_file = os.path.join("/tmp/", "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key, value in metrics.items():
                logger.info("  %s = %s", key, value)
                writer.write("%s = %s\n" % (key, value))

        # Save predictions
        output_test_predictions_file = os.path.join("/tmp/", "test_predictions.txt")
        with open(output_test_predictions_file, "w") as writer:
            example_id = 0
            for line in data_str.split('\n'):
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    writer.write(line)
                    if not preds_list[example_id]:
                        example_id += 1
                elif "\tCONTEXT" in line:
                    pass
                elif "\tBOS_" in line:
                    pass
                elif preds_list[example_id]:
                    output_line = (
                        line.split()[0]
                        + " "
                        + preds_list[example_id].pop(0)
                        + "\n"
                    )
                    writer.write(output_line)
                else:
                    logger.warning(
                        "Maximum sequence length exceeded: No prediction for '%s'.",
                        line.split()[0],
                    )



if __name__ == "__main__":
    args = {
        "max_seq_length": 512,
        "output_dir": "/tmp/",
        "labels": "./3_iob_data/labels.txt",
        "model_name_or_path": "distilroberta-base",
        "per_device_eval_batch_size": 1,
        "fp16": True,
    }
    predmodel = PredictionModel(args)
    data_str = "Finally	O\nĠGroup	B-EXPL_VAR\n"
    predmodel.do_predict(data_str)
