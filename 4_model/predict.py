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


import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

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


class PredictionModel:
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

        # Prepare label maps
        self.labels = get_labels(self.data_args.labels)
        self.label_map: Dict[int, str] = {
            i: label for i, label in enumerate(self.labels)
        }
        self.num_labels = len(self.labels)

        # Load pretrained model and tokenizer
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

    def create_dataset(self, data_str):
        """If the data_str does not contain any entity labels, the entity will be considered as
        label "O". If there is no relation id, the id will be "-100" and the loss will
        not be calculated."""
        return NerDataset(
            data_str=data_str,
            tokenizer=self.tokenizer,
            labels=self.labels,
            model_type=self.config.model_type,
            max_seq_length=self.data_args.max_seq_length,
            overwrite_cache=self.data_args.overwrite_cache,
            mode=Split.test,
            use_cache=False,
        )

    def do_predict(self, dataset):
        # predictions is a tuple containing softmaxed (Entity_Probabilities, Relation_Probabilities)
        predictions, true_entity_ids, metrics = self.trainer.predict(dataset)
        pred_entity_ids = np.argmax(predictions[0], axis=2)
        pred_relations = np.argmax(predictions[1], axis=1)
        true_relations = np.array([i.relation_labels[0] for i in dataset.features])
        # print("true_relations", true_relations)
        # print("pred_relations", pred_relations)
        # print("relation_accuracy", accuracy_score(true_relations, pred_relations))

        # print("true_entity_ids", true_entity_ids)
        # print("pred_entity_ids", pred_entity_ids)
        # trimmed_pred_entity_labels, trimmed_true_entity_labels = self.trim_and_convert_entity_ids(
        #     pred_entity_ids, true_entity_ids
        # )
        # print("trimmed_pred_entity_labels", trimmed_pred_entity_labels)
        # print("trimmed_true_entity_labels", trimmed_true_entity_labels)

        # print("eval_loss", metrics["eval_loss"])

        # self.generate_iob(trimmed_pred_entity_labels, data_str)
        return pred_relations, true_relations, pred_entity_ids, true_entity_ids

    def trim_and_convert_entity_ids(
        self,
        pred_ids: np.ndarray,
        true_ids: np.ndarray,
    ) -> Tuple[List[int], List[int]]:
        batch_size, seq_len = pred_ids.shape

        true_labels = [[] * batch_size]
        pred_labels = [[] * batch_size]

        for i in range(batch_size):
            for j in range(seq_len):
                if true_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                    true_labels[i].append(self.label_map[true_ids[i][j]])
                    pred_labels[i].append(self.label_map[pred_ids[i][j]])

        return pred_labels, true_labels

    def set_relation(self, data_str: str, relation_status: Optional[Union[int, bool]]):
        if relation_status is None:
            return data_str
        elif relation_status:
            return "<s>\tBOS_1\n" + data_str
        else:
            return "<s>\tBOS_0\n" + data_str

    def generate_iob(self, converted_entity_labels, data_str):
        i = 0
        for line in data_str.split("\n"):
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                print(line, end="")
                if not converted_entity_labels[i]:
                    i += 1
            elif "\tCONTEXT" in line:
                pass
            elif "\tBOS_" in line:
                pass
            elif converted_entity_labels[i]:
                output_line = (
                    line.split()[0] + " " + converted_entity_labels[i].pop(0) + "\n"
                )
                print(output_line, end="")
            else:
                logger.warning(
                    "Maximum sequence length exceeded: No prediction for '%s'.",
                    line.split()[0],
                )


def compute_metrics(p: EvalPrediction) -> Dict:
    preds_list, out_label_list = trim_and_convert_entity_ids(p.predictions, p.label_ids)
    return {
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }


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
    data_str = predmodel.set_relation("Finally\tO\nĠGroup\tB-EXPL_VAR\n", 1)
    dataset = predmodel.create_dataset(data_str)
    print(predmodel.do_predict(dataset))
