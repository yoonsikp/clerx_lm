from predict import PredictionModel
from transformers import AutoTokenizer, AutoModelForTokenClassification
from seqeval.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    performance_measure,
    classification_report,
)
import argparse
from glob import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument("--labels", required=True)
parser.add_argument("--model_name_or_path", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--test_relations", required=True)
parser.add_argument("--test_data", required=True)
parser.add_argument("--test_entity", required=True)

args = parser.parse_args()

final_args = {
    "max_seq_length": 512,
    "disable_tqdm": True,
    "output_dir": args.output_dir,
    "labels": args.labels,
    "model_name_or_path": args.model_name_or_path,
    "per_device_eval_batch_size": 1,
    "fp16": True,
}
predmodel = PredictionModel(final_args)
tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

color_dict = {
    "O": "\u001b[0m",
    "B-EXPL_VAR": "\u001b[38;5;" + "1" + "m",
    "I-EXPL_VAR": "\u001b[38;5;" + "124" + "m",
    "B-OUTCOME_VAR": "\u001b[38;5;" + "2" + "m",
    "I-OUTCOME_VAR": "\u001b[38;5;" + "28" + "m",
    "B-HR": "\u001b[38;5;" + "3" + "m",
    "I-HR": "\u001b[38;5;" + "3" + "m",
    "B-OR": "\u001b[38;5;" + "3" + "m",
    "I-OR": "\u001b[38;5;" + "3" + "m",
    "B-RR": "\u001b[38;5;" + "3" + "m",
    "I-RR": "\u001b[38;5;" + "3" + "m",
    "B-BASELINE": "\u001b[38;5;" + "21" + "m",
    "I-BASELINE": "\u001b[38;5;" + "4" + "m",
}


def pretty_iob(data_str):
    pretty_string = "\u001b[0m"
    for line in data_str.split("\n"):
        cols = line.split("\t")
        if len(cols) > 1:
            token = tokenizer.convert_tokens_to_string(cols[0])
            entity_label = cols[1]
            pretty_string += color_dict[entity_label]
            pretty_string += token
    return pretty_string


def print_debug_info():
    print("true_relations", true_relations)
    print("pred_relations", pred_relations)
    print("relation_accuracy", accuracy_score(true_relations, pred_relations))

    print("true_entity_ids", true_entity_ids)
    print("pred_entity_ids", pred_entity_ids)

    print("trimmed_pred_entity_labels", trimmed_pred_entity_labels)
    print("trimmed_true_entity_labels", trimmed_true_entity_labels)

    print("eval_loss", eval_loss)

    print(predmodel.generate_iob(trimmed_pred_entity_labels, data_str))

def test_entities():
    data_str = predmodel.set_relation(
        open(foldername + "./entity.txt", "r").read(), None
    )
    dataset = predmodel.create_dataset(data_str)
    (
        pred_relations,
        true_relations,
        pred_entity_ids,
        true_entity_ids,
        eval_loss,
    ) = predmodel.do_predict(dataset)
    (
        trimmed_pred_entity_labels,
        trimmed_true_entity_labels,
    ) = predmodel.trim_and_convert_entity_ids(pred_entity_ids, true_entity_ids)
    big_trimmed_true_entity_labels += trimmed_true_entity_labels
    big_trimmed_pred_entity_labels += trimmed_pred_entity_labels

    results = {
        "precision": precision_score(
            trimmed_true_entity_labels, trimmed_pred_entity_labels
        ),
        "recall": recall_score(trimmed_true_entity_labels, trimmed_pred_entity_labels),
        "f1": f1_score(trimmed_true_entity_labels, trimmed_pred_entity_labels),
        "performance_measure": performance_measure(
            trimmed_true_entity_labels, trimmed_pred_entity_labels
        ),
    }
    for metric, val in performance_measure(
        trimmed_true_entity_labels, trimmed_pred_entity_labels
    ).items():
        overall_results[metric] += val
    print("\nExample ID:", foldername)
    print("Metrics:", results)
    print(
        "Pred:",
        pretty_iob(predmodel.generate_iob(trimmed_pred_entity_labels, data_str)),
    )
    print(
        "True:",
        pretty_iob(predmodel.generate_iob(trimmed_true_entity_labels, data_str)),
    )

def test_predefined_entity_relations():
    concat_true_relations = []
    concat_pred_relations = []
    for relation_file in sorted(glob(os.path.join(foldername, "") + "/relation_*.txt")):
        data_str = open(relation_file, "r").read()
        dataset = predmodel.create_dataset(data_str)
        (
            pred_relations,
            true_relations,
            pred_entity_ids,
            true_entity_ids,
            eval_loss,
        ) = predmodel.do_predict(dataset)
        
        concat_true_relations.append(true_relations)
        concat_pred_relations.append(pred_relations)
        big_true_relations.append(true_relations)
        big_pred_relations.append(pred_relations)
    print("concat_true_relations", concat_true_relations)
    print("concat_pred_relations", concat_pred_relations)
    print("relation_accuracy", accuracy_score(concat_true_relations, concat_pred_relations))

def test_pipelined_entity_relations():
    raise NotImplementedError

def get_entities(tokens, labels):
    ratio_spans = []
    var_spans = []
    # stores entities
    # data structure: [(start_idx, end_idx)]
    # (inclusive, exclusive)
    ratio_spans  


overall_results = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
big_trimmed_true_entity_labels = []
big_trimmed_pred_entity_labels = []
big_true_relations = []
big_pred_relations = []

for foldername in sorted(glob(os.path.join(args.test_data, "") + "/*/")):
    if args.test_entities == "1":
        test_entities()
    if args.test_relations == "1":
        test_predefined_entity_relations()

if args.test_relations == "1":
    print("overall relation_accuracy", accuracy_score(big_true_relations, big_pred_relations))

if args.test_entities == "1":
    print(overall_results)
    results = {
        "precision": precision_score(
            big_trimmed_true_entity_labels, big_trimmed_pred_entity_labels
        ),
        "recall": recall_score(
            big_trimmed_true_entity_labels, big_trimmed_pred_entity_labels
        ),
        "f1": f1_score(big_trimmed_true_entity_labels, big_trimmed_pred_entity_labels),
        "performance_measure": performance_measure(
            big_trimmed_true_entity_labels, big_trimmed_pred_entity_labels
        ),
    }
    print(results)
    print(
        classification_report(
            big_trimmed_true_entity_labels, big_trimmed_pred_entity_labels
        )
    )
