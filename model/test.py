from predict import PredictionModel
from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score, performance_measure, classification_report
import argparse
from glob import glob
import os 

parser = argparse.ArgumentParser()
parser.add_argument("--labels", required=True)
parser.add_argument("--model_name_or_path", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--test_relations", required=True)
parser.add_argument("--test_data", required=True)

args = parser.parse_args()

final_args = {
    "max_seq_length": 512,
    "output_dir": args.output_dir,
    "labels": args.labels,
    "model_name_or_path": args.model_name_or_path,
    "per_device_eval_batch_size": 1,
    "fp16": True,
}
predmodel = PredictionModel(final_args)

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

overall_results = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
big_trimmed_true_entity_labels = []
big_trimmed_pred_entity_labels = []
for foldername in sorted(glob(os.path.join(args.test_data, "") + "/*/")):
    print(foldername)
    data_str = predmodel.set_relation(open(foldername + "./entity.txt", 'r').read(), None)
    dataset = predmodel.create_dataset(data_str)
    pred_relations, true_relations, pred_entity_ids, true_entity_ids, eval_loss = predmodel.do_predict(dataset)
    trimmed_pred_entity_labels, trimmed_true_entity_labels = predmodel.trim_and_convert_entity_ids(
        pred_entity_ids, true_entity_ids
    )
    big_trimmed_true_entity_labels += trimmed_true_entity_labels
    big_trimmed_pred_entity_labels += trimmed_pred_entity_labels

    results = {
                "precision": precision_score(trimmed_true_entity_labels, trimmed_pred_entity_labels),
                "recall": recall_score(trimmed_true_entity_labels, trimmed_pred_entity_labels),
                "f1": f1_score(trimmed_true_entity_labels, trimmed_pred_entity_labels),
                "performance_measure": performance_measure(trimmed_true_entity_labels, trimmed_pred_entity_labels),
            }
    for metric, val in performance_measure(trimmed_true_entity_labels, trimmed_pred_entity_labels).items():
        overall_results[metric] += val
    print(results)

    if args.test_relations == "1":
        print("testing relations TODO")
print(overall_results)

results = {
            "precision": precision_score(big_trimmed_true_entity_labels, big_trimmed_pred_entity_labels),
            "recall": recall_score(big_trimmed_true_entity_labels, big_trimmed_pred_entity_labels),
            "f1": f1_score(big_trimmed_true_entity_labels, big_trimmed_pred_entity_labels),
            "performance_measure": performance_measure(big_trimmed_true_entity_labels, big_trimmed_pred_entity_labels),
        }

print(results)
print(classification_report(big_trimmed_true_entity_labels, big_trimmed_pred_entity_labels))
