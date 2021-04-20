from predict import PredictionModel
from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score
import argparse

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
data_str = predmodel.set_relation(open(args.test_data + "./entity.txt", 'r').read(), None)
dataset = predmodel.create_dataset(data_str)
pred_relations, true_relations, pred_entity_ids, true_entity_ids, eval_loss = predmodel.do_predict(dataset)
trimmed_pred_entity_labels, trimmed_true_entity_labels = predmodel.trim_and_convert_entity_ids(
    pred_entity_ids, true_entity_ids
)

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
    
print_debug_info()
results = {
            "precision": precision_score(trimmed_true_entity_labels, trimmed_pred_entity_labels),
            "recall": recall_score(trimmed_true_entity_labels, trimmed_pred_entity_labels),
            "f1": f1_score(trimmed_true_entity_labels, trimmed_pred_entity_labels),
        }
print(results)

if args.test_relations == "1":
    print("testing relations TODO")
