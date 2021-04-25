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
from sklearn.metrics import confusion_matrix
import argparse
from glob import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument("--labels", required=True)
parser.add_argument("--model_name_or_path", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--test_relations", required=True)
parser.add_argument("--data_dir", required=True)
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

REL_SUMMARY_FILENAME = "./6_validation/relation_stats.csv"
ENT_SUMMARY_FILENAME = "./6_validation/entity_stats.csv"
split_model = args.model_name_or_path.split("/")
model_name = "-".join(split_model[-1].split("-")[:-1])
seed = split_model[-1].split("-")[-1]
model_type = split_model[-2][13:]
if "_nc" in model_type:
    model_type = model_type[:-3]
model_context = split_model[-2][-3:] == "_nc"


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
    print("eval_loss", eval_loss)
    print(predmodel.generate_iob(trim_pred_ent_labels, data_str))


def get_entities(tokens, labels):
    ratio_spans = []
    var_spans = []
    # stores entities
    # data structure: [(start_idx, end_idx)]
    # (inclusive, exclusive)
    ratio_spans


def append_entity_stats(text):
    if not os.path.isfile(ENT_SUMMARY_FILENAME):
        with open(os.path.join(ENT_SUMMARY_FILENAME), "x") as csvfile:
            csvfile.write("model,no_context,type,seed,")
            csvfile.write("label_name,prec,recall,f1,support\n")
    with open(os.path.join(ENT_SUMMARY_FILENAME), "a") as csvfile:
        csvfile.write(text)


def append_relation_stats(text):
    if not os.path.isfile(REL_SUMMARY_FILENAME):
        with open(os.path.join(REL_SUMMARY_FILENAME), "x") as csvfile:
            csvfile.write("model,no_context,type,seed,sentence_id,")
            csvfile.write("mcc,acc,prec,recall,f1\n")
    with open(os.path.join(REL_SUMMARY_FILENAME), "a") as csvfile:
        csvfile.write(text)


def get_relation_stats(true, pred):
    tn, fp, fn, tp = confusion_matrix(true, pred, labels=[0, 1]).ravel()
    ret_dict = {}
    ret_dict["acc"] = (tp + tn) / (tp + tn + fp + fn)
    ret_dict["prec"] = tp / (tp + fp) if (tp + fp > 0) else 0
    ret_dict["recall"] = tp / (tp + fn) if (tp + fn > 0) else 0
    ret_dict["f1"] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn > 0) else 0
    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
        ret_dict["mcc"] = 0
    else:
        ret_dict["mcc"] = (tp * tn - fp * fn) / (
            ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        )
    return ret_dict


all_trim_true_ent_labels = []
all_trim_pred_ent_labels = []
all_true_rels = []
all_pred_rels = []

rel_sentw_mcc = 0
rel_sentw_acc = 0
rel_sentw_prec = 0
rel_sentw_recall = 0
rel_sentw_f1 = 0
sentw_count = 0

for foldername in sorted(glob(os.path.join(args.data_dir, "") + "/*/")):
    print("\nExample ID:", foldername)
    if args.test_entity == "1":
        data_str = predmodel.set_relation(
            open(foldername + "./entity.txt", "r").read(), None
        )
        dataset = predmodel.create_dataset(data_str)
        (
            pred_rels,
            true_rels,
            pred_entity_ids,
            true_entity_ids,
            eval_loss,
        ) = predmodel.do_predict(dataset)
        (
            trim_pred_ent_labels,
            trim_true_ent_labels,
        ) = predmodel.trim_and_convert_entity_ids(pred_entity_ids, true_entity_ids)
        all_trim_true_ent_labels += trim_true_ent_labels
        all_trim_pred_ent_labels += trim_pred_ent_labels

        results = {
            "precision": precision_score(trim_true_ent_labels, trim_pred_ent_labels),
            "recall": recall_score(trim_true_ent_labels, trim_pred_ent_labels),
            "f1": f1_score(trim_true_ent_labels, trim_pred_ent_labels),
            "performance_measure": performance_measure(
                trim_true_ent_labels, trim_pred_ent_labels
            ),
        }

        print("Metrics:", results)
        print(
            "Pred:",
            pretty_iob(predmodel.generate_iob(trim_pred_ent_labels, data_str)),
        )
        print(
            "True:",
            pretty_iob(predmodel.generate_iob(trim_true_ent_labels, data_str)),
        )
    if args.test_relations == "1":
        concat_true_rels = []
        concat_pred_rels = []
        for relation_file in sorted(
            glob(os.path.join(foldername, "") + "/relation_*.txt")
        ):
            data_str = open(relation_file, "r").read()
            dataset = predmodel.create_dataset(data_str)
            (
                pred_rels,
                true_rels,
                pred_entity_ids,
                true_entity_ids,
                eval_loss,
            ) = predmodel.do_predict(dataset)

            concat_true_rels += list(true_rels)
            concat_pred_rels += list(pred_rels)
            all_true_rels += list(true_rels)
            all_pred_rels += list(pred_rels)
        sentw_count += 1
        rel_stats = get_relation_stats(concat_true_rels, concat_pred_rels)
        rel_sentw_mcc += rel_stats["mcc"]
        rel_sentw_acc += rel_stats["acc"]
        rel_sentw_prec += rel_stats["prec"]
        rel_sentw_recall += rel_stats["recall"]
        rel_sentw_f1 += rel_stats["f1"]

        append_relation_stats(
            f"{model_name},{str(model_context)},{model_type},{seed},{foldername.rstrip('/').split('/')[-1]},"
        )
        append_relation_stats(
            f"{rel_stats['mcc']},{rel_stats['acc']},{rel_stats['prec']},{rel_stats['recall']},{rel_stats['f1']}\n"
        )

if args.test_relations == "1":
    rel_stats = get_relation_stats(all_true_rels, all_pred_rels)

    rel_sentw_mcc /= sentw_count
    rel_sentw_acc /= sentw_count
    rel_sentw_prec /= sentw_count
    rel_sentw_recall /= sentw_count
    rel_sentw_f1 /= sentw_count

    append_relation_stats(
        f"{model_name},{str(model_context)},{model_type},{seed},AGGREGATE,"
    )
    append_relation_stats(
        f"{rel_stats['mcc']},{rel_stats['acc']},{rel_stats['prec']},{rel_stats['recall']},{rel_stats['f1']}\n"
    )
    append_relation_stats(
        f"{model_name},{str(model_context)},{model_type},{seed},SENTW_AVG,"
    )
    append_relation_stats(
        f"{rel_sentw_mcc},{rel_sentw_acc},{rel_sentw_prec},{rel_sentw_recall},{rel_sentw_f1}\n"
    )

if args.test_entity == "1":
    print(
        "performance_measure",
        performance_measure(all_trim_true_ent_labels, all_trim_pred_ent_labels),
    )
    print(classification_report(all_trim_true_ent_labels, all_trim_pred_ent_labels))
    entity_stats = classification_report(
        all_trim_true_ent_labels,
        all_trim_pred_ent_labels,
        output_dict=True,
    )

    for label, stats in entity_stats.items():
        append_entity_stats(f"{model_name},{str(model_context)},{model_type},{seed},")
        append_entity_stats(
            f"{label},{stats['precision']},{stats['recall']},{stats['f1-score']},{stats['support']}\n"
        )
