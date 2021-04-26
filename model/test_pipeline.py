from predict import PredictionModel
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.metrics import confusion_matrix
import argparse
from glob import glob
import os
import itertools
from utils_ner import get_labels

parser = argparse.ArgumentParser()
parser.add_argument("--labels", required=True)
parser.add_argument("--model_name_or_path", required=True)
parser.add_argument("--test_data_dir", required=True)
parser.add_argument("--entity_data_dir", required=True)

args = parser.parse_args()

final_args = {
    "max_seq_length": 512,
    "disable_tqdm": True,
    "output_dir": "/tmp",
    "labels": args.labels,
    "model_name_or_path": args.model_name_or_path,
    "per_device_eval_batch_size": 1,
    "fp16": True,
}
predmodel = PredictionModel(final_args)
tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
labels = get_labels(args.labels)
label_map = {label: i for i, label in enumerate(labels)}

JER_SUMMARY_FILENAME = "./6_validation/jer_stats.csv"
split_model = args.model_name_or_path.split("/")
model_name = "-".join(split_model[-1].split("-")[:-1])
seed = split_model[-1].split("-")[-1]
model_type = split_model[-2][13:]
if "_nc" in model_type:
    model_type = model_type[:-3]
model_context = split_model[-2][-3:] == "_nc"

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
        if "\tBOS_" in line or "\tCONTEXT" in line:
            continue
        cols = line.split("\t")
        if len(cols) > 1:
            token = tokenizer.convert_tokens_to_string(cols[0])
            entity_label = cols[1]
            pretty_string += color_dict[entity_label]
            pretty_string += token
    return pretty_string

def append_jer_stats(text):
    if not os.path.isfile(JER_SUMMARY_FILENAME):
        with open(os.path.join(JER_SUMMARY_FILENAME), "x") as csvfile:
            csvfile.write("model,no_context,type,seed,sentence_id,")
            csvfile.write("jaccard_similarity,intersect,union\n")
    with open(os.path.join(JER_SUMMARY_FILENAME), "a") as csvfile:
        csvfile.write(text)

def extract_rel_pair(data_str):
    if 'BOS_0' in data_str:
        true_pair = False
        return None
    elif 'BOS_1' in data_str:
        true_pair = True
    else:
        raise Exception("Must contain relation")
    
    if '$' not in data_str or '^' not in data_str:
        return None
    if "CONTEXT_END" in data_str:
        data_str = data_str.split('CONTEXT_END\n')[1]
    
    lines = data_str.split("\n")
    if '\tBOS_' in lines[0]:
        lines.pop(0)
    ratio_data = {'label': None, 'token_start': None, 'token_end': None}
    entity_data = {'label': None, 'token_start': None, 'token_end': None}
    token_num = 0
    for line in lines:
        if not (line.startswith("-DOCSTART-") or line == "" or line == "\n"):
            splits = line.split("\t")
            token = splits[0]
            label = splits[1]
            if label == 'CONTEXT' and token == "$":
                if entity_data['token_start'] is not None:
                    entity_data['token_end'] = token_num - 1
                else:
                    entity_data['token_start'] = token_num
                continue
            if label == 'CONTEXT' and token == "^":
                if ratio_data['token_start'] is not None:
                    ratio_data['token_end'] = token_num - 1
                else:
                    ratio_data['token_start'] = token_num
                continue
        
            if entity_data['token_start'] is not None and entity_data['label'] is None:
                entity_data['label'] = label[2:]
            elif ratio_data['token_start'] is not None and ratio_data['label'] is None:
                ratio_data['label'] = label[2:]
            token_num += 1
    return ((ratio_data['label'],ratio_data['token_start'],ratio_data['token_end']),(entity_data['label'],entity_data['token_start'],entity_data['token_end']))

def extract_entities(data_str):
    if "CONTEXT_END" in data_str:
        data_str = data_str.split('CONTEXT_END\n')[1]
    lines = data_str.split("\n")
    spans = []
    started = False
    token_num = 0
    previous_label = ""
    for line in lines:
        if not (line.startswith("-DOCSTART-") or line == "" or line == "\n"):
            splits = line.split("\t")
            token = splits[0]
            label = splits[1]
            if label[:2] == "B-":
                if len(spans) != 0 and spans[-1]['token_end'] is None:
                    spans[-1]['token_end'] = token_num - 1
                spans.append({'label': label[2:], 'token_start': token_num, 'token_end': None})
                previous_label = label[2:]
            elif label[:2] == "I-":
                if previous_label != label[2:]:
                    if len(spans) != 0 and spans[-1]['token_end'] is None:
                        spans[-1]['token_end'] = token_num - 1
                    spans.append({'label': label[2:], 'token_start': token_num, 'token_end': None})
                previous_label = label[2:]
            elif label[:2] == "O":
                if len(spans) != 0 and spans[-1]['token_end'] is None:
                    spans[-1]['token_end'] = token_num - 1
                previous_label = ""
            token_num += 1

    if len(spans) != 0 and spans[-1]['token_end'] is None:
        spans[-1]['token_end'] = token_num - 1

    ratio_spans = []
    var_spans = []
    for span in spans:
        if span['label'] in ["HR", "RR", "OR"]:
            ratio_spans.append((span['label'], span['token_start'], span['token_end']))
        else:
            var_spans.append((span['label'], span['token_start'], span['token_end']))
    return ratio_spans, var_spans


total_intersect = 0
total_union = 0


for foldername in sorted(glob(os.path.join(args.test_data_dir, "") + "/*/")):
    sentence_id = foldername.rstrip('/').split('/')[-1]
    print("\nExample ID:", foldername)

    true_pairs = set()
    pred_pairs = set()
    first_rel = True
    for relation_file in sorted(
        glob(os.path.join(foldername, "") + "/relation_*.txt")
    ):
        data_str = open(relation_file, "r").read()
        pair = extract_rel_pair(data_str)
        if pair is not None:
            true_pairs.add(pair)
        if first_rel:
            print(
                "True:",
                pretty_iob(data_str),
            )
            first_rel = False
    with open(os.path.join(args.entity_data_dir, sentence_id, "entity.txt"), 'r') as entity_file:
        data_str = entity_file.read()
        ratio_spans, var_spans = extract_entities(data_str)
        print("extracted_ent", ratio_spans, var_spans)
        print(
            "Pred:",
            pretty_iob(data_str),
        )
        print("True:", true_pairs)
        for ratio_span, var_span in itertools.product(ratio_spans, var_spans):
            if "CONTEXT_END" in data_str:
                context = data_str.split('CONTEXT_END\n')[0] + 'CONTEXT_END\n'
                token_lines =  data_str.split('CONTEXT_END\n')[1].split("\n")
            else:
                context = ""
                token_lines = data_str.split("\n")
            if ratio_span[1] < var_span[1]:
                token_lines.insert(ratio_span[1], "^\tCONTEXT")
                token_lines.insert(ratio_span[2] + 1, "^\tCONTEXT")
                token_lines.insert(var_span[1] + 2, "$\tCONTEXT")
                token_lines.insert(var_span[2] + 3, "$\tCONTEXT")
            else:
                token_lines.insert(var_span[1], "$\tCONTEXT")
                token_lines.insert(var_span[2] + 1, "$\tCONTEXT")
                token_lines.insert(ratio_span[1] + 2, "^\tCONTEXT")
                token_lines.insert(ratio_span[2] + 3, "^\tCONTEXT")
            new_data_str = context + "\n".join(token_lines)
            dataset = predmodel.create_dataset(new_data_str)
            pred_rels, _, _, _, _ = predmodel.do_predict(dataset)
            if pred_rels[0] == 1:
                pred_pairs.add((ratio_span, var_span))
        print("Pred:", pred_pairs)
    intersect = len(true_pairs.intersection(pred_pairs))
    union = len(true_pairs.union(pred_pairs))

    total_intersect += intersect
    total_union += union

    jaccard_sim = intersect/union
    append_jer_stats(
        f"{model_name},{str(model_context)},{model_type},{seed},{sentence_id},"
    )
    append_jer_stats(
        f"{jaccard_sim},{intersect},{union}\n"
    )


append_jer_stats(
    f"{model_name},{str(model_context)},{model_type},{seed},AGGREGATE,"
)
append_jer_stats(
    f"{total_intersect/total_union},{total_intersect},{total_union}\n"
)
