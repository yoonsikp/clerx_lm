from transformers import AutoTokenizer, AutoModelForTokenClassification
import random
from intervaltree import Interval, IntervalTree
import argparse
import os
import sys
import glob

parser = argparse.ArgumentParser(description='Read all tsv format and output IOB NER file')
parser.add_argument('--input', help='input folder', required=True)
parser.add_argument('--context', help='add context', action='store_true')
args = parser.parse_args()
args.input = os.path.join(args.input, '')
if not os.path.isdir(args.input):
    logging.error(args.input + " not a folder or doesn't exist")
    exit(1)

ID_TO_LABEL = {'1': 'EXPL_VAR', '2': 'OUTCOME_VAR', '3': 'HR', '4': 'OR', '5': 'RR', '6': 'BASELINE'}

tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

def add_context(text, title, first_sentence, line_idx):
    context = "" 
    context += title.strip('\n')
    if context[-1] == '.':
        context += ' '
    elif context[-2:-1] != '. ':
        context += '. '
    context += first_sentence.strip('\n')
    context_tokens = tokenizer.tokenize(context)
    builder = ''
    for idx, token in enumerate(context_tokens):
        if idx == len(context_tokens) - 1:
            builder += token + '\t' + 'CONTEXT_END' + '\n'
        else:
            builder += token + '\t' + 'CONTEXT' + '\n'
    return builder

final_output = ''
max_tokens = 0
for name in sorted(glob.glob(args.input + '*.*')):
    with open(name, 'r') as f:
        lines = f.readlines()
        title = lines[0]
        first_sentence = lines[1]
        text = lines[2:]
        for (line_idx, line) in enumerate(text):
            tree = IntervalTree()
            line = line.strip()
            columns = line.split('\t')
            tags = []
            if len(columns) > 1:
                tags = columns[1].split('.')
            if len(columns) == 1:
                continue
            final_output += '\n'
            if args.context:
                final_output += add_context(text, title, first_sentence, line_idx)
            for tag in tags:
                tag_vals = tag.split(',')
                span = (int(tag_vals[1]) - 1, int(tag_vals[2]) - 1)
                tree[span[0]:span[1]] = tag_vals
            text = columns[0]
            tokens = tokenizer.tokenize(text)
            max_tokens = max(len(tokens), max_tokens)
            counter = 0
            old_tag = None
            for token_idx, token in enumerate(tokens):
                iob_text = ''
                intersect = tree[counter:counter + len(token)]
                if len(intersect) > 1:
                    print("too many intersections", name, line)
                    exit(1)
                if len(intersect) == 1:
                    tag_interval = intersect.pop()
                    if old_tag != tag_interval.data:
                        iob_text = 'B-'
                    else:
                        iob_text = 'I-'
                    iob_text += ID_TO_LABEL[tag_interval.data[0]]
                    old_tag = tag_interval.data
                else:
                    iob_text = 'O'
                final_output += token + '\t' + iob_text + '\n'
                counter = len(tokenizer.convert_tokens_to_string(tokens[:token_idx + 1]))
            
print(final_output)
print(len(max(final_output.split("\n\n"), key=lambda x:len(x)).split('\n')), file=sys.stderr)
