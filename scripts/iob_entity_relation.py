from transformers import AutoTokenizer, AutoModelForTokenClassification
import random
from intervaltree import Interval, IntervalTree
import itertools
import argparse
import os
import sys
import glob

parser = argparse.ArgumentParser(description='Read all tsv format and output IOB NER file, with pairwise relations')
parser.add_argument('--input', help='input folder', required=True)
parser.add_argument('--context', help='add context', action='store_true')
args = parser.parse_args()
args.input = os.path.join(args.input, '')
if not os.path.isdir(args.input):
    logging.error(args.input + " not a folder or doesn't exist")
    exit(1)

ID_TO_LABEL = {'1': 'EXPL_VAR', '2': 'OUTCOME_VAR', '3': 'HR', '4': 'OR', '5': 'RR', '6': 'BASELINE'}
TOKEN_1 = '^'
TOKEN_2 = '$'

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
            if len(columns) != 3:
                continue

            tags = list(enumerate(columns[1].split('.')))
            true_pairs = set() # [(int, int), (int, int)]
            for str_pair in columns[2].split('.'):
                int_pair = tuple(sorted([int(tag_idx) - 1 for tag_idx in str_pair.split(',')]))
                true_pairs.add(int_pair)
            false_pairs = set()

            ratio_tags = []
            variable_tags = []
            for tag in tags:
                if tag[1].split(',')[0] in ['3', '4', '5']:
                    ratio_tags.append(tag)
                else:
                    variable_tags.append(tag)

            for ratio_tag, variable_tag in itertools.product(ratio_tags, variable_tags):
                int_pair = tuple(sorted([ratio_tag[0], variable_tag[0]]))
                if int_pair not in true_pairs:
                    false_pairs.add(int_pair)

            all_pairs = [] # [((int, int), str), ((int, int), str)]
            for pair in true_pairs:
                all_pairs.append((pair, '1'))
            for pair in false_pairs:
                all_pairs.append((pair, '0'))
            for pair, relation_label in all_pairs:
                # final_output += '\n'
                final_output += '<s>' + '\t' + 'BOS_' + relation_label + '\n'
                if args.context:
                    final_output += add_context(text, title, first_sentence, line_idx)
                for tag_idx, tag in tags:
                    data = tag.split(',') + [tag_idx]
                    span = (int(data[1]) - 1, int(data[2]) - 1)
                    tree[span[0]:span[1]] = data # [str, str, str, int]
                text = columns[0]
                tokens = tokenizer.tokenize(text)
                max_tokens = max(len(tokens), max_tokens)
                cur_pos = 0
                previous_tag_data = None
                tag1_seen = False
                tag1_closed = False
                tag2_seen = False
                tag2_closed = False
                cur_token = ''
                next_pair = 0
                for token in tokens:
                    iob_text = ''
                    intersect = tree[cur_pos:cur_pos + len(token)]
                    if len(intersect) > 1:
                        print("too many intersections", name, line)
                        exit(1)
                    if len(intersect) == 1:
                        tag_interval = intersect.pop()
                        if previous_tag_data is not tag_interval.data:
                            tag_idx = tag_interval.data[3]
                            
                            if (pair[0] == tag_idx or pair[1] == tag_idx) and not tag1_seen:
                                tag1_seen = True
                                next_pair = pair[0] if pair[1] == tag_idx else pair[1]
                                cur_token = TOKEN_1 if tag_interval.data[0] in ['3','4','5'] else TOKEN_2
                                final_output += cur_token + '\t' + 'CONTEXT' + '\n'
                            if next_pair == tag_idx and tag1_closed and not tag2_seen:
                                tag2_seen = True
                                cur_token = TOKEN_1 if tag_interval.data[0] in ['3','4','5'] else TOKEN_2
                                final_output += cur_token + '\t' + 'CONTEXT' + '\n'
                            iob_text = 'B-'
                        else:
                            iob_text = 'I-'
                        iob_text += ID_TO_LABEL[tag_interval.data[0]]
                        previous_tag_data = tag_interval.data
                    else:
                        if tag1_seen and not tag1_closed:
                            tag1_closed = True
                            final_output += cur_token + '\t' + 'CONTEXT' + '\n'
                        if tag2_seen and not tag2_closed:
                            tag2_closed = True
                            final_output += cur_token + '\t' + 'CONTEXT' + '\n'
                        iob_text = 'O'
                    final_output += token + '\t' + iob_text + '\n'
                    cur_pos += len(token)
                final_output += '\n'
print(final_output.strip())
print(len(max(final_output.split("\n\n"), key=lambda x:len(x)).split('\n')), file=sys.stderr)
