import argparse
import os
parser = argparse.ArgumentParser(description='Validate all tsv format')
parser.add_argument('--input', help='input folder', required=True)
args = parser.parse_args()

args.input = os.path.join(args.input, '')
if not os.path.isdir(args.input):
    raise Exception(args.input + " not a folder or doesn't exist")

import glob

def represents_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False
num_errors = 0

def error():
    global num_errors
    num_errors += 1
    print("error in file: " + name + ', line: ' + line)
    
for name in glob.glob(args.input + '*.*'):
    with open(name, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            sections = line.split('\t')
            if len(sections) > 3:
                error()
                print('too many tab chars')
                continue
            tag_list = []
            raw_tag_list = []
            tag_count = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0}
            pair_dict = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0}
            if len(sections) > 1:
                tags = sections[1].split('.')
                raw_tag_list = tags
                print('\n' + name)
                print(sections[0] + '\n')
                for tag_idx, tag in enumerate(tags):
                    if len(tag.split(',')) != 3:
                        error()
                        print('wrong tag tuple')
                        continue
                    tagvals = tag.split(',')
                    if not represents_int(tagvals[0]) or not represents_int(tagvals[1]) or not represents_int(tagvals[2]):
                        error()
                        print('not ints in tag tuple')
                    if tagvals[0] not in ['1', '2', '3', '4', '5', '6']:
                        error()
                        print('wrong tag type')
                    tag_count[tagvals[0]] += 1
                    if int(tagvals[1]) - 1 > len(sections[0]) or int(tagvals[2]) - 1 > len(sections[0]):
                        error()
                        print('overflow span')
                    if sections[0][int(tagvals[1]) - 1:int(tagvals[2]) - 1].startswith(' ') or sections[0][int(tagvals[1]) - 1:int(tagvals[2]) - 1].endswith(' '):
                        error()
                        print('extra space')
                    print(tag_idx + 1, 'type:' + tagvals[0], sections[0][int(tagvals[1]) - 1:int(tagvals[2]) - 1])
                    tag_list.append(sections[0][int(tagvals[1]) - 1:int(tagvals[2]) - 1])
                    
            
            if len(sections) > 2:
                print('')
                pairs = sections[2].split('.')
                for pair in pairs:
                    if len(pair.split(',')) != 2:
                        error()
                        print('too many in pair')
                        continue
                    pairvals = pair.split(',')
                    if not represents_int(pairvals[0]) or not represents_int(pairvals[1]):
                        error()
                        print('not ints in pair')
                        continue
                    if int(pairvals[0]) > len(tags) or int(pairvals[1]) > len(tags):
                        error()
                        print('pair val too large')
                    if int(pairvals[0]) < 1 or int(pairvals[1]) < 1:
                        error()
                        print('pair val too large')
                    if int(pairvals[0]) == int(pairvals[1]):
                        error()
                        print('cannot be pair with self')
                    pair_dict[raw_tag_list[int(pairvals[0])-1].split(',')[0]] += 1
                    pair_dict[raw_tag_list[int(pairvals[1])-1].split(',')[0]] += 1
                   
                        
                    print(pair, tag_list[int(pairvals[0])-1],tag_list[int(pairvals[1])-1])
            if pair_dict['1'] == 0 and tag_count['1'] > 0:
                print("suggested for Explanatory:")
                builder = ''
                for tag_idx, tag in enumerate(raw_tag_list):
                    if tag.split(',')[0] == '1':
                        for tag_idx_2, tag_2 in enumerate(raw_tag_list):
                            if tag_2.split(',')[0] in ['3','4','5']:
                                builder += ('.' + str(tag_idx_2 + 1) + ',' +str(tag_idx + 1))
                print(builder)
            if pair_dict['2'] == 0 and tag_count['2'] > 0:
                print("suggested for Outcome:")
                builder = ''
                for tag_idx, tag in enumerate(raw_tag_list):
                    if tag.split(',')[0] == '2':
                        for tag_idx_2, tag_2 in enumerate(raw_tag_list):
                            if tag_2.split(',')[0] in ['3','4','5']:
                                builder += ('.' + str(tag_idx_2 + 1) + ',' +str(tag_idx + 1))
                print(builder)
            if pair_dict['6'] == 0 and tag_count['6'] > 0:
                print("suggested for Baseline:")
                builder = ''
                for tag_idx, tag in enumerate(raw_tag_list):
                    if tag.split(',')[0] == '6':
                        for tag_idx_2, tag_2 in enumerate(raw_tag_list):
                            if tag_2.split(',')[0] in ['3','4','5']:
                                builder += ('.' + str(tag_idx_2 + 1) + ',' +str(tag_idx + 1))
                print(builder)
print("errors: ", num_errors)
