import xml.etree.ElementTree as ET
import re
import os
import argparse
import nltk
nltk.download('punkt')

from nltk.tokenize import sent_tokenize
import unicodedata

parser = argparse.ArgumentParser(description='Create Articles from Pubmed XML')
parser.add_argument('--input', help='input XML', required=True)
parser.add_argument('--output', help='output folder', required=True)
args = parser.parse_args()

args.output = os.path.join(args.output, '')
if not os.path.isdir(args.output):
    print(args.output + " not a folder or doesn't exist")
    exit(1)

or_regex = re.compile('.*(odds ratio|Odds Ratio|[^">FVPTC]OR[^FA<]).*')
rr_regex = re.compile('.*(risk ratio|relative risk|Relative Risk|Relative risk|relative Risk|risk Ratio|Risk ratio|Risk Ratio|[^">FVPTC]RR[^FA<]).*')
hr_regex = re.compile('.*((Hazard ratio)|(hazard ratio)|(Hazard Ratio)|([^">FVPTCI]HR[^FRAV<])).*')

unicode_replace_dict = {" ": " ", " ": " ", "•": " ", "∼": "~", "’": "'", "＞": ">", "℃": "°C", '“': '"', '”': '"', 
                        "⩾": "≥", "⩽": "≤", "−": "-", "≧": '≥', '≦': '≤', ' ': " ", ' ': " "}

def replace_unicode(orig_str):
    for before, after in unicode_replace_dict.items():
        orig_str = orig_str.replace(before, after)
    return orig_str

def write_article(pmid, title, sentences, suffix):
    with open(args.output + pmid + suffix + '.tsv', 'w') as f:
        f.write(title + '\n')
        f.write('\n'.join(sentences) + '\n')

stats = {(0,0,0): [0, "_none"], (0,0,1): [0, "_rr"], (0,1,0): [0, "_hr"], (1,0,0): [0, "_or"],
         (0,1,1): [0, "_hr_rr"], (1,1,0): [0, "_or_hr"], (1,0,1): [0, "_or_rr"], (1,1,1): [0, "_or_hr_rr"]}

tree = ET.parse(args.input)
root = tree.getroot()

for article in root:
    abstract = None
    pmid = article.find('MedlineCitation').find('PMID').text
    article_ = article.find('MedlineCitation').find('Article')
    if article_ is not None:
        abstract = article_.find('Abstract')
        try:
            title = ''.join(article_.find('ArticleTitle').itertext())
            title = replace_unicode(title)
        except(TypeError):
            continue
    if title is None:
        continue
    full_abstract = None
    if abstract is not None:
        try:
            full_abstract = ' '.join([''.join(e.itertext()) for e in abstract.findall('AbstractText')])
            full_abstract = replace_unicode(full_abstract)
        except(TypeError):
            continue
    if full_abstract is not None:
        or_match = int(bool(or_regex.match(full_abstract)))
        hr_match = int(bool(hr_regex.match(full_abstract)))
        rr_match = int(bool(rr_regex.match(full_abstract)))
        stats[(or_match, hr_match, rr_match)][0] += 1
        if not (rr_match or or_match or hr_match):
            continue
        sentences = sent_tokenize(full_abstract)
        write_article(pmid, title, sentences, stats[(or_match, hr_match, rr_match)][1])

print(stats.values())
