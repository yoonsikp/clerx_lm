# CLERx-LM

### Install Requirements
Tested on Python 3.8!
```
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

### Extract tar.gz files:

```
gunzip -dk ./0_pubmed_data/*.gz
```

### Create naive filtered TSV files:

```
for f in ./0_pubmed_data/*.xml; do python3 ./scripts/extract_articles.py --input "$f" --output ./1_articles; done
```

### Annotate articles manually
See appendix

### Run TSV validation tool

```
for f in ./2_annotated_articles/*; do python3 ./scripts/validate_tsv.py --input "$f" |grep "errors"; done
```

### Generate IOB files

```
./scripts/create_iob.sh
```

### Run Training Scripts
```
# with context
./5_training/scripts/biomed_roberta_base_e.sh
./5_training/scripts/biomed_roberta_base_er.sh
./5_training/scripts/roberta_base_e.sh
./5_training/scripts/roberta_base_er.sh
./5_training/scripts/roberta_large_e.sh
./5_training/scripts/roberta_large_er.sh

# without context
./5_training/scripts/biomed_roberta_base_e_nc.sh
./5_training/scripts/biomed_roberta_base_er_nc.sh
./5_training/scripts/roberta_base_e_nc.sh
./5_training/scripts/roberta_base_er_nc.sh
./5_training/scripts/roberta_large_e_nc.sh
./5_training/scripts/roberta_large_er_nc.sh
```

### Generate Intermediate Data files
```
./scripts/create_intermediate.sh
```

### Run Testing Scripts

