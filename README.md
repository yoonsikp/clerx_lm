# CLERx-LM

### Extract tar.gz files:

```
gunzip -dk ./0_pubmed_data/*.gz
```

### Create naive filtered TSV files:

```
for f in ./0_pubmed_data/*.xml; do python3 ./scripts/extract_articles.py --input "$f" --output ./1_articles; done

```

### Annotate articles manually

### Run TSV validation tool

```
for f in ./2_annotated_articles/*; do python3 ./scripts/validate_tsv.py --input "$f"; done
```

### Generate IOB files

```
./scripts/create_iob.sh
```

### Run Training Scripts

### Run Testing Scripts

