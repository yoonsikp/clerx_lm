#!/usr/bin/env bash
# stop on all errors
set -e

# death after SIGINT
trap 'exit' INT

export SILENT_MODE=1

for i in 1; do
    export BERT_MODEL=./5_training/saved_models_er/roberta-base-final-$i
    export OUTPUT_DIR_NAME=roberta-base-final-$i
    export OUTPUT_DIR=./6_validation/results/saved_models_er/${OUTPUT_DIR_NAME}
    export TEST_RELATIONS=1
    export TEST_DATA=./4_intermediate_data/context/validation/
    mkdir -p $OUTPUT_DIR
    ./6_validation/scripts/run.sh
done
