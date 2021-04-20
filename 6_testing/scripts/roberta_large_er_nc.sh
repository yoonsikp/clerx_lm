#!/usr/bin/env bash
# stop on all errors
set -e

# death after SIGINT
trap 'exit' INT

export SILENT_MODE=1

for i in 9; do
    export BERT_MODEL=./5_training/saved_models_er_nc/roberta-large-final-$i
    export OUTPUT_DIR_NAME=roberta-large-final-$i
    export OUTPUT_DIR=./6_testing/results/saved_models_er_nc/${OUTPUT_DIR_NAME}
    export TEST_RELATIONS=1
    export TEST_DATA=./4_intermediate_data/no_context/validation/
    mkdir -p $OUTPUT_DIR
    ./6_testing/scripts/run.sh
done