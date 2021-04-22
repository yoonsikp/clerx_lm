#!/usr/bin/env bash
# stop on all errors
set -e

# death after SIGINT
trap 'exit' INT

export SILENT_MODE=1

for i in 1; do
    export BERT_MODEL=./5_training/saved_models_e_nc/roberta-large-final-$i
    export OUTPUT_DIR_NAME=roberta-large-final-$i
    export OUTPUT_DIR=./6_testing/results/saved_models_e_nc/${OUTPUT_DIR_NAME}
    export TEST_RELATIONS=0
    export TEST_DATA=./4_intermediate_data/no_context/validation/
    mkdir -p $OUTPUT_DIR
    ./6_testing/scripts/run.sh
done
