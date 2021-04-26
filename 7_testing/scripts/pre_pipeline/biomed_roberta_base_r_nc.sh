#!/usr/bin/env bash
# stop on all errors
set -e

# death after SIGINT
trap 'exit' INT

export SILENT_MODE=1

for i in {1..10}; do
    export BERT_MODEL=./5_training/saved_models_r_nc/biomed-roberta-base-final-$i
    export OUTPUT_DIR_NAME=biomed-roberta-base-final-$i
    export OUTPUT_DIR=./7_testing/results/saved_models_r_nc/${OUTPUT_DIR_NAME}
    export TEST_RELATIONS=1
    export TEST_ENTITY=0
    export TEST_DATA=./4_intermediate_data/no_context/test/
    mkdir -p $OUTPUT_DIR
    ./7_testing/scripts/run.sh
done
