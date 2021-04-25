#!/usr/bin/env bash
# stop on all errors
set -e

# death after SIGINT
trap 'exit' INT

export SILENT_MODE=1

for i in 1; do
    export BERT_MODEL=./5_training/saved_models_e_nc/biomed-roberta-base-final-$i
    export OUTPUT_DIR_NAME=biomed-roberta-base-final-$i
    export OUTPUT_DIR=./6_validation/results/saved_models_e_nc/${OUTPUT_DIR_NAME}
    export TEST_RELATIONS=0
    export TEST_DATA=./4_intermediate_data/no_context/validation/
    mkdir -p $OUTPUT_DIR
    ./6_validation/scripts/run.sh
done
