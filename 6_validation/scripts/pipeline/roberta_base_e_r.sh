#!/usr/bin/env bash
# stop on all errors
set -e

# death after SIGINT
trap 'exit' INT

export SILENT_MODE=1

for i in {1..10}; do
    export BERT_MODEL=./5_training/saved_models_r/roberta-base-final-$i
    export MODEL_DIR_NAME=roberta-base-final-$i
    export ENTITY_DATA=./6_validation/results/saved_models_e/${MODEL_DIR_NAME}
    export TEST_DATA=./4_intermediate_data/context/validation/
    ./6_validation/scripts/pipeline.sh
done
