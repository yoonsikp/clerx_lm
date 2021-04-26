#!/usr/bin/env bash
# stop on all errors
set -e

# death after SIGINT
trap 'exit' INT

export SILENT_MODE=1

for i in {1..10}; do
    export BERT_MODEL=./5_training/saved_models_er_nc/biomed-roberta-base-final-$i
    export MODEL_DIR_NAME=biomed-roberta-base-final-$i
    export ENTITY_DATA=./6_validation/results/saved_models_er_nc/${MODEL_DIR_NAME}
    export TEST_DATA=./4_intermediate_data/no_context/validation/
    ./6_validation/scripts/pipeline.sh
done
