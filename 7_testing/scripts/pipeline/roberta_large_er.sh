#!/usr/bin/env bash
# stop on all errors
set -e

# death after SIGINT
trap 'exit' INT

export SILENT_MODE=1

for i in {1..10}; do
    export BERT_MODEL=./5_training/saved_models_er/roberta-large-final-$i
    export MODEL_DIR_NAME=roberta-large-final-$i
    export ENTITY_DATA=./7_testing/results/saved_models_er/${MODEL_DIR_NAME}
    export TEST_DATA=./4_intermediate_data/context/test/
    ./7_testing/scripts/pipeline.sh
done
