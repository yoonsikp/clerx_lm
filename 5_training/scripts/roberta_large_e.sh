#!/usr/bin/env bash
# stop on all errors
set -e

# death after SIGINT
trap 'exit' INT

export MAX_LENGTH=512
export BERT_MODEL=roberta-large
export BATCH_SIZE=2
export GRAD_ACCUM_SIZE=8
export NUM_EPOCHS=60
export CURRENT_DIR=${PWD}
export LEARNING_RATE=0.00005
export DATA_DIR=./3_iob_data/context/train_entity

for i in {1..10}; do
    export SEED=$i
    export OUTPUT_DIR_NAME=roberta-large-final-$i
    export OUTPUT_DIR=./5_training/saved_models_e/${OUTPUT_DIR_NAME}
    mkdir -p $OUTPUT_DIR
    ./5_training/scripts/run.sh
done
