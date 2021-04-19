#!/usr/bin/env bash
# stop on all errors
set -e

# death after SIGINT
trap 'exit' INT

export MAX_LENGTH=512
export BERT_MODEL=allenai/biomed_roberta_base
export BATCH_SIZE=4
export GRAD_ACCUM_SIZE=8
export NUM_EPOCHS=20
export CURRENT_DIR=${PWD}
export LEARNING_RATE=0.00005
export DATA_DIR=./3_iob_data/context/train_entity_relation

for i in {1..10}; do
    export SEED=$i
    export OUTPUT_DIR_NAME=biomed-roberta-base-final-$i
    export OUTPUT_DIR=./5_training/saved_models_er/${OUTPUT_DIR_NAME}
    mkdir -p $OUTPUT_DIR
    ./5_training/scripts/run.sh
done

