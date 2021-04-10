python3 ./4_model/run_ner.py \
--data_dir $DATA_DIR \
--labels ./3_iob_data/labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_device_train_batch_size $BATCH_SIZE \
--per_device_eval_batch_size $BATCH_SIZE \
--save_steps 1750 \
--eval_steps 8 \
--seed $SEED \
--do_train \
--do_eval \
--evaluate_during_training \
--learning_rate $LEARNING_RATE \
--gradient_accumulation_steps $GRAD_ACCUM_SIZE \
--warmup_steps 10