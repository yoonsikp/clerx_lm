python3 ./model/test.py \
--labels ./3_iob_data/labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--test_relations $TEST_RELATIONS \
--test_data $TEST_DATA
