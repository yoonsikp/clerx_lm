python3 ./model/test_pipeline.py \
--labels ./3_iob_data/labels.txt \
--model_name_or_path $BERT_MODEL \
--entity_data_dir $ENTITY_DATA \
--test_data_dir $TEST_DATA \
--summary_dir ./6_validation/
