#!/usr/bin/env bash

# NO CONTEXT

python3 ./scripts/iob_entity.py --input 2_annotated_articles/train > ./3_iob_data/no_context/train_entity/train.txt
python3 ./scripts/iob_entity.py --input 2_annotated_articles/validation > ./3_iob_data/no_context/train_entity/dev.txt
python3 ./scripts/iob_entity.py --input 2_annotated_articles/test > ./3_iob_data/no_context/test_entity/test.txt

python3 ./scripts/iob_entity_relation.py --input 2_annotated_articles/train > ./3_iob_data/no_context/train_entity_relation/train.txt
## use plain NER for validation set
cp ./3_iob_data/no_context/train_entity/dev.txt ./3_iob_data/no_context/train_entity_relation/dev.txt
python3 ./scripts/iob_entity_relation.py --input 2_annotated_articles/test > ./3_iob_data/no_context/test_relation/test.txt
## add plain NER to training set for JER
cat ./3_iob_data/no_context/train_entity/train.txt >> ./3_iob_data/no_context/train_entity_relation/train.txt
## shuffle plain NER and JER
python3 ./scripts/shuffler.py --seed 4 --input ./3_iob_data/no_context/train_entity_relation/train.txt > /tmp/nc_jer_train.txt
mv /tmp/nc_jer_train.txt ./3_iob_data/no_context/train_entity_relation/train.txt

# WITH CONTEXT

python3 ./scripts/iob_entity.py --input 2_annotated_articles/train --context > ./3_iob_data/context/train_entity/train.txt
python3 ./scripts/iob_entity.py --input 2_annotated_articles/validation --context > ./3_iob_data/context/train_entity/dev.txt
python3 ./scripts/iob_entity.py --input 2_annotated_articles/test --context > ./3_iob_data/context/test_entity/test.txt

python3 ./scripts/iob_entity_relation.py --input 2_annotated_articles/train --context > ./3_iob_data/context/train_entity_relation/train.txt
## use plain NER for validation set
cp ./3_iob_data/context/train_entity/dev.txt ./3_iob_data/context/train_entity_relation/dev.txt
python3 ./scripts/iob_entity_relation.py --input 2_annotated_articles/test --context > ./3_iob_data/context/test_relation/test.txt
## add plain NER to training set for JER
cat ./3_iob_data/context/train_entity/train.txt >> ./3_iob_data/context/train_entity_relation/train.txt
## shuffle plain NER and JER
python3 ./scripts/shuffler.py --seed 4 --input ./3_iob_data/context/train_entity_relation/train.txt > /tmp/c_jer_train.txt
mv /tmp/c_jer_train.txt ./3_iob_data/context/train_entity_relation/train.txt


# Add labels.txt to all folders
for f in ./3_iob_data/context/*; do pushd "$f"; ln -s ../../labels.txt labels.txt; popd; done
for f in ./3_iob_data/no_context/*; do pushd "$f"; ln -s ../../labels.txt labels.txt; popd; done


