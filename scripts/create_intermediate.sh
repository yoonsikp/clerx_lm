#!/usr/bin/env bash

# NO CONTEXT

python3 ./scripts/intermediate_entity.py --input 2_annotated_articles/validation --output 4_intermediate_data/no_context/validation
python3 ./scripts/intermediate_entity.py --input 2_annotated_articles/test --output 4_intermediate_data/no_context/test

python3 ./scripts/intermediate_entity_relation.py --input 2_annotated_articles/validation --output 4_intermediate_data/no_context/validation
python3 ./scripts/intermediate_entity_relation.py --input 2_annotated_articles/test --output 4_intermediate_data/no_context/test

# WITH CONTEXT

python3 ./scripts/intermediate_entity.py --input 2_annotated_articles/validation --output 4_intermediate_data/context/validation --context
python3 ./scripts/intermediate_entity.py --input 2_annotated_articles/test --output 4_intermediate_data/context/test --context

python3 ./scripts/intermediate_entity_relation.py --input 2_annotated_articles/validation --output 4_intermediate_data/context/validation --context
python3 ./scripts/intermediate_entity_relation.py --input 2_annotated_articles/test --output 4_intermediate_data/context/test --context
