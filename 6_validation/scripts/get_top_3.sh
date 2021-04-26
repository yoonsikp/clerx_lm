#!/bin/bash

cat ./6_validation/jer_stats.csv| grep AGGREGATE | python3 ./6_validation/scripts/get_top_3.py > ./6_validation/top_3.txt
