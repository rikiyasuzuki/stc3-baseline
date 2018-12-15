#!/usr/bin/env bash
PYTHON="python"

for task in "nugget" "quality"
do
    for language in "chinese" "english"
    do
        name="attention_based_representation"
        $PYTHON train.py --task $task --language $language --num-epoch 5 --tag $name || exit 1
    done
done
