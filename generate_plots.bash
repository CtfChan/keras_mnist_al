#!/bin/bash
# Basic range in for loop

for af in RANDOM ENTROPY VAR_RATIO BALD
do
    echo $af
    for value in {1..5}
    do
        echo $value
        python3 active_training.py --trial_number $value --acquisition_function $af
    done
done
echo All done