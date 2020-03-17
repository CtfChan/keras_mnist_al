#!/bin/bash
# Basic range in for loop

# for af in RANDOM ENTROPY VAR_RATIO BALD MEAN_STD MARGIN_SAMPLING CLASSIFICATION_STABILITY
# do
#     echo $af
#     for value in {1..3}
#     do
#         echo $value
#         python3 active_training.py --trial_number $value --acquisition_function $af --reverse_metrics True
#     done
# done
# echo All done


for af in RANDOM ENTROPY VAR_RATIO BALD MEAN_STD MARGIN_SAMPLING CLASSIFICATION_STABILITY
do
    python3 active_training.py --trial_number 1 --acquisition_function $af --reverse_metrics True
done
echo All done