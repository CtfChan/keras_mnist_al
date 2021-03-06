#!/bin/bash
# Basic range in for loop

# no reverse
for value in {1..3}
do
    for af in RANDOM ENTROPY VAR_RATIO BALD MEAN_STD MARGIN_SAMPLING CLASSIFICATION_STABILITY
    do
        python3 active_training.py --trial_number $value  --acquisition_function $af --reverse_metrics False
    done
done
echo All done

# # with reversed metrics, so pick nonoptimal images
# for value in {1..3}
# do
#     for af in RANDOM ENTROPY VAR_RATIO BALD MEAN_STD MARGIN_SAMPLING CLASSIFICATION_STABILITY
#     do
#         python3 active_training.py --trial_number $value  --acquisition_function $af --reverse_metrics True
#     done
# done
# echo All done