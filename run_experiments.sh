#!/bin/bash

# MIMIC-CXR
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset mimic-cxr --protected_attr sex --male_percent 0.0 --num_seeds 10 --log_dir logs/FAE_mimic-cxr_sex/male_percent_00
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset mimic-cxr --protected_attr sex --male_percent 0.1 --num_seeds 10 --log_dir logs/FAE_mimic-cxr_sex/male_percent_01
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset mimic-cxr --protected_attr sex --male_percent 0.2 --num_seeds 10 --log_dir logs/FAE_mimic-cxr_sex/male_percent_02
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset mimic-cxr --protected_attr sex --male_percent 0.3 --num_seeds 10 --log_dir logs/FAE_mimic-cxr_sex/male_percent_03
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset mimic-cxr --protected_attr sex --male_percent 0.4 --num_seeds 10 --log_dir logs/FAE_mimic-cxr_sex/male_percent_04
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset mimic-cxr --protected_attr sex --male_percent 0.5 --num_seeds 10 --log_dir logs/FAE_mimic-cxr_sex/male_percent_05
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset mimic-cxr --protected_attr sex --male_percent 0.6 --num_seeds 10 --log_dir logs/FAE_mimic-cxr_sex/male_percent_06
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset mimic-cxr --protected_attr sex --male_percent 0.7 --num_seeds 10 --log_dir logs/FAE_mimic-cxr_sex/male_percent_07
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset mimic-cxr --protected_attr sex --male_percent 0.8 --num_seeds 10 --log_dir logs/FAE_mimic-cxr_sex/male_percent_08
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset mimic-cxr --protected_attr sex --male_percent 0.9 --num_seeds 10 --log_dir logs/FAE_mimic-cxr_sex/male_percent_09
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset mimic-cxr --protected_attr sex --male_percent 1.0 --num_seeds 10 --log_dir logs/FAE_mimic-cxr_sex/male_percent_10

python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset mimic-cxr --protected_attr age --old_percent 0.0 --num_seeds 10 --log_dir logs/FAE_mimic-cxr_age/old_percent_00
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset mimic-cxr --protected_attr age --old_percent 0.1 --num_seeds 10 --log_dir logs/FAE_mimic-cxr_age/old_percent_01
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset mimic-cxr --protected_attr age --old_percent 0.2 --num_seeds 10 --log_dir logs/FAE_mimic-cxr_age/old_percent_02
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset mimic-cxr --protected_attr age --old_percent 0.3 --num_seeds 10 --log_dir logs/FAE_mimic-cxr_age/old_percent_03
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset mimic-cxr --protected_attr age --old_percent 0.4 --num_seeds 10 --log_dir logs/FAE_mimic-cxr_age/old_percent_04
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset mimic-cxr --protected_attr age --old_percent 0.5 --num_seeds 10 --log_dir logs/FAE_mimic-cxr_age/old_percent_05
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset mimic-cxr --protected_attr age --old_percent 0.6 --num_seeds 10 --log_dir logs/FAE_mimic-cxr_age/old_percent_06
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset mimic-cxr --protected_attr age --old_percent 0.7 --num_seeds 10 --log_dir logs/FAE_mimic-cxr_age/old_percent_07
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset mimic-cxr --protected_attr age --old_percent 0.8 --num_seeds 10 --log_dir logs/FAE_mimic-cxr_age/old_percent_08
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset mimic-cxr --protected_attr age --old_percent 0.9 --num_seeds 10 --log_dir logs/FAE_mimic-cxr_age/old_percent_09
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset mimic-cxr --protected_attr age --old_percent 1.0 --num_seeds 10 --log_dir logs/FAE_mimic-cxr_age/old_percent_10

python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset mimic-cxr --protected_attr race --white_percent 0.0 --num_seeds 10 --log_dir logs/FAE_mimic-cxr_race/white_percent_00
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset mimic-cxr --protected_attr race --white_percent 0.1 --num_seeds 10 --log_dir logs/FAE_mimic-cxr_race/white_percent_01
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset mimic-cxr --protected_attr race --white_percent 0.2 --num_seeds 10 --log_dir logs/FAE_mimic-cxr_race/white_percent_02
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset mimic-cxr --protected_attr race --white_percent 0.3 --num_seeds 10 --log_dir logs/FAE_mimic-cxr_race/white_percent_03
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset mimic-cxr --protected_attr race --white_percent 0.4 --num_seeds 10 --log_dir logs/FAE_mimic-cxr_race/white_percent_04
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset mimic-cxr --protected_attr race --white_percent 0.5 --num_seeds 10 --log_dir logs/FAE_mimic-cxr_race/white_percent_05
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset mimic-cxr --protected_attr race --white_percent 0.6 --num_seeds 10 --log_dir logs/FAE_mimic-cxr_race/white_percent_06
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset mimic-cxr --protected_attr race --white_percent 0.7 --num_seeds 10 --log_dir logs/FAE_mimic-cxr_race/white_percent_07
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset mimic-cxr --protected_attr race --white_percent 0.8 --num_seeds 10 --log_dir logs/FAE_mimic-cxr_race/white_percent_08
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset mimic-cxr --protected_attr race --white_percent 0.9 --num_seeds 10 --log_dir logs/FAE_mimic-cxr_race/white_percent_09
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset mimic-cxr --protected_attr race --white_percent 1.0 --num_seeds 10 --log_dir logs/FAE_mimic-cxr_race/white_percent_10


# CXR14
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset cxr14 --protected_attr sex --male_percent 0.0 --num_seeds 10 --log_dir logs/FAE_cxr14_sex/male_percent_00
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset cxr14 --protected_attr sex --male_percent 0.1 --num_seeds 10 --log_dir logs/FAE_cxr14_sex/male_percent_01
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset cxr14 --protected_attr sex --male_percent 0.2 --num_seeds 10 --log_dir logs/FAE_cxr14_sex/male_percent_02
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset cxr14 --protected_attr sex --male_percent 0.3 --num_seeds 10 --log_dir logs/FAE_cxr14_sex/male_percent_03
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset cxr14 --protected_attr sex --male_percent 0.4 --num_seeds 10 --log_dir logs/FAE_cxr14_sex/male_percent_04
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset cxr14 --protected_attr sex --male_percent 0.5 --num_seeds 10 --log_dir logs/FAE_cxr14_sex/male_percent_05
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset cxr14 --protected_attr sex --male_percent 0.6 --num_seeds 10 --log_dir logs/FAE_cxr14_sex/male_percent_06
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset cxr14 --protected_attr sex --male_percent 0.7 --num_seeds 10 --log_dir logs/FAE_cxr14_sex/male_percent_07
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset cxr14 --protected_attr sex --male_percent 0.8 --num_seeds 10 --log_dir logs/FAE_cxr14_sex/male_percent_08
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset cxr14 --protected_attr sex --male_percent 0.9 --num_seeds 10 --log_dir logs/FAE_cxr14_sex/male_percent_09
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset cxr14 --protected_attr sex --male_percent 1.0 --num_seeds 10 --log_dir logs/FAE_cxr14_sex/male_percent_10

python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset cxr14 --protected_attr age --old_percent 0.0 --num_seeds 10 --log_dir logs/FAE_cxr14_age/old_percent_00
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset cxr14 --protected_attr age --old_percent 0.1 --num_seeds 10 --log_dir logs/FAE_cxr14_age/old_percent_01
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset cxr14 --protected_attr age --old_percent 0.2 --num_seeds 10 --log_dir logs/FAE_cxr14_age/old_percent_02
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset cxr14 --protected_attr age --old_percent 0.3 --num_seeds 10 --log_dir logs/FAE_cxr14_age/old_percent_03
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset cxr14 --protected_attr age --old_percent 0.4 --num_seeds 10 --log_dir logs/FAE_cxr14_age/old_percent_04
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset cxr14 --protected_attr age --old_percent 0.5 --num_seeds 10 --log_dir logs/FAE_cxr14_age/old_percent_05
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset cxr14 --protected_attr age --old_percent 0.6 --num_seeds 10 --log_dir logs/FAE_cxr14_age/old_percent_06
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset cxr14 --protected_attr age --old_percent 0.7 --num_seeds 10 --log_dir logs/FAE_cxr14_age/old_percent_07
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset cxr14 --protected_attr age --old_percent 0.8 --num_seeds 10 --log_dir logs/FAE_cxr14_age/old_percent_08
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset cxr14 --protected_attr age --old_percent 0.9 --num_seeds 10 --log_dir logs/FAE_cxr14_age/old_percent_09
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset cxr14 --protected_attr age --old_percent 1.0 --num_seeds 10 --log_dir logs/FAE_cxr14_age/old_percent_10

# CheXpert
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset chexpert --protected_attr sex --male_percent 0.0 --num_seeds 10 --log_dir logs/FAE_chexpert_sex/male_percent_00
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset chexpert --protected_attr sex --male_percent 0.1 --num_seeds 10 --log_dir logs/FAE_chexpert_sex/male_percent_01
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset chexpert --protected_attr sex --male_percent 0.2 --num_seeds 10 --log_dir logs/FAE_chexpert_sex/male_percent_02
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset chexpert --protected_attr sex --male_percent 0.3 --num_seeds 10 --log_dir logs/FAE_chexpert_sex/male_percent_03
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset chexpert --protected_attr sex --male_percent 0.4 --num_seeds 10 --log_dir logs/FAE_chexpert_sex/male_percent_04
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset chexpert --protected_attr sex --male_percent 0.5 --num_seeds 10 --log_dir logs/FAE_chexpert_sex/male_percent_05
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset chexpert --protected_attr sex --male_percent 0.6 --num_seeds 10 --log_dir logs/FAE_chexpert_sex/male_percent_06
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset chexpert --protected_attr sex --male_percent 0.7 --num_seeds 10 --log_dir logs/FAE_chexpert_sex/male_percent_07
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset chexpert --protected_attr sex --male_percent 0.8 --num_seeds 10 --log_dir logs/FAE_chexpert_sex/male_percent_08
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset chexpert --protected_attr sex --male_percent 0.9 --num_seeds 10 --log_dir logs/FAE_chexpert_sex/male_percent_09
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset chexpert --protected_attr sex --male_percent 1.0 --num_seeds 10 --log_dir logs/FAE_chexpert_sex/male_percent_10

python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset chexpert --protected_attr age --old_percent 0.0 --num_seeds 10 --log_dir logs/FAE_chexpert_age/old_percent_00
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset chexpert --protected_attr age --old_percent 0.1 --num_seeds 10 --log_dir logs/FAE_chexpert_age/old_percent_01
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset chexpert --protected_attr age --old_percent 0.2 --num_seeds 10 --log_dir logs/FAE_chexpert_age/old_percent_02
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset chexpert --protected_attr age --old_percent 0.3 --num_seeds 10 --log_dir logs/FAE_chexpert_age/old_percent_03
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset chexpert --protected_attr age --old_percent 0.4 --num_seeds 10 --log_dir logs/FAE_chexpert_age/old_percent_04
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset chexpert --protected_attr age --old_percent 0.5 --num_seeds 10 --log_dir logs/FAE_chexpert_age/old_percent_05
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset chexpert --protected_attr age --old_percent 0.6 --num_seeds 10 --log_dir logs/FAE_chexpert_age/old_percent_06
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset chexpert --protected_attr age --old_percent 0.7 --num_seeds 10 --log_dir logs/FAE_chexpert_age/old_percent_07
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset chexpert --protected_attr age --old_percent 0.8 --num_seeds 10 --log_dir logs/FAE_chexpert_age/old_percent_08
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset chexpert --protected_attr age --old_percent 0.9 --num_seeds 10 --log_dir logs/FAE_chexpert_age/old_percent_09
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset chexpert --protected_attr age --old_percent 1.0 --num_seeds 10 --log_dir logs/FAE_chexpert_age/old_percent_10


# Intersectional
python src/train.py --disable_wandb --max_steps 10000 --val_frequency 11000 --model_type FAE --dataset mimic-cxr --protected_attr intersectional_age_sex_race --num_seeds 10 --log_dir logs/FAE_mimic-cxr_intersectional_age_sex_race/only
