#!/bin/bash

# nes untargeted baseline attack on robust target model [Tab 3 in the paper]
python3 hybrid_attack.py -n 100 --attack_method nes --attack_type untargeted --model_type robust --local_model_names no_local_model --no_save_model

# nes untargeted hybrid attack on robust target model with normal local models [Tab 3,4,6 in the paper]
python3 hybrid_attack.py -n 100 --load_imgs --with_local --no_tune_local --attack_method nes --attack_type untargeted --model_type robust --local_model_names modelB modelD modelE --no_save_model 

# nes untargeted hybrid attack on robust target model with robust local models [Tab 4 in the paper]
python3 hybrid_attack.py -n 100 --load_imgs --with_local --no_tune_local --attack_method nes --attack_type untargeted --model_type robust --local_model_names adv_densenet adv_resnet --no_save_model

# nes untargeted hybrid attack (with local model tuning) on robust target model with normal local models [Tab 6 in the paper]
python3 hybrid_attack.py -n 100 --load_imgs --with_local --attack_method nes --attack_type untargeted --model_type robust --local_model_names modelB modelD modelE --no_save_model

# nes untargeted baseline attack (with local model tuning) on robust target model with normal local models [Tab 5 in the paper]
python3 hybrid_attack.py -n 100 --load_imgs --with_local --force_tune_baseline --attack_method nes --attack_type untargeted --model_type robust --local_model_names modelB modelD modelE --no_save_text --no_save_model


###########################################################
# nes targeted baseline attack on normal target model [Tab 3 in the paper]
python3 hybrid_attack.py -n 100 --attack_method nes --attack_type targeted --model_type normal --local_model_names no_local_model --no_save_model

# nes targeted hybrid attack on normal target model with normal local models [Tab 3,4,6 in the paper]
python3 hybrid_attack.py -n 100 --load_imgs --with_local --no_tune_local --attack_method nes --attack_type targeted --model_type normal --local_model_names modelB modelD modelE --no_save_model

# nes targeted hybrid attack on normal target model with robust local models [Tab 4 in the paper]
python3 hybrid_attack.py -n 100 --load_imgs --with_local --no_tune_local --attack_method nes --attack_type targeted --model_type normal --local_model_names adv_densenet adv_resnet --no_save_model

# nes targeted hybrid attack (with local model tuning) on normal target model with normal local models [Tab 6 in the paper]
python3 hybrid_attack.py -n 100 --load_imgs --with_local --attack_method nes --attack_type targeted --model_type normal --local_model_names modelB modelD modelE --no_save_model

# nes targeted baseline attack (with local model tuning) on normal target model with normal local models [Tab 5 in the paper]
python3 hybrid_attack.py -n 100 --load_imgs --with_local --force_tune_baseline --attack_method nes --attack_type targeted --model_type normal --local_model_names modelB modelD modelE --no_save_text --no_save_model