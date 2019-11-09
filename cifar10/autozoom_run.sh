#!/bin/bash

# autozoom targeted attack on normal target model (baseline attack) [Tab 3 in the paper]
python3 hybrid_attack.py -n 100 --local_model_names no_local_model --no_save_model

# autozoom targeted attack on normal target model (hybrid attack) with normal local models [Tab 3,4,6 in the paper]
python3 hybrid_attack.py -n 100 --load_imgs --with_local --no_tune_local --local_model_names modelB modelD modelE --no_save_model

# autozoom targeted attack on normal target model (hybrid attack) with robust local models [Tab 4 in the paper]
python3 hybrid_attack.py -n 100 --load_imgs --with_local --no_tune_local --local_model_names adv_densenet adv_resnet --no_save_model

# autozoom targeted attack on normal target model (hybrid attack with local model tuning) with normal local models [Tab 6 in the paper]
python3 hybrid_attack.py -n 100 --load_imgs --with_local --local_model_names modelB modelD modelE --no_save_model

# autozoom targeted attack on normal target model (baseline attack with local model tuning) with normal local models [Tab 5 in the paper]
python3 hybrid_attack.py -n 100 --load_imgs --with_local --force_tune_baseline --attack_method autozoom --attack_type targeted --model_type normal --local_model_names modelB modelD modelE --no_save_text --no_save_model

############################################################################
# autozoom untargeted attack on robust target model (baseline attack) [Tab 3 in the paper]
python3 hybrid_attack.py -n 100 --local_model_names no_local_model --no_save_model --model_type robust --attack_type untargeted

# autozoom untargeted attack on robust target model (hybrid attack) with normal local models [Tab 3,4,6 in the paper]
python3 hybrid_attack.py -n 100 --load_imgs --with_local --no_tune_local --local_model_names modelB modelD modelE --no_save_model --model_type robust --attack_type untargeted

# autozoom untargeted attack on robust target model (hybrid attack) with robust local models [Tab 4 in the paper]
python3 hybrid_attack.py -n 100 --load_imgs --with_local --no_tune_local --local_model_names adv_densenet adv_resnet --no_save_model --model_type robust --attack_type untargeted

# autozoom untargeted attack on robust target model (hybrid attack with local model tuning) with normal local models [Tab 6 in the paper]
python3 hybrid_attack.py -n 100 --load_imgs --with_local --local_model_names modelB modelD modelE --no_save_model --model_type robust --attack_type untargeted

# autozoom targeted attack on robust target model (baseline attack with local model tuning) with normal local models [Tab 5 in the paper]
python3 hybrid_attack.py -n 100 --load_imgs --with_local --force_tune_baseline --attack_method autozoom --attack_type untargeted --model_type robust --local_model_names modelB modelD modelE --no_save_text --no_save_model