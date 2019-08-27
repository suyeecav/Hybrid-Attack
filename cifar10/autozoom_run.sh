#!/bin/bash

# autozoom targeted attack on normal target model (baseline attack)
python3 hybrid_attack.py -n 100 --local_model_names no_local_model --no_save_model

# autozoom targeted attack on normal target model (hybrid attack) with normal local models
python3 hybrid_attack.py -n 100 --load_imgs --with_local --no_tune_local --local_model_names modelB modelD modelE --no_save_model

# autozoom targeted attack on normal target model (hybrid attack) with robust local models
python3 hybrid_attack.py -n 100 --load_imgs --with_local --no_tune_local --local_model_names adv_densenet adv_resnet --no_save_model

# autozoom targeted attack on normal target model (hybrid attack with local model tuning) with normal local models
python3 hybrid_attack.py -n 100 --load_imgs --with_local --local_model_names modelB modelD modelE --no_save_model



# autozoom untargeted attack on robust target model (baseline attack)
python3 hybrid_attack.py -n 100 --local_model_names no_local_model --no_save_model --model_type robust --attack_type untargeted

# autozoom untargeted attack on robust target model (hybrid attack) with normal local models
python3 hybrid_attack.py -n 100 --load_imgs --with_local --no_tune_local --local_model_names modelB modelD modelE --no_save_model --model_type robust --attack_type untargeted

# autozoom untargeted attack on robust target model (hybrid attack) with robust local models
python3 hybrid_attack.py -n 100 --load_imgs --with_local --no_tune_local --local_model_names adv_densenet adv_resnet --no_save_model --model_type robust --attack_type untargeted

# autozoom untargeted attack on robust target model (hybrid attack with local model tuning) with normal local models
python3 hybrid_attack.py -n 100 --load_imgs --with_local --local_model_names modelB modelD modelE --no_save_model --model_type robust --attack_type untargeted