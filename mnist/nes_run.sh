#!/bin/bash

# nes baseline targeted attack on normal mnist model
python3 hybrid_attack.py -n 100 --no_save_model --attack_method nes --attack_type targeted  

# nes hybrid targeted attack on normal mnist model
python3 hybrid_attack.py --with_local --no_tune_local --load_imgs -n 100 --no_save_model --attack_method nes

# nes hybrid targeted attack on normal mnist model with local model tuning
python3 hybrid_attack.py --with_local --load_imgs -n 100 --no_save_model --attack_method nes



# nes baseline untargeted attack on robust mnist model
python3 hybrid_attack.py -n 100 --no_save_model --attack_method nes --attack_type untargeted --model_type robust  

# nes hybrid untargeted attack on robust mnist model
python3 hybrid_attack.py --with_local --no_tune_local --load_imgs -n 100 --no_save_model --attack_method nes --attack_type untargeted --model_type robust 

# nes hybrid untargeted attack on robust mnist model with local model tuning
python3 hybrid_attack.py --with_local --load_imgs -n 100 --no_save_model --attack_method nes --attack_type untargeted --model_type robust 
