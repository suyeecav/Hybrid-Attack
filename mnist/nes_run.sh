#!/bin/bash

# nes baseline targeted attack on normal mnist model [Tab 3 in the paper]
python3 hybrid_attack.py -n 100 --no_save_model --attack_method nes --attack_type targeted  

# nes hybrid targeted attack on normal mnist model without tuning local models [Tab 3,6 in the paper]
python3 hybrid_attack.py --with_local --no_tune_local --load_imgs -n 100 --no_save_model --attack_method nes

# nes hybrid targeted attack on normal mnist model with local model tuning [Tab 6 in the paper]
python3 hybrid_attack.py --with_local --load_imgs -n 100 --no_save_model --attack_method nes

# nes baseline targeted attack on normal mnist model with local model tuning [Tab 5 in the paper]
python3 hybrid_attack.py -n 100 --load_imgs --with_local --force_tune_baseline --attack_type targeted --model_type normal --no_save_model --no_save_text --attack_method nes

###########################################################
# nes baseline untargeted attack on robust mnist model [Tab 3 in the paper]
python3 hybrid_attack.py -n 100 --no_save_model --attack_method nes --attack_type untargeted --model_type robust  

# nes hybrid untargeted attack on robust mnist model without tuning local models [Tab 3,6 in the paper]
python3 hybrid_attack.py --with_local --no_tune_local --load_imgs -n 100 --no_save_model --attack_method nes --attack_type untargeted --model_type robust 

# nes hybrid untargeted attack on robust mnist model with local model tuning [Tab 6 in the paper]
python3 hybrid_attack.py --with_local --load_imgs -n 100 --no_save_model --attack_method nes --attack_type untargeted --model_type robust 

# nes baseline untargeted attack on robust mnist model with local model tuning [Tab 5 in the paper]
python3 hybrid_attack.py --load_imgs -n 100 --with_local --force_tune_baseline --attack_type untargeted --model_type robust --no_save_model --no_save_text --attack_method nes