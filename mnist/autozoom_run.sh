#!/bin/bash
# autozoom baseline targeted attack on normal mnist model [Tab 3 in the paper]
python3 hybrid_attack.py -n 100 --no_save_model --attack_type targeted --model_type normal

# autozoom hybrid targeted attack on normal mnist model without tuning local models [Tab 3,6 in the paper]
python3 hybrid_attack.py --load_imgs -n 100 --with_local --no_tune_local --no_save_model --attack_type targeted --model_type normal 

# autozoom hybrid targeted attack on normal mnist model with tuning local models [Tab 6 in the paper]
python3 hybrid_attack.py --load_imgs -n 100 --with_local  --no_save_model --attack_type targeted --model_type normal 

# autozoom baseline targeted attack on normal mnist model with tuning local models [Tab 5 in the paper]
python3 hybrid_attack.py --load_imgs -n 100 --with_local --force_tune_baseline --attack_type targeted --model_type normal --no_save_model --no_save_text


################################################################
# autozoom baseline untargeted attack on robust mnist model [Tab 3 in the paper]
python3 hybrid_attack.py -n 100 --no_save_model --attack_type untargeted --model_type robust

# autozoom hybrid untargeted attack on robust mnist model without tuning local models [Tab 3,6 in the paper]
python3 hybrid_attack.py --load_imgs -n 100 --with_local --no_tune_local --no_save_model --attack_type untargeted --model_type robust 

# autozoom hybrid untargeted attack on robust mnist model with tuning local models [Tab 6 in the paper]
python3 hybrid_attack.py --load_imgs -n 100 --with_local  --no_save_model --attack_type untargeted --model_type robust 

# autozoom baseline untargeted attack on robust mnist model with tuning local models [Tab 5 in the paper]
python3 hybrid_attack.py --load_imgs -n 100 --with_local --force_tune_baseline --attack_type untargeted --model_type robust --no_save_model --no_save_text