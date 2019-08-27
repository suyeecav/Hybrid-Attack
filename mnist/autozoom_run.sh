#!/bin/bash
# autozoom baseline targeted attack on normal mnist model
python3 hybrid_attack.py -n 100 --no_save_model --attack_type targeted --model_type normal

# autozoom hybrid targeted attack on normal mnist model
python3 hybrid_attack.py --load_imgs -n 100 --with_local --no_tune_local --no_save_model --attack_type targeted --model_type normal 

# autozoom hybrid targeted attack on normal mnist model with tuning
python3 hybrid_attack.py --load_imgs -n 100 --with_local  --no_save_model --attack_type targeted --model_type normal 



# autozoom baseline untargeted attack on robust mnist model
python3 hybrid_attack.py -n 100 --no_save_model --attack_type untargeted --model_type robust

# autozoom hybrid untargeted attack on robust mnist model
python3 hybrid_attack.py --load_imgs -n 100 --with_local --no_tune_local --no_save_model --attack_type untargeted --model_type robust 

# autozoom hybrid untargeted attack on robust mnist model with tuning
python3 hybrid_attack.py --load_imgs -n 100 --with_local  --no_save_model --attack_type untargeted --model_type robust 