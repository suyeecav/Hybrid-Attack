#!/bin/bash

# autozoom targeted attack starting from local adversarial example [Tab 3 in the paper]
python main.py --attack_seed_type adv

# autozoom targeted attack starting from original examples [Tab 3 in the paper]
python main.py --attack_seed_type orig


