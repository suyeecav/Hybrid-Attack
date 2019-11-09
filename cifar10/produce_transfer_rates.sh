#!/bin/bash

echo "Transfer Rate of Attacking CIFAR10 normal target model with robust-2 local models" 
python3 hybrid_attack.py --load_imgs -n 100 --with_local --test_trans_rate_only --attack_type targeted --model_type normal --local_model_names adv_densenet adv_resnet --seed 1234 --no_save_text --no_save_model
echo "Transfer Rate of Attacking CIFAR10 robust target model with robust-2 local models"
python3 hybrid_attack.py --load_imgs -n 100 --with_local --test_trans_rate_only --attack_type untargeted --model_type robust --local_model_names adv_densenet adv_resnet --seed 1234 --no_save_text --no_save_model

echo "Transfer Rate of Attacking CIFAR10 normal target with normal-3 local models" 
python3 hybrid_attack.py --load_imgs -n 100 --with_local --test_trans_rate_only --attack_type targeted --model_type normal --local_model_names modelB modelD modelE --seed 1234 --no_save_text --no_save_model
echo "Transfer Rate of Attacking CIFAR10 robust target with normal-3 local models" 
python3 hybrid_attack.py --load_imgs -n 100 --with_local --test_trans_rate_only --attack_type untargeted --model_type robust --local_model_names modelB modelD modelE --seed 1234 --no_save_text --no_save_model

echo "Transfer Rate of Attacking CIFAR10 normal Target model with all 5 local models" 
python3 hybrid_attack.py --load_imgs -n 100 --with_local --test_trans_rate_only --attack_type targeted --model_type normal --local_model_names modelB modelD modelE adv_densenet adv_resnet --seed 1234 --no_save_text --no_save_model
echo "Transfer Rate of Attacking CIFAR10 robust Target model with all 5 local models" 
python3 hybrid_attack.py --load_imgs -n 100 --with_local --test_trans_rate_only --attack_type untargeted --model_type robust --local_model_names modelB modelD modelE adv_densenet adv_resnet --seed 1234 --no_save_text --no_save_model

echo "Transfer Rate of Attacking CIFAR10 normal target model with NA-NB local models" 
python3 hybrid_attack.py --load_imgs -n 100 --with_local --test_trans_rate_only --attack_type targeted --model_type normal --local_model_names modelB modelD --seed 1234 --no_save_text --no_save_model
echo "Transfer Rate of Attacking CIFAR10 robust target model with NA-NB local models" 
python3 hybrid_attack.py --load_imgs -n 100 --with_local --test_trans_rate_only --attack_type untargeted --model_type robust --local_model_names modelB modelD --seed 1234 --no_save_text --no_save_model

echo "Transfer Rate of Attacking CIFAR10 normal target model with NA local model" 
python3 hybrid_attack.py --load_imgs -n 100 --with_local --test_trans_rate_only --attack_type targeted --model_type normal --local_model_names modelB --seed 1234 --no_save_text --no_save_model
echo "Transfer Rate of Attacking CIFAR10 robust target model with NA local model" 
python3 hybrid_attack.py --load_imgs -n 100 --with_local --test_trans_rate_only --attack_type untargeted --model_type robust --local_model_names modelB --seed 1234 --no_save_text --no_save_model

echo "Transfer Rate of Attacking CIFAR10 normal target model with NB local model" 
python3 hybrid_attack.py --load_imgs -n 100 --with_local --test_trans_rate_only --attack_type targeted --model_type normal --local_model_names modelD --seed 1234 --no_save_text --no_save_model
cho "Transfer Rate of Attacking CIFAR10 robust target model with NB local model"
python3 hybrid_attack.py --load_imgs -n 100 --with_local --test_trans_rate_only --attack_type untargeted --model_type robust --local_model_names modelD --seed 1234 --no_save_text --no_save_model

echo "Transfer Rate of Attacking CIFAR10 normal target model with NC local model" 
python3 hybrid_attack.py --load_imgs -n 100 --with_local --test_trans_rate_only --attack_type targeted --model_type normal --local_model_names modelE --seed 1234 --no_save_text --no_save_model
echo "Transfer Rate of Attacking CIFAR10 robust target model with NC local model" 
python3 hybrid_attack.py --load_imgs -n 100 --with_local --test_trans_rate_only --attack_type untargeted --model_type robust --local_model_names modelE --seed 1234 --no_save_text --no_save_model

echo "Transfer Rate of Attacking CIFAR10 normal target model with R_Resnet model" 
python3 hybrid_attack.py --load_imgs -n 100 --with_local --test_trans_rate_only --attack_type targeted --model_type normal --local_model_names adv_resnet --seed 1234 --no_save_text --no_save_model
echo "Transfer Rate of Attacking CIFAR10 robust target model with R_Resnet model" 
python3 hybrid_attack.py --load_imgs -n 100 --with_local --test_trans_rate_only --attack_type untargeted --model_type robust --local_model_names adv_resnet --seed 1234 --no_save_text --no_save_model


echo "Transfer Rate of Attacking CIFAR10 normal target model with R_Densenet" 
python3 hybrid_attack.py --load_imgs -n 100 --with_local --test_trans_rate_only --attack_type targeted --model_type normal --local_model_names adv_densenet --seed 1234 --no_save_text --no_save_model
echo "Transfer Rate of Attacking CIFAR10 robust target model with R_Densenet"
python3 hybrid_attack.py --load_imgs -n 100 --with_local --test_trans_rate_only --attack_type untargeted --model_type robust --local_model_names adv_densenet --seed 1234 --no_save_text --no_save_model
