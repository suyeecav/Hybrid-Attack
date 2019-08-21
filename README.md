# About
This repository is the code corresponding to paper "Hybrid Batch Attacks: Finding Black-box Adversarial Examples with Limited Queries", published at the 29th Usenix Security Symbosium. Arxiv version can be found at [here](https://arxiv.org/abs/1908.07000). The code is tested on Python3. TensorFlow version is 1.7.0 and Keras version is 2.2.4. The codes are grouped by the datasets for clarity.

# Pre-requisites
The following steps should be sufficient to install the required environments 
```
sudo apt-get install python3-pip  
sudo pip3 install --upgrade pip  
sudo pip3 install pillow scipy numpy tensorflow-gpu keras h5py numba  
sudo pip3 install cleverhans
```

# Reproduce Results
To reproduce our results, please follow the steps below:
1. Download the pretrained model weight files from [here](https://drive.google.com/drive/folders/1tQRRASL2qySOqUtDs12s62ssPhzgvRrZ?usp=sharing), if you do not want to train models on yourself. 
2. Place these model weight files in proper location and replace the file directories in the following files: **CIFAR10** directory (`hybrid_attack.py`, `cifar10_complex_models`, `cifar10_robust_models.py`, `cifar10_simple_models.py`). **MNIST** directory (`hybrid_attack.py`, `mnist_models.py`, `mnist_robust_models.py`). File directories need replacements in thses files are marked with **TODO**.
3. Go to specific dataset directory and run the shell files `autozoom_run.sh` and `nes_run.sh`. [AutoZOOM](https://github.com/IBM/Autozoom-Attack) and [NES](https://github.com/labsix/limited-blackbox-attacks) attacks are configured to run directly for the baseline, hybrid and hybrid with local fine-tuning attacks. 
