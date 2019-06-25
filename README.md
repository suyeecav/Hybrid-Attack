# About
This repository is the code corresponding to paper "Improved Estimation of Cost of Black-box Attacks", published at the 29th Usenix Security Symbosium. Arxiv version can be found at [here](http://www.cs.virginia.edu/~evans/). The code is tested on Python3. TensorFlow version is 1.7.0 and Keras version is 2.2.4. The codes are grouped by the datasets for clarity.

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
1. Download the pretrained model weight files from [here](https://www.dropbox.com/sh/gdipubr7rp0d8qv/AADPgrs4ZGfOl4ob_dXmYsFla?dl=0), if you do not want to train models on yourself. 
2. Place these model weight files in proper location and replace the file directories in the following files: **CIFAR10** directory (`hybrid_attack.py`, `cifar10_complex_models`, `cifar10_robust_models.py`, `cifar10_simple_models.py`). **MNIST** directory (`hybrid_attack.py`, `mnist_models.py`, `mnist_robust_models.py`). File directories need replacements in thses files are marked with **TODO**.
3. Go to specific dataset directory and run the shell file. [AutoZOOM](https://github.com/IBM/Autozoom-Attack) and [NES](https://github.com/labsix/limited-blackbox-attacks) attacks are configured to run directly for the baseline, hybrid and hybrid with local fine-tuning attacks. 
