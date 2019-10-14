# About
This repository is the code corresponding to paper "Hybrid Batch Attacks: Finding Black-box Adversarial Examples with Limited Queries", published at the 29th Usenix Security Symbosium. Arxiv version can be found at [here](https://arxiv.org/abs/1908.07000). The code is tested on Python3.6. TensorFlow version is 1.7.0 and Keras version is 2.2.4. You will need a GPU to run the code efficiently. Batch attack related scripts are written in Matlab and are executed locally (scripts are tested on Matlab 2018b, but should be compatible with most of the Matlab versions). The codes are grouped by the datasets for clarity. **To Artifact Evaluation Committee: a note of "For AEC Artifact Evaluation:" is also added to the relevant Matlab scripts and please follow the specific instructions.** 

# Install Dependencies
You can directly install the depencies by running the following command:
```
pip install -r requirements.txt
```

# Reproduce Results
To reproduce our results, please follow the steps below:
1. Download the pretrained model weight files from [here](https://drive.google.com/drive/folders/1tQRRASL2qySOqUtDs12s62ssPhzgvRrZ?usp=sharing), if you do not want to train models on yourself. 
2. Place the model weight files of specific dataset into their corresponding dataset folder (e.g., place `MNIST_models` folder of the model weights into the `mnist` folder).
3. Go to specific dataset directory and run the shell files `autozoom_run.sh` and `nes_run.sh`. [AutoZOOM](https://github.com/IBM/Autozoom-Attack) and [NES](https://github.com/labsix/limited-blackbox-attacks) attacks are configured to run directly for the baseline, hybrid and hybrid with local fine-tuning attacks. We recommend to store the screen output into an output file such that key attack statistics (e.g., local model transfer rate) can be easily tracked. For example, for the command in each shell file, considering adding an additional command of `|tee output.txt`). Some of the results in the paper are averaged over 5 runs (specified in the paper). To obtain these 5 results, simply set the `args['seed']` to `1234`, `2345`, `3456`, `4567`, `5678` respectively (running each script 5 times can be very time-consuming). 
4. After you finish running the scripts, attack results will be stored in the form of `.txt` files. To reproduce the hybrid attack results and batch attack results in the paper, please run the Matlab script in the `batch_attacks` folder (more instructions available inside). MNIST and CIFAR10 results can be obtained by running the scripts in `batch_attacks` folder. ImageNet results have a separate folder of `batch_attacks` inside the folder `imagenet`.



# Incorporate New Attack into our Framework

To incorporate new black-box attack into our framework, The attack function should have the following parameters as the augments with data type numpy.ndarray:

attack_seed: starting points to attack, it has the size of (nb_imgs, img_size, img_size, num_channel)

initial_img: natural images to attack, it has the same size as attack_seed (nb_imgs, img_size, img_size, num_channel)

target_class: target class to attack. If the attack is targeted attack, it indiates the target class; If the attack is untargeted attack, it indicates original class. Depending on you impelementation of attack, its size can be (num_imgs,) or (num_imgs, num_class).

In our implementation, AutoZOOM attack is implemented using Tensorflow and NES attack is implemnented using Numpy Operation. You can refer [attack_util.py (AutoZOOM)](https://github.com/suyeecav/Hybrid-Attack/blob/master/imagenet/autozoom/attack_utils.py) and [attack_util.py (NES)](https://github.com/suyeecav/Hybrid-Attack/blob/master/imagenet/nes/attack_utils.py) more detailed implementation. If you want incorpate new attack using Pytorch (e.g., test H1 in our paper using Pytorch Implemented attack), here is a [demo implmentation](https://drive.google.com/file/d/16PodfFGcUpMIOO20Uuyry6xf79njt7Ho/view?usp=sharing)