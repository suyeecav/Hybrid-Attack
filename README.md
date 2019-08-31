# About
This repository is the code corresponding to paper "Hybrid Batch Attacks: Finding Black-box Adversarial Examples with Limited Queries", published at the 29th Usenix Security Symbosium. Arxiv version can be found at [here](https://arxiv.org/abs/1908.07000). The code is tested on Python3. TensorFlow version is 1.7.0 and Keras version is 2.2.4. You will need a GPU to run the code efficiently. Batch attack related scripts are written in Matlab and are executed locally (scripts are tested on Matlab 2018b, but should be compatible with most of the Matlab versions). The codes are grouped by the datasets for clarity. **To Artifact Evaluation Committee: a note of "For AEC Artifact Evaluation:" is also added to the relevant Matlab scripts and please follow the specific instructions.** 

# Install Dependencies
You can directly install the depencies by running the following command:
```
pip install -r requirements.txt
```

# Reproduce Results
To reproduce our results, please follow the steps below:
1. Download the pretrained model weight files from [here](https://drive.google.com/drive/folders/1tQRRASL2qySOqUtDs12s62ssPhzgvRrZ?usp=sharing), if you do not want to train models on yourself. 
<<<<<<< HEAD
2. Place the model weight files of specific dataset into their corresponding dataset folder (e.g., place `MNIST_models` folder of the model weights into the `mnist` folder).
=======
2. Place the model weight files of specific dataset into their corresponding dataset folder (e.g., place `MNIST_models` folder of the mnist model weights into the `mnist` folder).
>>>>>>> 2978ab4b6e691caffd6c99644da8d57598c322ee
3. Go to specific dataset directory and run the shell files `autozoom_run.sh` and `nes_run.sh`. [AutoZOOM](https://github.com/IBM/Autozoom-Attack) and [NES](https://github.com/labsix/limited-blackbox-attacks) attacks are configured to run directly for the baseline, hybrid and hybrid with local fine-tuning attacks. We recommend to store the screen output into an output file such that key attack statistics (e.g., local model transfer rate) can be easily tracked. For example, for the command in each shell file, considering adding an additional command of `|tee output.txt`). Some of the results in the paper are averaged over 5 runs (specified in the paper). To obtain these 5 results, simply set the `args['seed']` to `1234`, `2345`, `3456`, `4567`, `5678` respectively (running each script 5 times can be very time-consuming). 
4. After you finish running the scripts, attack results will be stored in the form of `.txt` files. To reproduce the hybrid attack results and batch attack results in the paper, please run the Matlab script in the `batch_attacks` folder (more instructions available inside). MNIST and CIFAR10 results can be obtained by running the scripts in `batch_attacks` folder. ImageNet results have a separate folder of `batch_attacks` inside the folder `imagenet`.
