# About
This repository is the code corresponding to paper "Hybrid Batch Attacks: Finding Black-box Adversarial Examples with Limited Queries", published at the 29th Usenix Security Symbosium. Arxiv version can be found at [here](https://arxiv.org/abs/1908.07000). The code is tested on Python3.6. TensorFlow version is 1.7.0 and Keras version is 2.2.4. You will need a GPU to run the code efficiently. Batch attack related scripts are written in Matlab and are executed locally (scripts are tested on Matlab 2018b, but should be compatible with most of the Matlab versions). The codes are grouped by the datasets for clarity. 

# Install Dependencies
You can directly install the depencies by running the following command:
```
pip install -r requirements.txt
```

# Reproduce Results
To reproduce our results, please follow the steps below:
1. Download the pretrained model weight files from [here](https://drive.google.com/drive/folders/1tQRRASL2qySOqUtDs12s62ssPhzgvRrZ?usp=sharing), if you do not want to train models on yourself. 
2. Place the model weight files of specific dataset into their corresponding dataset folder (e.g., place `MNIST_models` folder of the model weights into the `mnist` folder).
3. Go to specific dataset directory and refer to shell scripts `autozoom_run.sh` and `nes_run.sh` for running the corresponding attacks (you may want to selectively run some of the commands as running all of them is extremely time conusming). [AutoZOOM](https://github.com/IBM/Autozoom-Attack) and [NES](https://github.com/labsix/limited-blackbox-attacks) attacks are configured to run directly for the baseline attack, hybrid attack without tuning, hybrid attack with local fine-tuning and baseline attack with local fine-tuning. For CIFAR10 dataset, you can check the transfer rates of different local models to different target models by running the shell script `produce_transfer_rates.sh`. We recommend to store the screen output into an output file such that key attack statistics (e.g., local model transfer rate) can be easily tracked. For example, for the command in each shell file, consider adding an additional command of `|tee output.txt`). All results in the paper are averaged over 5 runs and to exactly reproduce them, simply set the `args['seed']` (default is `1234`) to `1234`, `2345`, `3456`, `4567`, `5678` respectively (WARNING: running each script 5 times with different seeds can be very time-consuming). 
4. After you finish running the attack scripts, attack results will be stored in the form of `.txt` files. To reproduce the hybrid attack results and batch attack results in the paper, please run the Matlab script in the `batch_attacks` folder (more instructions available inside). MNIST and CIFAR10 results can be obtained by running the scripts in `batch_attacks` folder. ImageNet results have a separate folder of `batch_attacks` inside the folder `imagenet`. Note that, Matlab scripts can run locally if the Python scripts run on a remote server, as long as the attack results in `.txt` files are downloaded to the local machine. 

# Incorporate New Attack into our Framework

To incorporate new black-box attack into our framework, please see our tutorial folder for more instruction.