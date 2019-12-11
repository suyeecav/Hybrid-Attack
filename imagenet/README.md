This folder is for code of imagenet. To execute the NES or AutoZOOM attack, please go to specific folder and run the corresponding command of `python3 main.py`. Batch attack results of ImageNet can be obtained by running the Matlab scripts in the `batch_attacks` folder (more instructions inside). 

ImageNet validation dataset can be downloaded from the [official website](http://image-net.org/download). Next, create a folder named `imagenet_dataset` inside the `imagenet` folder and put the downloaded dataset here. If you do it correctly, you are able to see a `val` folder and `val.txt` in the `imagenet_dataset` folder.


In the AutoZOOM experiment, please download the autoencoder weights from [here](https://drive.google.com/drive/folders/12sRVYMaDPOhhph7cWi40tFWpnDoD-ocy). Next, create a folder named `codec/` in the `autozoom` folder and put the autoencoder weight files here.  