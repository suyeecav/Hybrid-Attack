# Tutorial

In this tutorial, we outline the key steps to incorporate a new black-box attack into the hybrid attack framework.

To incorporate new black-box attacks, the attack function should have the following key parameters as the augments of type `numpy.ndarray`:

* ```attack_seed```: starting point for the black-box attack with a shape of (```nb_imgs```, ```img_size```, ```img_size```, ```num_channel```)

* ```initial_img```: original (i.e., natural) image used for constraining the perturbation size. ```initial_img``` has same shape as the ```attack_seed```

* ```target_class```: the target class to attack. If the black-box attack is targeted, it denotes the specific class to cause misclassification; If the attack is untargeted, it denotes the original class. The shape of ```target_class``` can be (num_imgs,) or (num_imgs, num_class) and freely choose the one that fits for the new black-box attack to be incorporated.

We provide versions of the hybrid attack implemented in TensorFlow and PyTorch. 

If you are using blackbox attacks implemented in numpy or tensorflow, please refer to [NES.py](https://github.com/suyeecav/Hybrid-Attack/blob/master/tutorial/NES.py) and [AUTOZOOM.py](https://github.com/suyeecav/Hybrid-Attack/blob/master/tutorial/AUTOZOOM.py) in the folder for details. 

If your attack is written in PyTorch, please refer to the demo of [simBA.py](https://github.com/suyeecav/Hybrid-Attack/blob/master/tutorial/simBA.py), which is adapted from the original [simple black-box pixel-attack](https://github.com/cg563/simple-blackbox-attack/blob/master/simba_single.py). 

All attacks take ```attack_seed```, ```initial_img``` and ```target_class``` as input parameters in addition to other attack-specfic parameters. As an illustration, in order to run the ```simple black-box pixel-attack```, execute the following command:

```
python test_h1_pytorch.py --black-attack simBA
```
