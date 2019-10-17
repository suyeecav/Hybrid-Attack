# Tutorial

In this tutorial, we will makes several notes on how to incorporate new attacks in our framework.

Since blackbox attack can take into different forms: TBD
 
The attack function should have the following parameters as the augments with data type numpy.ndarray:

attack_seed: starting points to attack, it has the size of (nb_imgs, img_size, img_size, num_channel)

initial_img: natural images to attack, it has the same size as attack_seed (nb_imgs, img_size, img_size, num_channel)

target_class: target class to attack. If the attack is targeted attack, it indiates the target class; If the attack is untargeted attack, it indicates original class. Depending on you impelementation of attack, its size can be (num_imgs,) or (num_imgs, num_class).

In our implementation, AutoZOOM attack is implemented using Tensorflow and NES attack is implemnented using Numpy Operation. You can refer [attack_util.py (AutoZOOM)](https://github.com/suyeecav/Hybrid-Attack/blob/master/imagenet/autozoom/attack_utils.py) and [attack_util.py (NES)](https://github.com/suyeecav/Hybrid-Attack/blob/master/imagenet/nes/attack_utils.py) more detailed implementation. If you want incorpate new attack using Pytorch (e.g., test H1 in our paper using Pytorch Implemented attack), here is a [demo implmentation](https://drive.google.com/file/d/16PodfFGcUpMIOO20Uuyry6xf79njt7Ho/view?usp=sharing)