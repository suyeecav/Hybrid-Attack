# Tutorial

In this tutorial, we will make several notes on how to incorporate new attacks in our framework.

In general, to incorporate new attacks in our framework, The attack function should have the following parameters as the augments with data type numpy.ndarray:

* attack_seed: starting points to attack, it has the size of (nb_imgs, img_size, img_size, num_channel)

* initial_img: natural images to attack, it has the same size as attack_seed (nb_imgs, img_size, img_size, num_channel)

* target_class: target class to attack. If the attack is targeted attack, it indiates the target class; If the attack is untargeted attack, it indicates original class. Depending on you impelementation of attack, its size can be (num_imgs,) or (num_imgs, num_class).

Since blackbox attacks can take into different forms and can have different parameters, there are many ways to write a blackbox adversarial attack. If you are using tensorflow or numpy-like method to implement the blackbox attack. You can check [NES.py](https://github.com/suyeecav/Hybrid-Attack/blob/master/tutorial/NES.py) and [AUTOZOOM.py](https://github.com/suyeecav/Hybrid-Attack/blob/master/tutorial/AUTOZOOM.py) in the folder for detail. Both attacks take ```attack_seed```, ```initial_img``` and ```target_class``` as input parameters, as well as other attack-specfic parameters.

Also, we provide Pytorch codes to verify our framework (e.g., hypothesis 1). The attack function interface is similar to the tensorflow implentation. We implement [simple black-box attack](https://github.com/cg563/simple-blackbox-attack/blob/master/simba_single.py) and incorpate it to test hypthesis 1 for demonstration. To run the code, 

```
python test_h1_pytorch --black-attack simBA
```

If you are using Pytorch to write the attack function, please check [simBA.py](https://github.com/suyeecav/Hybrid-Attack/blob/master/tutorial/simBA.py) for details.
