# Prerequisites
Python 3.8.16<br>
Pytorch 1.12.1<br>
Numpy 1.22.4<br>
Scipy 1.7.3<br>

# Usage
# Usage
1. To train the "target_model.py" with dataset PvaiaU ,which will generate checkpoint:'/900(1000)net_resnet.pkl'.  It's trained by a simple CNN classifier,you can try other targetmodel，such as VGG, Inc-v3.<br>
 ```asp
                        $ python target_model.py --dataset PaviaU --train 
   ```
  
2. Besides, the hyperspectral dataset is sourced from the link below , you can use your own dataset by matlab. I wish "SelectSample.m" could help you to select training examples. dataset link:<br>
```asp
                  https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes
```
3. Run the "SS_FGSM.py" to generate adversarial examples.<br>
  ```asp
                             $ python "SS_FGSM.py" --dataset PaviaU
   ```
					  
# Related works
>[ super_pix_adv](https://github.com/LightDXY/Super_Pix_Adv#super_pix_adv "> Super_Pix_Adv")<br>
[ Torchattacks](https://github.com/Harry24k/adversarial-attacks-pytorch "> Torchattacks")
