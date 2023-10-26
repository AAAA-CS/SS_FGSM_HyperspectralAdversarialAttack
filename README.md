# Prerequisites
Python 3.8.16<br>
Pytorch 1.12.1<br>
Numpy 1.22.4<br>
Scipy 1.7.3<br>

# Usage
1. The hyperspectral dataset is sourced from the link below , you can use your own dataset by matlab. I wish "SelectSample.m" could help you to select training examples. dataset link:<br>
```asp
                  https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes
```

3. To train the "AIAF_train.py" to defend against adversarial examples.<br>
  ```asp
                             $ python "AIAF_train.py" --dataset PaviaU
   ```
   
4. To test with a existing model:<br>
    ```asp
                             $ python AIAF_test.py --dataset PaviaU
   ```
						  
# Related works
>[ ARN](https://github.com/dwDavidxd/ARN " ARN")<br>
[ Torchattacks](https://github.com/Harry24k/adversarial-attacks-pytorch "> Torchattacks")
