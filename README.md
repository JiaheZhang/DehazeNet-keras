# DehazeNet implemented by Keras
DehazeNet: An End-to-End System for Single Image Haze Removal [1]

DehazeNet keras+tensorflow version

This version includes the generation of training set. The only thing you need to do is running the code. Note that this version only can be used for training DehazeNet. If you want to remove the haze of images, you should implement the testing codes and post-processing codes by youself. I believe it is easy.

## Requirements

```py
python 3
tensorflow
keras
opencv
```

## Download the codes
git clone https://github.com/JiaheZhang/DehazeNet-keras

## Collect images

You should collect enough haze-free images for training. These images should be put into "./haze_free".


## Train

Run the codes by

```py
python DehazeNet.py
```

Good Luck



[1] B. Cai, X. Xu, K. Jia, C. Qing, and D. Tao. Dehazenet: An end-to-end system for single image haze removal. IEEE Transactions on Image Processing, 25(11), 2016.
