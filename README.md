# DehazeNet-keras
DehazeNet keras+tensorflow version


The dataset should be generated from fog-free image patch by yourself. The patch size is 32 by 32, 3 channels.

The shape of x_train and y_train is (num, 32, 32, 3) and (num, 1), respectively, where num is number of training samples. 
Maybe, you should expand dim of y_train to (num, 1, 1, 1).

Good Luck
