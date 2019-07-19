import os
import numpy as np
import cv2
import random
from keras.models import Model
from keras.layers import Conv2D, Input, Concatenate
from keras.layers import MaxPool2D, Activation
from keras import backend as K
from keras.engine.topology import Layer
from keras import optimizers



# DehazeNet keras+tensorflow version 
# The implement code of the paper which is "DehazeNet: An End-to-End System for Single Image Haze Removal"


# choose a GTX 
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Maxout layer
class MaxoutConv2D(Layer):

    def __init__(self, **kwargs):

        super(MaxoutConv2D, self).__init__(**kwargs)

    def call(self, x):

        output = K.max(x, axis=-1, keepdims=True)
        return output

    def compute_output_shape(self, input_shape):
        input_height = input_shape[1]
        input_width = input_shape[2]

        output_height = input_height
        output_width = input_width

        return (input_shape[0], output_height, output_width, 1)


# BRelu
def BRelu(x):
    return K.relu(x, max_value=1)

## metric---r2
def r2(y_true, y_pred):
    return 1 - K.sum(K.square(y_pred - y_true)) / K.sum(K.square(y_true - K.mean(y_true)))


def create_dehaze_net():

    input_1 = Input(shape=(16, 16, 3))

    # conv2d  
    conv_1 = Conv2D(filters=4, kernel_size=(5, 5), strides=(1, 1))(input_1)

    conv_2 = Conv2D(filters=4, kernel_size=(5, 5), strides=(1, 1))(input_1)

    conv_3 = Conv2D(filters=4, kernel_size=(5, 5), strides=(1, 1))(input_1)

    conv_4 = Conv2D(filters=4, kernel_size=(5, 5), strides=(1, 1))(input_1)

    # Maxout layer
    max_out_1 = MaxoutConv2D()(conv_1)

    max_out_2 = MaxoutConv2D()(conv_2)

    max_out_3 = MaxoutConv2D()(conv_3)

    max_out_4 = MaxoutConv2D()(conv_4)

    max_out = Concatenate()([max_out_1, max_out_2, max_out_3, max_out_4])


    # multi-scale conv
    multi_layer_1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(max_out)

    multi_layer_2 = Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same')(max_out)

    multi_layer_3 = Conv2D(filters=16, kernel_size=(7, 7), strides=(1, 1), padding='same')(max_out)

    multi_layer = Concatenate()([multi_layer_1, multi_layer_2, multi_layer_3])

    # max pool
    max_pool_1 = MaxPool2D(pool_size=(7, 7), strides=(1,1))(multi_layer)

    conv_5 = Conv2D(filters=1, kernel_size=(6, 6), strides=(1, 1))(max_pool_1)

    output_1 = Activation(BRelu)(conv_5)


    model = Model(inputs=input_1, outputs=output_1)

    model.summary()

    return model


def create_dataset(img_dir, num_t=10, patch_size = 16):
    # img_dir: dir of haze-free images
    # num_t: number of t(x)
    # patch_size: size of image patch

    img_path = os.listdir(img_dir)

    x_train = []
    y_train = []

    for image_name in img_path:
        fullname = os.path.join(img_dir, image_name)
        img = cv2.imread(fullname)

        w,h,_ = img.shape

        num_w = int(w / patch_size)
        num_h = int(h / patch_size)
        for i in range(num_w):
            for j in range(num_h):

                free_patch = img[0+i*patch_size:patch_size+i*patch_size, 0+j*patch_size:patch_size+j*patch_size, :]

                for k in range(num_t):

                    t = random.random()
                    hazy_patch = free_patch * t + 255 * (1 - t)
                    
                    x_train.append(hazy_patch)
                    y_train.append(t)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    # to make the dimension of y_train equal with x_train's
    y_train = np.expand_dims(y_train,axis=-1)
    y_train = np.expand_dims(y_train,axis=-1)
    y_train = np.expand_dims(y_train,axis=-1)

    print('The shape of x_train: ',x_train.shape)
    print('The shape of y_train: ',y_train.shape)

    return x_train, y_train

def main():

    x_train, y_train = create_dataset('./haze_free',num_t=10,patch_size=16)

    batch_size = 500
    epochs = 400

    opt = optimizers.rmsprop(lr=0.01, decay=1e-4)

    model = create_dehaze_net()

    model.compile(optimizer=opt,
                loss='mse',
                metrics=[r2])

    history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs, validation_split=0.2,
                    shuffle=True)


    model.save('model.hdf5')

    print(history.history)

if __name__ == "__main__":
    main()

