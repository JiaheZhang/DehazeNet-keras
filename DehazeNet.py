import os
import numpy as np
import time
from keras.models import Model
from keras.layers import Conv2D, Input, Concatenate
from keras.layers import MaxPool2D, Activation
  
from keras import backend as K
from keras.engine.topology import Layer

###########################################3
# Dehaze Net keras+tensorflow version 

# choose a GTX 
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
##############################################################################
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
#################################################################################

#################################################################################
def BRelu(x):
    return K.relu(x, max_value=1)
################################################################################

input_1 = Input(shape=(16, 16, 3))

###############  conv2d  ##########
conv_1 = Conv2D(filters=4, kernel_size=(5, 5), strides=(1, 1))(input_1)

conv_2 = Conv2D(filters=4, kernel_size=(5, 5), strides=(1, 1))(input_1)

conv_3 = Conv2D(filters=4, kernel_size=(5, 5), strides=(1, 1))(input_1)

conv_4 = Conv2D(filters=4, kernel_size=(5, 5), strides=(1, 1))(input_1)

############# Maxout layer ##########
max_out_1 = MaxoutConv2D()(conv_1)

max_out_2 = MaxoutConv2D()(conv_2)

max_out_3 = MaxoutConv2D()(conv_3)

max_out_4 = MaxoutConv2D()(conv_4)

max_out = Concatenate()([max_out_1, max_out_2, max_out_3, max_out_4])


#################### multi-scale conv ##########
multi_layer_1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(max_out)

multi_layer_2 = Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same')(max_out)

multi_layer_3 = Conv2D(filters=16, kernel_size=(7, 7), strides=(1, 1), padding='same')(max_out)

multi_layer = Concatenate()([multi_layer_1, multi_layer_2, multi_layer_3])

############### max pool###############
max_pool_1 = MaxPool2D(pool_size=(7, 7), strides=(1,1))(multi_layer)

conv_5 = Conv2D(filters=1, kernel_size=(6, 6), strides=(1, 1))(max_pool_1)

output_1 = Activation(BRelu)(conv_5)


model = Model(inputs=input_1, outputs=output_1)

model.summary()

## r2
def r2(y_true, y_pred):
    return 1 - K.sum(K.square(y_pred - y_true)) / K.sum(K.square(y_true - K.mean(y_true)))


batch_size = 500
epochs = 400

opt = optimizers.rmsprop(lr=0.01, decay=1e-4)


model.compile(optimizer=opt,
              loss='mse',
              metrics=[r2])

history = model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs, validation_split=0.2,
                  shuffle=True)


model.save('model.hdf5')

print(history.history)
