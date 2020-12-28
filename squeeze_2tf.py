import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model

SqueezeNet(
  (features): Sequential(
    (0): Conv2d(3, 96, kernel_size=(7, 7), stride=(2, 2))
    (1): ReLU(inplace=True)
    (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
    (3): Fire(
      (squeeze): Conv2d(96, 16, kernel_size=(1, 1), stride=(1, 1))
      (squeeze_activation): ReLU(inplace=True)
      (expand1x1): Conv2d(16, 64, kernel_size=(1, 1), stride=(1, 1))
      (expand1x1_activation): ReLU(inplace=True)
      (expand3x3): Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (expand3x3_activation): ReLU(inplace=True)
    )
    (4): Fire(
      (squeeze): Conv2d(128, 16, kernel_size=(1, 1), stride=(1, 1))
      (squeeze_activation): ReLU(inplace=True)
      (expand1x1): Conv2d(16, 64, kernel_size=(1, 1), stride=(1, 1))
      (expand1x1_activation): ReLU(inplace=True)
      (expand3x3): Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (expand3x3_activation): ReLU(inplace=True)
    )
    (5): Fire(
      (squeeze): Conv2d(128, 32, kernel_size=(1, 1), stride=(1, 1))
      (squeeze_activation): ReLU(inplace=True)
      (expand1x1): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
      (expand1x1_activation): ReLU(inplace=True)
      (expand3x3): Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (expand3x3_activation): ReLU(inplace=True)
    )
    (6): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
    (7): Fire(
      (squeeze): Conv2d(256, 32, kernel_size=(1, 1), stride=(1, 1))
      (squeeze_activation): ReLU(inplace=True)
      (expand1x1): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
      (expand1x1_activation): ReLU(inplace=True)
      (expand3x3): Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (expand3x3_activation): ReLU(inplace=True)
    )
    (8): Fire(
      (squeeze): Conv2d(256, 48, kernel_size=(1, 1), stride=(1, 1))
      (squeeze_activation): ReLU(inplace=True)
      (expand1x1): Conv2d(48, 192, kernel_size=(1, 1), stride=(1, 1))
      (expand1x1_activation): ReLU(inplace=True)
      (expand3x3): Conv2d(48, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (expand3x3_activation): ReLU(inplace=True)
    )
    (9): Fire(
      (squeeze): Conv2d(384, 48, kernel_size=(1, 1), stride=(1, 1))
      (squeeze_activation): ReLU(inplace=True)
      (expand1x1): Conv2d(48, 192, kernel_size=(1, 1), stride=(1, 1))
      (expand1x1_activation): ReLU(inplace=True)
      (expand3x3): Conv2d(48, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (expand3x3_activation): ReLU(inplace=True)
    )
    (10): Fire(
      (squeeze): Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1))
      (squeeze_activation): ReLU(inplace=True)
      (expand1x1): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1))
      (expand1x1_activation): ReLU(inplace=True)
      (expand3x3): Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (expand3x3_activation): ReLU(inplace=True)
    )
    (11): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
    (12): Fire(
      (squeeze): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1))
      (squeeze_activation): ReLU(inplace=True)
      (expand1x1): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1))
      (expand1x1_activation): ReLU(inplace=True)
      (expand3x3): Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (expand3x3_activation): ReLU(inplace=True)
    )
  )
  (classifier): Sequential(
    (0): Dropout(p=0.5, inplace=False)
    (1): Conv2d(512, 1000, kernel_size=(1, 1), stride=(1, 1))
    (2): ReLU(inplace=True)
    (3): AdaptiveAvgPool2d(output_size=(1, 1))
  )
)

class fire(tf.keras.Model):
    
    def __init__(self, squeeze, expand1x1, expand3x3):
        super(fire, self).__init__()
        
        self.squeeze = layers.Conv2D(squeeze, (1, 1))
        self.squeeze_activation = layers.ReLU()
        self.expand1x1 = layers.Conv2D(expand1x1, (1, 1))
        self.expand3x3 = layers.Conv2D(expand3x3, (3, 3), padding='same')
        self.concatenate = layers.Concatenate(axis=1)
        
    def call(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        
        return self.concatenate([self.squeeze_activation(self.expand1x1(x)),
                                self.squeeze_activation(self.expand3x3(x))])
                                

class SqueezeNet(tf.keras.Model):
    
    def __init__(self, version='1.0', num_classes=1000):
        super(SqueezeNet, self).__init__()
        
        self.num_classes = num_classes
        self.features = tf.keras.Sequential([
                layers.Conv2D(96, (7, 7), strides=2),
                layers.ReLU(),
                layers.MaxPool2D(pool_size=(3, 3), strides=2),
                fire(16, 64, 64),
                fire(16, 64, 64),
                fire(32, 128, 128),
                layers.MaxPool2D(pool_size=(3, 3), strides=2),
                fire(32, 128, 128),
                fire(48, 192, 192),
                fire(48, 192, 192),
                layers.MaxPool2D(pool_size=(3, 3), strides=2),
                fire(64, 256, 256)
            ])
            
#         elif version =='1.1':
#             self.features = tf.keras.Sequential([
#                 layers.Conv2D(64, (3, 3), strides=2),
#                 layers.ReLU(),
#                 layers.MaxPool2D(pool_size=(3, 3), strides=2),
#                 fire(16, 64, 64),
#                 fire(16, 64, 64),
#                 layers.MaxPool2D(pool_size=(3, 3), strides=2),
#                 fire(32, 128, 128),
#                 fire(32, 128, 128),
#                 layers.MaxPool2D(pool_size=(3, 3), strides=2),
#                 fire(48, 192, 192),
#                 fire(48, 192, 192),
#                 fire(64, 256, 256),
#                 fire(64, 256, 256),
#             ])
            
        final_conv = layers.Conv2D(self.num_classes, (1, 1))
        
        self.classifier = tf.keras.Sequential([
            layers.Dropout(rate=0.5),
            final_conv,
            layers.ReLU(),
            layers.GlobalAveragePooling2D(),
        ])
        self.flatten = layers.Flatten()
        
        
    def call(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = self.flatten(x)
        
        return x
    
def _squeezenet(version, **kwargs):
    model = SqueezeNet(version, **kwargs)
    return model


def squeezenet1_0(**kwargs):
    r"""SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.
    """
    return _squeezenet('1_0',**kwargs)


def squeezenet1_1(**kwargs):
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.
    """
    return _squeezenet('1_1', **kwargs)


if __name__ == "__main__":
    model = squeezenet1_0()
    out = model(tf.ones([10,224,224,3]))
    print(out)


model.save_weights('test.h5')

import h5py
import os, shutil
import matplotlib.pyplot as plt
from keras import backend
from keras.models import Model
from keras.layers import Add, Activation, Concatenate, Conv2D, Dropout 
from keras.layers import Flatten, Input, GlobalAveragePooling2D, MaxPooling2D

def SqueezeNet(input_shape, nb_classes, dropout_rate=None, compression=1.0):
    input_img = Input(shape=input_shape)

    x = Conv2D(int(64 * compression), (3, 3), activation='relu', strides=(2, 2), padding='same', name='conv1')(input_img)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool1')(x)
    
    x = fire_module(x, int(16 * compression), name='fire2')
    x = fire_module(x, int(16 * compression), name='fire3')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool3')(x)
    
    x = fire_module(x, int(32 * compression), name='fire4')
    x = fire_module(x, int(32 * compression), name='fire5')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool5')(x)
    
    x = fire_module(x, int(48 * compression), name='fire6')
    x = fire_module(x, int(48 * compression), name='fire7')
    x = fire_module(x, int(64 * compression), name='fire8')
    x = fire_module(x, int(64 * compression), name='fire9')

    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    
    # Creating last conv10
    x = output(x, nb_classes)

    return Model(inputs=input_img, outputs=x)

# Create fire module for SqueezeNet
# x                 : input (keras.layers)
# nb_squeeze_filter : number of filters for Squeezing. Filter size of expanding is 4x of Squeezing filter size
# name              : name of module
# RETURNS fire module x
def fire_module(x, nb_squeeze_filter, name, use_bypass=False):
    nb_expand_filter = 4 * nb_squeeze_filter
    squeeze    = Conv2D(nb_squeeze_filter,(1, 1), activation='relu', padding='same', name='%s_squeeze'%name)(x)
    expand_1x1 = Conv2D(nb_expand_filter, (1, 1), activation='relu', padding='same', name='%s_expand_1x1'%name)(squeeze)
    expand_3x3 = Conv2D(nb_expand_filter, (3, 3), activation='relu', padding='same', name='%s_expand_3x3'%name)(squeeze)
    
    if backend.image_data_format() == 'channels_last':
        axis = -1
    else:
        axis = 1

    x_ret = Concatenate(axis=axis, name='%s_concatenate'%name)([expand_1x1, expand_3x3])
    
    if use_bypass:
        x_ret = Add(name='%s_concatenate_bypass'%name)([x_ret, x])
        
    return x_ret

def output(x, nb_classes):
    x = Conv2D(nb_classes, (1, 1), strides=(1, 1), padding='valid', name='conv10')(x)
    x = GlobalAveragePooling2D(name='avgpool10')(x)
    x = Activation("softmax", name='softmax')(x)
    return x
