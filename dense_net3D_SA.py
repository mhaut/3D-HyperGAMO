# For running in python 2.x
from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Dropout, Dense, RepeatVector, Lambda, Reshape, Conv3D, Conv2D, Flatten, InputSpec
from keras.layers import BatchNormalization, Concatenate, Multiply, Add, Conv2DTranspose, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU, Softmax
from keras.models import Model

def denseGamoGenCreate(latDim, num_class):
    noise = Input(shape=(latDim, ))
    labels = Input(shape=(num_class, ))
    gamoGenInput = Concatenate()([noise, labels])

    x = Dense(7 * 7 * 256, use_bias=False)(gamoGenInput)
    x = BatchNormalization(momentum=0.9)(x)
    x = LeakyReLU()(x)

    x = Reshape((7, 7, 256))(x)

    x = Conv2DTranspose(
        128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(
        64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(
        1, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = BatchNormalization(momentum=0.9)(x)

    gamoGenFinal = Flatten()(x)

    gamoGen = Model([noise, labels], gamoGenFinal)
    return gamoGen



def denseGenProcessCreate(numMinor, dataMinor,sh,mul):
    ip1=Input(shape=(784,))
    x=Dense(numMinor, activation='softmax')(ip1)
    x=RepeatVector(mul)(x)
    z = np.reshape(dataMinor,(numMinor,mul))
    genProcessFinal=Lambda(lambda x: K.sum(x*K.transpose(K.constant(z)), axis=2))(x)
    genProcessReshape = Reshape(sh)(genProcessFinal)
    genProcess=Model(ip1, genProcessReshape)
    return genProcess

def denseDisCreate(sh, num_class):
    imIn=Input(shape=sh)
    conv_layer1 = Conv3D(filters=8, kernel_size=(1, 1, 7), activation='relu', padding='same')(imIn)
    conv_layer2 = Conv3D(filters=16, kernel_size=(3, 3, 5), activation='relu',padding='same')(conv_layer1)
    conv_layer3 = Conv3D(filters=32, kernel_size=(5, 5, 7), activation='relu',padding='same')(conv_layer2)
    conv3d_shape = conv_layer3.shape
    conv_layer3 = Reshape((conv3d_shape[1], conv3d_shape[2], conv3d_shape[3]*conv3d_shape[4]))(conv_layer3)
    conv_layer4 = Conv2D(filters=64, kernel_size=(3,3), activation='relu',padding='same')(conv_layer3)
    conv_layer4 = GlobalAveragePooling2D()(conv_layer4)
    flatten_layer = Flatten()(conv_layer4)
    labels=Input(shape=(num_class,))
    disInput=Concatenate()([flatten_layer, labels])
    x=Dropout(0.5)(disInput)
    disFinal=Dense(1, activation='sigmoid', kernel_initializer="he_normal")(x)
    dis=Model([imIn, labels], disFinal)
    return dis

def denseMlpCreate(sh, num_class):
    imIn=Input(shape=sh)
    conv_layer1 = Conv3D(filters=8, kernel_size=(1, 1, 7), activation='relu', padding='same')(imIn)
    conv_layer2 = Conv3D(filters=16, kernel_size=(3, 3, 5), activation='relu',padding='same')(conv_layer1)
    conv_layer3 = Conv3D(filters=32, kernel_size=(5, 5, 7), activation='relu',padding='same')(conv_layer2)
    conv3d_shape = conv_layer3.shape
    conv_layer3 = Reshape((conv3d_shape[1], conv3d_shape[2], conv3d_shape[3]*conv3d_shape[4]))(conv_layer3)
    conv_layer4 = Conv2D(filters=64, kernel_size=(3,3), activation='relu',padding='same')(conv_layer3)
    conv_layer4 = GlobalAveragePooling2D()(conv_layer4)
    flatten_layer = Flatten()(conv_layer4)
    x=Dropout(0.5)(flatten_layer)
    mlpFinal = Dense(num_class, activation="softmax", kernel_initializer="he_normal")(x)
    mlp=Model(imIn, mlpFinal)
    return mlp
