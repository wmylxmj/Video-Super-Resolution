# -*- coding: utf-8 -*-
"""
Created on Tue May 14 14:23:34 2019

@author: wmy
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import backend as K
from keras.layers import Add, Conv2D, Input, Lambda, Activation
from keras.models import Model
from keras.layers import Conv3D, ZeroPadding3D, BatchNormalization
from keras.layers import LeakyReLU, concatenate, Reshape, Softmax
from IPython.display import SVG
from keras.utils import plot_model
from keras import layers
from PIL import Image

def SubpixelConv2D(scale, **kwargs):
    return Lambda(lambda x: tf.depth_to_space(x, scale), **kwargs)

def Normalization(**kwargs):
    rgb_mean = np.array([0.5, 0.5, 0.5]) * 255
    return Lambda(lambda x: (x - rgb_mean) / 127.5, **kwargs)

def Denormalization(**kwargs):
    rgb_mean = np.array([0.5, 0.5, 0.5]) * 255
    return Lambda(lambda x: x * 127.5 + rgb_mean, **kwargs)

def DenseBlock(X, F, G, D):
    for i in range(D):
        T = BatchNormalization(axis=4)(X)
        T = LeakyReLU(alpha=0.1)(T)
        T = Conv3D(filters=F, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='valid')(T)
        T = BatchNormalization(axis=4)(T)
        T = LeakyReLU(alpha=0.1)(T)
        T = ZeroPadding3D(padding=(1, 1, 1))(T)
        T = Conv3D(filters=G, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='valid')(T)
        X = concatenate([X, T], axis=4)
        F += G
        pass
    return X

def ResidualDenseBlock(X, num_channals, F, G, D):
    branch = DenseBlock(X, F, G, D)
    branch = Conv3D(filters=num_channals, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='valid')(branch)
    X = Add()([branch, X])
    return X
    
def VSR(scale=4, depth=7, num_blocks=8, nD=4, name=None):
    if depth % 2 == 0:
        raise ValueError("bad depth")
    Input_Tensor = Input((depth, None, None, 3))
    X_in = Normalization()(Input_Tensor)
    X = ZeroPadding3D(padding=(0, 3, 3))(X_in)
    X = Conv3D(filters=64, kernel_size=(1, 7, 7), strides=(1, 1, 1), padding='valid')(X)
    # residual dense blocks
    for i in range(num_blocks):
        X = ResidualDenseBlock(X, 64, 64, 16, nD)
        pass
    # depth turn to 1
    num_conv = int((depth-1)/2)
    nF = 128
    nG = 64
    for i in range(num_conv):
        T = BatchNormalization(axis=4)(X)
        T = LeakyReLU(alpha=0.1)(T)
        T = Conv3D(filters=nF, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='valid')(T)
        T = BatchNormalization(axis=4)(T)
        T = LeakyReLU(alpha=0.1)(T)
        T = ZeroPadding3D(padding=(0, 1, 1))(T)
        T = Conv3D(filters=nG, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='valid')(T)
        S = Lambda(lambda x: x[:, 1:-1])(X)
        X = concatenate([S, T], axis=4)
        nF += nG
        pass
    X = BatchNormalization(axis=4)(X)
    X = LeakyReLU(alpha=0.1)(X)
    X = ZeroPadding3D(padding=(0, 1, 1))(X)
    X = Conv3D(filters=256, kernel_size=(1, 3, 3), strides=(1, 1, 1), padding='valid')(X)
    X = LeakyReLU(alpha=0.1)(X)
    R = Conv3D(filters=256, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='valid')(X)
    R = LeakyReLU(alpha=0.1)(R)
    R = Conv3D(filters=3*scale*scale, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='valid')(R)
    F = Conv3D(filters=512, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='valid')(X)
    F = LeakyReLU(alpha=0.1)(F)
    F = Conv3D(filters=1*5*5*scale*scale, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='valid')(F)
    shape = (K.shape(F)[0], K.shape(F)[1], K.shape(F)[2], K.shape(F)[3], 25, scale*scale)
    F = Lambda(K.reshape, arguments={'shape':shape})(F)
    F = Softmax(axis=4)(F)
    # dynamic filter
    filter_localexpand_np = np.reshape(np.eye(25, 25), (5, 5, 1, 25))
    filter_localexpand = K.constant(filter_localexpand_np, dtype='float32')
    x_channals = []
    for c in range(3):
        x = Lambda(lambda x: x[:, depth//2:depth//2+1, :, :, c])(X_in)        
        f = Lambda(lambda x: x[:, 0, :, :, :, :])(F)
        x = Lambda(tf.transpose, arguments={'perm':[0,2,3,1]})(x) 
        x_localexpand = Lambda(K.conv2d, arguments={'kernel':filter_localexpand, \
                                                    'strides':(1, 1), 'padding':'same'})(x)
        x_localexpand = Lambda(tf.expand_dims, arguments={'axis':3})(x_localexpand)
        x = Lambda(tf.matmul, arguments={'b':f})(x_localexpand)    
        x = Lambda(tf.squeeze, arguments={'axis':3})(x)
        x = SubpixelConv2D(scale=scale)(x)
        x_channals.append(x)
        pass
    x = concatenate(x_channals, axis=3)
    x = Lambda(tf.expand_dims, arguments={'axis':1})(x) 
    # depth to space 3D
    r_shape = K.shape(R)
    shape = (r_shape[0]*r_shape[1], r_shape[2], r_shape[3], r_shape[4])
    r = Lambda(K.reshape, arguments={'shape':shape})(R)
    y = Lambda(tf.depth_to_space, arguments={'block_size':scale})(r)
    y_shape = K.shape(y)
    shape = (r_shape[0], r_shape[1], y_shape[1], y_shape[2], y_shape[3])
    r = Lambda(K.reshape, arguments={'shape':shape})(y)
    # add
    G = layers.add([x, r])
    G = Lambda(lambda x: x[:, 0, :, :, :])(G)
    Output_Tensor = Denormalization()(G)
    if name==None:
        name = "VSR-" + str(num_blocks) + "-" + str(nD) + "-x" + str(scale)
        pass
    model = Model(Input_Tensor, Output_Tensor, name=name)
    return model

