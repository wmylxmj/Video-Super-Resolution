# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 20:38:49 2019

@author: wmy
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from keras import backend as K
from keras.losses import mean_absolute_error, mean_squared_error
from keras.models import load_model
from keras.optimizers import Adam
import random
import os
from model import VSR
from optimizer import AdamWithWeightsNormalization
from utils import DataLoader
from vsr import VideoSuperResolution

def make_dir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("You created a new path!")
        print("Path: " + str(path))
        pass
    else:
        print("Path: " + str(path) + " is already existed.")
        pass
    pass

vsr = VideoSuperResolution(pretrained_weights="weights/VSR-8-4-x4.h5")

for i in range(242, 250):
    video = r'D:\tianchi\testsets\bmp\youku_00200_00249_l\Youku_00' + str(i) + '_l'
    sp = r'D:\tianchi\video' + "\\" + str(i)
    make_dir(sp)
    vsr.predict(video, sp=sp)
    pass

