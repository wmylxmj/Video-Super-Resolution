# -*- coding: utf-8 -*-
"""
Created on Thu May  9 16:11:59 2019

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
from optimizer import AdamWithWeightsNormalization
from utils import DataLoader
from vsr import VideoSuperResolution

vsr = VideoSuperResolution(pretrained_weights="weights/VSR-8-4-x4.h5")
vsr.data_loader.crop_size = 128
vsr.train(sample=False)

