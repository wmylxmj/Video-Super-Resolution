# -*- coding: utf-8 -*-
"""
Created on Thu May  9 15:49:47 2019

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

class VideoSuperResolution(object):
    
    def __init__(self, scale=4, depth=7, num_blocks=8, nD=4, pretrained_weights=None, name=None):
        self.scale = scale
        self.depth = depth
        self.num_blocks = num_blocks
        self.nD = nD
        self.pretrained_weights = pretrained_weights
        self.name = name
        self.model = VSR(scale=scale, depth=depth, num_blocks=num_blocks, nD=nD)
        self.model.compile(optimizer=AdamWithWeightsNormalization(lr=0.001), \
                           loss=self.mae, metrics=[self.psnr])
        print("[OK] model created.")
        if pretrained_weights != None:
            self.model.load_weights(pretrained_weights)
            print("[OK] weights loaded.")
            pass
        self.data_loader = DataLoader(scale=scale, crop_size=256)
        self.default_weights_save_path = 'weights/VSR-' + \
        str(self.num_blocks) + "-" + str(self.nD) + '-x' + str(self.scale) + '.h5'
        pass
    
    def mae(self, hr, sr):
        margin = (tf.shape(hr)[1] - tf.shape(sr)[1]) // 2
        hr_crop = tf.cond(tf.equal(margin, 0), lambda: hr, lambda: hr[:, margin:-margin, margin:-margin, :])
        hr = K.in_train_phase(hr_crop, hr)
        hr.uses_learning_phase = True
        return mean_absolute_error(hr, sr)

    def psnr(self, hr, sr):
        margin = (tf.shape(hr)[1] - tf.shape(sr)[1]) // 2
        hr_crop = tf.cond(tf.equal(margin, 0), lambda: hr, lambda: hr[:, margin:-margin, margin:-margin, :])
        hr = K.in_train_phase(hr_crop, hr)
        hr.uses_learning_phase = True
        return tf.image.psnr(hr, sr, max_val=255)
    
    def train(self, epoches=10000, batch_size=1, weights_save_path=None, sample=True):
        if weights_save_path == None:
            weights_save_path = self.default_weights_save_path
            pass
        for epoch in range(epoches):
            for batch_i, (X, Y) in enumerate(self.data_loader.batches(batch_size=batch_size)):
                temp_loss, temp_psnr = self.model.train_on_batch(X, Y)
                print("[epoch: {}/{}][batch: {}/{}][loss: {}][psnr: {}]".format(epoch+1, epoches, \
                      batch_i+1, self.data_loader.n_batches, temp_loss, temp_psnr))
                if (batch_i+1) % 25 == 0:
                    self.model.save_weights(weights_save_path)
                    print("[OK] weights saved.")
                    pass
                if sample:
                    if (batch_i+1) % 500 == 0:
                        self.sample(epoch=epoch+1, batch=batch_i+1)
                        pass
                    pass
                pass
            self.model.save_weights(weights_save_path)
            print("[OK] weights saved.")
            pass
        pass
    
    def sample(self, mode='train', save_folder='samples', epoch=1, batch=1):
        video_pairs = self.data_loader.search(mode=mode)
        frames_pairs = self.data_loader.load(video_pairs)
        sliced_frames_pairs = self.data_loader.slice(frames_pairs)
        np.random.shuffle(sliced_frames_pairs)
        lrs, hrs = random.choice(sliced_frames_pairs)
        lrs, hrs = self.data_loader.pair(lrs, hrs, rotate=False, flip=False, crop=False)
        X = np.array([lrs])
        lr = np.array(lrs[self.depth//2])
        hr = np.array(hrs[self.depth//2])
        hr = Image.fromarray(hr)
        lr = Image.fromarray(lr)
        lr_resize = lr.resize(hr.size)
        Y = self.model.predict(X)[0]
        sr = np.clip(Y, 0, 255)
        sr = sr.astype('uint8')
        sr = Image.fromarray(sr)
        lr_resize.save(save_folder + "/" + "epoch_" + str(epoch) + "_batch_" + str(batch) + "_lr.jpg")
        sr.save(save_folder + "/" + "epoch_" + str(epoch) + "_batch_" + str(batch) + "_sr.jpg")
        hr.save(save_folder + "/" + "epoch_" + str(epoch) + "_batch_" + str(batch) + "_hr.jpg")
        pass
    
    def predict(self, video, sp='./video', fn='frame'):
        frames = []
        images = os.listdir(video)
        num_frames = len(images)
        for image in images:
            frames.append(os.path.join(video, image))
            pass
        for i in range((self.depth-1)//2):
            frames.insert(0, 'black')
            frames.append('black')
            pass
        lrs = []
        w = 0
        h = 0
        for frame in frames:
            if frame != 'black':
                frame = Image.open(frame)
                w, h = frame.size
                break
            pass
        if w==0 or h==0:
            raise ValueError("no vaild frame in video")
        black = np.zeros((h, w, 3), dtype=np.uint8)
        black_image = Image.fromarray(black)
        for frame in frames:
            if frame == 'black':
                image = black_image
                lrs.append(image)
                pass
            else:
                image = Image.open(frame)
                lrs.append(image)
                pass
            pass
        lrs_array = []
        for lr in lrs:
            lr = np.asarray(lr)
            lrs_array.append(lr)
            pass
        for i in range(num_frames):
            lrs_shuffed = lrs_array[i:i+self.depth]
            X = np.array([lrs_shuffed])
            Y = self.model.predict(X)[0]
            sr = np.clip(Y, 0, 255)
            sr = sr.astype('uint8')
            sr = Image.fromarray(sr)
            file_name = fn + '_' + '0'*(3-len(str(i+1))) + str(i+1) + '.bmp'
            sr.save(os.path.join(sp, file_name))
            print("[OK] frame " + str(i+1) + " saved.")
            pass
        pass
    
    def evaluate(self, mode='val', batch_size=1):
        loss_sum = 0.0
        psnr_sum = 0.0
        for batch_i, (X, Y) in enumerate(self.data_loader.batches(batch_size=batch_size, mode=mode)):
            temp_loss, temp_psnr = self.model.test_on_batch(X, Y)
            loss_sum += temp_loss
            psnr_sum += temp_psnr
            print("[after batch: {}/{}][avg loss: {}][avg psnr: {}]".format((batch_i+1), \
                  self.data_loader.n_batches, loss_sum/(batch_i+1), psnr_sum/(batch_i+1)))
            pass
        pass
    
    pass
    
