# -*- coding: utf-8 -*-
"""
Created on Wed May  8 22:38:18 2019

@author: wmy
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import random
from PIL import Image
from PIL import ImageFilter

class DataLoader(object):
    
    def __init__(self, scale=4, depth=7, crop_size=96, name=None):
        self.__scale = 4
        self.__depth = 7
        self.__crop_size = 96
        self.scale = scale 
        self.depth = depth
        self.crop_size = crop_size
        self.name = name
        pass
    
    @property
    def scale(self):
        return self.__scale
    
    @scale.setter
    def scale(self, value):
        if not isinstance(value, int):
            raise ValueError("scale must be int")
        elif value <= 0:
            raise ValueError("scale must > 0")
        else:
            self.__scale = value
            pass
        pass
    
    @property
    def depth(self):
        return self.__depth
    
    @depth.setter
    def depth(self, value):
        if not isinstance(value, int):
            raise ValueError("depth must be int")
        elif value <= 0:
            raise ValueError("depth must > 0")
        elif value % 2 == 0:
            raise ValueError("bad depth")
        else:
            self.__depth = value
            pass
        pass
    
    @property
    def crop_size(self):
        return self.__crop_size
    
    @crop_size.setter
    def crop_size(self, value):
        if not isinstance(value, int):
            raise ValueError("crop size must be int")
        elif value <= 0:
            raise ValueError("crop size must > 0")
        else:
            self.__crop_size = value
            pass
        pass
    
    def imread(self, path):
        return Image.open(path)
    
    def rotate(self, lrs, hrs):
        angle = random.choice([0, 90, 180, 270])
        lrs_rotated = []
        for lr in lrs:
            lr_rotated = lr.rotate(angle, expand=True)
            lrs_rotated.append(lr_rotated)
            pass
        hrs_rotated = []
        for hr in hrs:
            hr_rotated = hr.rotate(angle, expand=True)
            hrs_rotated.append(hr_rotated)
            pass
        return lrs_rotated, hrs_rotated
    
    def flip(self, lrs, hrs):
        mode = random.choice([0, 1, 2, 3])
        lrs_flipped = []
        hrs_flipped = []
        if mode == 0:
            lrs_flipped = lrs
            hrs_flipped = hrs
            pass
        elif mode == 1:
            for lr in lrs:
                lr_flipped = lr.transpose(Image.FLIP_LEFT_RIGHT)
                lrs_flipped.append(lr_flipped)
                pass
            for hr in hrs:
                hr_flipped = hr.transpose(Image.FLIP_LEFT_RIGHT)
                hrs_flipped.append(hr_flipped)
                pass
            pass
        elif mode == 2:
            for lr in lrs:
                lr_flipped = lr.transpose(Image.FLIP_TOP_BOTTOM)
                lrs_flipped.append(lr_flipped)
                pass
            for hr in hrs:
                hr_flipped = hr.transpose(Image.FLIP_TOP_BOTTOM)
                hrs_flipped.append(hr_flipped)
                pass
            pass
        elif mode == 3:
            for lr in lrs:
                lr_flipped = lr.transpose(Image.FLIP_LEFT_RIGHT)
                lr_flipped = lr_flipped.transpose(Image.FLIP_TOP_BOTTOM)
                lrs_flipped.append(lr_flipped)
                pass
            for hr in hrs:
                hr_flipped = hr.transpose(Image.FLIP_LEFT_RIGHT)
                hr_flipped = hr_flipped.transpose(Image.FLIP_TOP_BOTTOM)
                hrs_flipped.append(hr_flipped)
                pass
            pass
        return lrs_flipped, hrs_flipped
    
    def crop(self, lrs, hrs):
        hr_crop_size = self.crop_size
        lr_crop_size = hr_crop_size//self.scale
        lr_sample = lrs[0]
        lr_w = np.random.randint(lr_sample.size[0] - lr_crop_size + 1)
        lr_h = np.random.randint(lr_sample.size[1] - lr_crop_size + 1)
        hr_w = lr_w * self.scale
        hr_h = lr_h * self.scale
        lrs_cropped = []
        hrs_cropped = []
        for lr in lrs:
            lr_cropped = lr.crop([lr_w, lr_h, lr_w+lr_crop_size, lr_h+lr_crop_size])
            lrs_cropped.append(lr_cropped)
            pass
        for hr in hrs:
            hr_cropped = hr.crop([hr_w, hr_h, hr_w+hr_crop_size, hr_h+hr_crop_size])
            hrs_cropped.append(hr_cropped)
            pass
        return lrs_cropped, hrs_cropped
    
    def pad(self, frames):
        for i in range((self.depth-1)//2):
            frames.insert(0, 'black')
            frames.append('black')
            pass
        return frames
    
    def pair(self, lrs_paths, hrs_paths, rotate=True, flip=True, crop=True):
        lrs = []
        hrs = []
        lr_w = 0
        lr_h = 0
        hr_w = 0
        hr_h = 0
        for lr_path in lrs_paths:
            if lr_path != 'black':
                lr = self.imread(lr_path)
                lr_w, lr_h = lr.size
                break
            pass
        for hr_path in hrs_paths:
            if hr_path != 'black':
                hr = self.imread(hr_path)
                hr_w, hr_h = hr.size
                break
            pass
        if lr_w==0 or lr_h==0 or hr_w==0 or hr_h==0:
            raise ValueError("no vaild path in paths")
        lr_black = np.zeros((lr_h, lr_w, 3), dtype=np.uint8)
        lr_black_image = Image.fromarray(lr_black)
        hr_black = np.zeros((hr_h, hr_w, 3), dtype=np.uint8)
        hr_black_image = Image.fromarray(hr_black)
        for lr_path in lrs_paths:
            if lr_path == 'black':
                lr = lr_black_image
                lrs.append(lr)
                pass
            else:
                lr = self.imread(lr_path)
                lrs.append(lr)
                pass
            pass
        for hr_path in hrs_paths:
            if hr_path == 'black':
                hr = hr_black_image
                hrs.append(hr)
                pass
            else:
                hr = self.imread(hr_path)
                hrs.append(hr)
                pass
            pass
        if rotate:
            lrs, hrs = self.rotate(lrs, hrs)
            pass
        if flip:
            lrs, hrs = self.flip(lrs, hrs)
            pass
        if crop:
            lrs, hrs = self.crop(lrs, hrs)
            pass
        lrs, hrs = self.asarray(lrs, hrs)
        return lrs, hrs
    
    def asarray(self, lrs, hrs):
        lrs_array = []
        hrs_array = []
        for lr in lrs:
            lr = np.asarray(lr)
            lrs_array.append(lr)
            pass
        for hr in hrs:
            hr = np.asarray(hr)
            hrs_array.append(hr)
            pass
        return lrs_array, hrs_array
    
    def search(self, mode='train'):
        video_pairs = []
        if mode=='train':
            for i in range(50):
                video_lr = 'D:/tianchi/datasets/bmp/youku_00000_00049_l/Youku_' + \
                '0' * (5-len(str(i))) + str(i) + '_l'
                video_hr = 'D:/tianchi/datasets/bmp/youku_00000_00049_h_GT/Youku_' + \
                '0' * (5-len(str(i))) + str(i) + '_h_GT'
                video_pairs.append((video_lr, video_hr))
                pass
            for i in range(50, 100):
                video_lr = 'D:/tianchi/datasets/bmp/youku_00050_00099_l/Youku_' + \
                '0' * (5-len(str(i))) + str(i) + '_l'
                video_hr = 'D:/tianchi/datasets/bmp/youku_00050_00099_h_GT/Youku_' + \
                '0' * (5-len(str(i))) + str(i) + '_h_GT'
                video_pairs.append((video_lr, video_hr))
                pass
            for i in range(100, 150):
                video_lr = 'D:/tianchi/datasets/bmp/youku_00100_00149_l/Youku_' + \
                '0' * (5-len(str(i))) + str(i) + '_l'
                video_hr = 'D:/tianchi/datasets/bmp/youku_00100_00149_h_GT/Youku_' + \
                '0' * (5-len(str(i))) + str(i) + '_h_GT'
                video_pairs.append((video_lr, video_hr))
                pass
            pass
        elif mode=='val':
            for i in range(150, 200):
                video_lr = 'D:/tianchi/datasets/bmp/youku_00150_00199_l/Youku_' + \
                '0' * (5-len(str(i))) + str(i) + '_l'
                video_hr = 'D:/tianchi/datasets/bmp/youku_00150_00199_h_GT/Youku_' + \
                '0' * (5-len(str(i))) + str(i) + '_h_GT'
                video_pairs.append((video_lr, video_hr))
                pass
            pass
        else:
            raise ValueError("unknown mode")
        return video_pairs
    
    def load(self, video_pairs):
        frames_pairs = []
        for video_pair in video_pairs:
            video_lr, video_hr = video_pair
            frames_lr = []
            frames = os.listdir(video_lr)
            frames = sorted(frames)
            for frame in frames:
                frames_lr.append(os.path.join(video_lr, frame))
                pass
            frames_lr = self.pad(frames_lr)
            frames_hr = []
            frames = os.listdir(video_hr)
            frames = sorted(frames)
            for frame in frames:
                frames_hr.append(os.path.join(video_hr, frame))
                pass
            frames_hr = self.pad(frames_hr)
            frames_pairs.append((frames_lr, frames_hr))
            pass
        return frames_pairs
        
    def slice(self, frames_pairs):
        sliced_frames_pairs = []
        for frames_pair in frames_pairs:
            num_frames = len(frames_pair[0]) - self.depth + 1
            frames_lr, frames_hr = frames_pair
            for i in range(num_frames):
                sliced_frames_lr = frames_lr[i:i+self.depth]
                sliced_frames_hr = frames_hr[i:i+self.depth]
                sliced_frames_pair = (sliced_frames_lr, sliced_frames_hr)
                sliced_frames_pairs.append(sliced_frames_pair)
                pass
            pass
        return sliced_frames_pairs
    
    def batches(self, batch_size=1, mode='train', complete_batch_only=False):
        video_pairs = self.search(mode=mode)
        frames_pairs = self.load(video_pairs)
        sliced_frames_pairs = self.slice(frames_pairs)
        np.random.shuffle(sliced_frames_pairs)
        n_complete_batches = int(len(sliced_frames_pairs)/batch_size)
        self.n_batches = int(len(sliced_frames_pairs) / batch_size)
        have_res_batch = (len(sliced_frames_pairs)/batch_size) > n_complete_batches
        if have_res_batch and complete_batch_only==False:
            self.n_batches += 1
            pass
        for i in range(n_complete_batches):
            batch = sliced_frames_pairs[i*batch_size:(i+1)*batch_size]
            X, Y = [], []
            for (lrs, hrs) in batch:
                lrs, hrs = self.pair(lrs, hrs)
                X.append(lrs)
                Y.append(hrs[self.depth//2])
                pass
            X = np.array(X)
            Y = np.array(Y)
            yield X, Y
        if self.n_batches > n_complete_batches:
            batch = sliced_frames_pairs[n_complete_batches*batch_size:]
            X, Y = [], []
            for (lrs, hrs) in batch:
                lrs, hrs = self.pair(lrs, hrs)
                X.append(lrs)
                Y.append(hrs[self.depth//2])
                pass
            X = np.array(X)
            Y = np.array(Y)
            yield X, Y
        pass
    
    pass

