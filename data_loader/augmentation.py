'''
©2020 Siemens Corporation, Corporate Technology
All Rights Reserved

NOTICE: All information contained herein is, and remains, the property of Siemens Corporation, Corporate Technology.
It is proprietary to Siemens and may be covered by patent and copyright laws.

--------------------------
File name: augmentation.py
Date: 2020-06-08 20:38
Python Version: 3.6
'''

# ==============================================================================
# Imported Modules
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import typing

import torch
import numpy as np
import cv2
from scipy import ndimage
from PIL import Image
import torchvision.transforms.functional as TF

# ==============================================================================
# Constant Definitions
# ==============================================================================

# ==============================================================================
# Function Definitions
# ==============================================================================

# ==============================================================================
# Main Classes
# ==============================================================================


class Augmentor4SequencesOfImagesAndGT:

    def __init__(self,
                 output_size=(64, 64),
                 rotation_jitter=10.0,
                 translation_jitter=10.0,
                 # scale_jitter=.1,
                 crop_proportion=.1,
                 vflip=True,
                 hflip=True,
                 brightness_jitter=.2,
                 contrast_jitter=.2,
                 is_test_data=False):

        self.output_size = output_size
        self.rotation_jitter = rotation_jitter
        self.translation_jitter = translation_jitter
        # self.scale_jitter = scale_jitter
        self.crop_proportion = crop_proportion
        self.vflip = vflip
        self.hflip = hflip
        self.brightness_jitter = brightness_jitter
        self.contrast_jitter = contrast_jitter

        self.is_test_data = is_test_data

    def __call__(self, img_sequence, gt_sequence):

        aug_img_sequence, aug_gt_sequence = [], []
        if not self.is_test_data: # Randomly augment the training data:
            height, width = img_sequence[0].size

            # Sampling of augmentation parameters from normal distributions
            translation_var = [np.clip(np.random.normal(), -3, 3) * self.translation_jitter,
                               np.clip(np.random.normal(), -3, 3) * self.translation_jitter]
            rotate_var = np.clip(np.random.normal(), -3, 3) * self.rotation_jitter
            # scale_var = 1 + np.clip(np.random.normal(), -3, 3) * self.scale_jitter
            scale_var = 1.
            crop_size = np.round([height * (1 - np.clip(np.random.normal(), 0, 3) * self.crop_proportion),
                                  width * (1 - np.clip(np.random.normal(), 0, 3) * self.crop_proportion)])
            crop_topleft = [np.random.randint(0, height - crop_size[0] + 1),
                            np.random.randint(0, width - crop_size[1] + 1)]
            brightness_var = 1 + np.clip(np.random.normal(), -3, 3) * self.brightness_jitter
            contrast_var = 1 + np.clip(np.random.normal(), -3, 3) * self.contrast_jitter

            vflip = self.vflip and np.random.random() > 0.5
            hflip = self.hflip and np.random.random() > 0.5

            for img, gt in zip(img_sequence, gt_sequence):

                if translation_var[0] != 0 and translation_var[1] != 0 and rotate_var != 0 and scale_var != 1:
                    img = TF.affine(img, rotate_var, translation_var, scale_var, 0., resample=Image.BILINEAR)
                    gt = TF.affine(gt, rotate_var, translation_var, scale_var, 0., resample=Image.NEAREST)

                img = TF.resized_crop(img, *crop_topleft, *crop_size, self.output_size, interpolation=Image.BILINEAR)
                gt = TF.resized_crop(gt, *crop_topleft, *crop_size, self.output_size, interpolation=Image.NEAREST)

                if vflip:
                    img, gt = TF.vflip(img), TF.vflip(gt)
                if hflip:
                    img, gt = TF.hflip(img), TF.hflip(gt)

                img = TF.adjust_brightness(img, brightness_var)
                img = TF.adjust_contrast(img, contrast_var)

                aug_img_sequence.append(img)
                aug_gt_sequence.append(gt)

        else:
            for img, gt in zip(img_sequence, gt_sequence):
                img = TF.resize(img, self.output_size, interpolation=Image.BILINEAR)
                aug_img_sequence.append(img)

                if gt is not None:
                    gt = TF.resize(gt, self.output_size, interpolation=Image.NEAREST)
                aug_gt_sequence.append(gt)

        # Convert the data to tensors:
        conv_img_sequence, conv_gt_sequence = [], []
        for img, gt in zip(aug_img_sequence, aug_gt_sequence):
            img = TF.to_tensor(img)
            if gt is not None:
                gt = torch.from_numpy(np.array(gt, dtype='int16'))
            conv_img_sequence.append(img)
            conv_gt_sequence.append(gt)

        return conv_img_sequence, conv_gt_sequence

# ==============================================================================
# Main Function
# ==============================================================================