'''
©2020 Siemens Corporation, Corporate Technology
All Rights Reserved

NOTICE: All information contained herein is, and remains, the property of Siemens Corporation, Corporate Technology.
It is proprietary to Siemens and may be covered by patent and copyright laws.

--------------------------
File based on:
--------------------------
Joint-Learning-of-Motion-Estimation-and-Segmentation-for-Cardiac-MR-Image-Sequences

MIT License

Copyright (c) 2019 Imperial College London

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

-----

Citation and Acknowledgement

If you use the code for your work, or if you found the code useful, please cite the following works:

Qin, C., Bai, W., Schlemper, J., Petersen, S.E., Piechnik, S.K., Neubauer, S. and Rueckert, D. Joint learning of motion
estimation and segmentation for cardiac MR image sequences. In International Conference on Medical Image Computing and
Computer-Assisted Intervention (MICCAI), 2018: 472-480.

C. Qin, W. Bai, J. Schlemper, S. Petersen, S. Piechnik, S. Neubauer and D. Rueckert. Joint Motion Estimation and
Segmentation from Undersampled Cardiac MR Image. International Workshop on Machine Learning for Medical Image
Reconstruction, 2018: 55-63.

--------------------------
Edits made to original file:
--------------------------
> Added utils functions
'''

import numpy as np
import torch
import torch.nn as nn


def generate_grid(x, offset):
    x_shape = x.size()
    grid_w, grid_h = torch.meshgrid([torch.linspace(-1, 1, x_shape[2]), torch.linspace(-1, 1, x_shape[3])])  # (h, w)
    grid_w = grid_w.cuda().float()
    grid_h = grid_h.cuda().float()

    grid_w = nn.Parameter(grid_w, requires_grad=False)
    grid_h = nn.Parameter(grid_h, requires_grad=False)

    offset_h, offset_w = torch.split(offset, 1, 1)
    offset_w = offset_w.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))  # (b*c, h, w)
    offset_h = offset_h.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))  # (b*c, h, w)

    offset_w = grid_w + offset_w
    offset_h = grid_h + offset_h

    offsets = torch.stack((offset_h, offset_w), 3)
    return offsets


def convert_to_1hot(label, n_class):
    # Convert a label map (N x 1 x H x W) into a one-hot representation (N x C x H x W)
    label_swap = label.swapaxes(1, 3)
    label_flat = label_swap.flatten()
    n_data = len(label_flat)
    label_1hot = np.zeros((n_data, n_class), dtype='int16')
    label_1hot[range(n_data), label_flat] = 1
    label_1hot = label_1hot.reshape((label_swap.shape[0], label_swap.shape[1], label_swap.shape[2], n_class))
    label_1hot = label_1hot.swapaxes(1, 3)
    return label_1hot


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".nii.gz"])


def crop_and_fill(img,size):
    img_new = np.zeros((img.shape[0],img.shape[1],size,size))
    h = np.amin([size,img.shape[2]])
    w = np.amin([size,img.shape[3]])
    img_new[:,:,size//2-h//2:size//2+h//2,size//2-w//2:size//2+w//2]=img[:,:,img.shape[2]//2-h//2:img.shape[2]//2+h//2,img.shape[3]//2-w//2:img.shape[3]//2+w//2]
    return img_new


def upscale_2d_array_with_checkered_pattern(a, N=1):
    # a : Input array
    # N : number of zeros to be inserted between consecutive rows and cols
    out = np.zeros( (N+1)*np.array(a.shape)+N,dtype=a.dtype)
    out[N::N+1,N::N+1] = a
    return out