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
> Edited to fit training framework
> Added additional network outputs for sup. losses and visualization
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from base import BaseModel
from utils.util_mri import generate_grid

def relu():
    return nn.ReLU(inplace=True)


def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, nonlinearity=relu):
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                           padding=padding, bias=False)
    nn.init.xavier_uniform(conv_layer.weight, gain=np.sqrt(2.0))

    nll_layer = nonlinearity()
    bn_layer = nn.BatchNorm2d(out_channels)
    # nn.init.constant_(bn_layer.weight, 1)
    # nn.init.constant_(bn_layer.bias, 0)

    layers = [conv_layer, bn_layer, nll_layer]
    return nn.Sequential(*layers)


def conv_blocks_2(in_channels, out_channels, strides=1):
    conv1 = conv(in_channels, out_channels, stride=strides)
    conv2 = conv(out_channels, out_channels, stride=1)
    layers = [conv1, conv2]
    return nn.Sequential(*layers)


def conv_blocks_3(in_channels, out_channels, strides=1):
    conv1 = conv(in_channels, out_channels, stride=strides)
    conv2 = conv(out_channels, out_channels, stride=1)
    conv3 = conv(out_channels, out_channels, stride=1)
    layers = [conv1, conv2, conv3]
    return nn.Sequential(*layers)


class Registration_Net(nn.Module):
    """Deformable registration network with input from image space """

    def __init__(self, n_ch=1):
        super(Registration_Net, self).__init__()

        self.conv_blocks = [conv_blocks_2(n_ch, 64), conv_blocks_2(64, 128, 2), conv_blocks_3(128, 256, 2),
                            conv_blocks_3(256, 512, 2), conv_blocks_3(512, 512, 2)]
        self.conv = []
        for in_filters in [128, 256, 512, 1024, 1024]:
            self.conv += [conv(in_filters, 64)]

        self.conv_blocks = nn.Sequential(*self.conv_blocks)
        self.conv = nn.Sequential(*self.conv)

        self.conv6 = nn.Conv2d(64 * 5, 64, 1)
        self.conv7 = conv(64, 64, 1, 1, 0)
        self.conv8 = nn.Conv2d(64, 2, 1)

    def forward(self, x, x_pred, x_img):
        # x: source image; x_pred: target image; x_img: source image or segmentation map
        net = {}
        net['conv0'] = x
        net['conv0s'] = x_pred
        for i in range(5):
            net['conv%d' % (i + 1)] = self.conv_blocks[i](net['conv%d' % i])
            net['conv%ds' % (i + 1)] = self.conv_blocks[i](net['conv%ds' % i])
            net['concat%d' % (i + 1)] = torch.cat((net['conv%d' % (i + 1)], net['conv%ds' % (i + 1)]), 1)
            net['out%d' % (i + 1)] = self.conv[i](net['concat%d' % (i + 1)])
            if i > 0:
                net['out%d_up' % (i + 1)] = F.interpolate(net['out%d' % (i + 1)], scale_factor=2 ** i, mode='bilinear',
                                                          align_corners=True)

        net['concat'] = torch.cat((net['out1'], net['out2_up'], net['out3_up'], net['out4_up'], net['out5_up']), 1)
        net['comb_1'] = self.conv6(net['concat'])
        net['comb_2'] = self.conv7(net['comb_1'])

        net['out'] = torch.tanh(self.conv8(net['comb_2']))
        net['grid'] = generate_grid(x_img, net['out'])
        net['fr_st'] = F.grid_sample(x_img, net['grid'])

        return net


class SegMotionNet(BaseModel):
    """Joint motion estimation and segmentation """

    def __init__(self, in_ch=1, out_ch=4, height=None, width=None,
                 is_recurrent=False,
                 is_trained_with_backward_consistency=False,
                 is_class_1_only_dynamic=False):
        super(SegMotionNet, self).__init__()
        self.out_ch = out_ch
        self.height, self.width = height, width

        self.conv_blocks = [conv_blocks_2(in_ch, 64), conv_blocks_2(64, 128, 2), conv_blocks_3(128, 256, 2),
                            conv_blocks_3(256, 512, 2), conv_blocks_3(512, 512, 2)]
        self.conv = []
        for in_filters in [128, 256, 512, 1024, 1024]:
            self.conv += [conv(in_filters, 64)]

        self.conv_blocks = nn.Sequential(*self.conv_blocks)
        self.conv = nn.Sequential(*self.conv)

        self.conv6 = nn.Conv2d(64 * 5, 64, 1)
        self.conv7 = conv(64, 64, 1, 1, 0)
        self.conv8 = nn.Conv2d(64, 2, 1)

        self.convs = []
        for in_filters in [64, 128, 256, 512, 512]:
            self.convs += [conv(in_filters, 64)]
        self.convs = nn.Sequential(*self.convs)
        self.conv6s = nn.Conv2d(64 * 5, 64, 1)

        self.conv7s = conv(64, 64, 1, 1, 0)
        self.conv8s = nn.Conv2d(64, out_ch, 1)

        self.is_trained_with_backward_consistency = is_trained_with_backward_consistency
        self.is_class_1_only_dynamic = is_class_1_only_dynamic
        if is_recurrent:
            assert(self.height is not None and self.width is not None)
            rnn_input_size = out_ch * self.height * self.width
            self.rnn_layer = nn.RNN(input_size=rnn_input_size, hidden_size=rnn_input_size,
                                    num_layers=1, nonlinearity="tanh", bias=False, batch_first=True)

    def forward(self, x, x_pred, x_img):

        if len(x.shape) == 5:  # batch, seq, channel, H, W
            batch_size, sequence_length, num_channel, height, width = x.shape
        elif len(x.shape) == 4:  # batch, channel, H, W
            batch_size, num_channel, height, width = x.shape
            sequence_length = 1
        else:
            print("Data not batched?")
            num_channel, height, width = x.shape
            sequence_length, batch_size = 1, 1

        # If inputs are image sequences, we first process the images separately, flattening the seq and batch dims:
        x = x.reshape(batch_size * sequence_length, num_channel, height, width)
        x_pred = x_pred.reshape(batch_size * sequence_length, num_channel, height, width)
        x_img = x_img.reshape(batch_size * sequence_length, num_channel, height, width)

        # x: source image; x_pred: target image; x_img: image to be segmented
        # motion estimation branch
        net = {}
        net['conv0'] = x
        net['conv0s'] = x_pred
        for i in range(5):
            net['conv%d' % (i + 1)] = self.conv_blocks[i](net['conv%d' % i])
            net['conv%ds' % (i + 1)] = self.conv_blocks[i](net['conv%ds' % i])
            net['concat%d' % (i + 1)] = torch.cat((net['conv%d' % (i + 1)], net['conv%ds' % (i + 1)]), 1)
            net['out%d' % (i + 1)] = self.conv[i](net['concat%d' % (i + 1)])
            if i > 0:
                net['out%d_up' % (i + 1)] = F.interpolate(
                    net['out%d' % (i + 1)], scale_factor=2 ** i, mode='bilinear', align_corners=True)
            if self.is_trained_with_backward_consistency:
                net['concat%d_backw' % (i + 1)] = torch.cat((net['conv%ds' % (i + 1)], net['conv%d' % (i + 1)]), 1)
                net['out%d_backw' % (i + 1)] = self.conv[i](net['concat%d_backw' % (i + 1)])
                if i > 0:
                    net['out%d_up_backw' % (i + 1)] = F.interpolate(
                        net['out%d_backw' % (i + 1)], scale_factor=2 ** i, mode='bilinear', align_corners=True)


        net['concat'] = torch.cat((net['out1'], net['out2_up'], net['out3_up'], net['out4_up'], net['out5_up']), 1)
        net['comb_1'] = self.conv6(net['concat'])
        net['comb_2'] = self.conv7(net['comb_1'])
        net['comb_3'] = self.conv8(net['comb_2'])

        if sequence_length > 1:
            # Perform RNN logic:
            net['comb_3_rshp'] = net['comb_3'].reshape(batch_size, sequence_length, 2 * height * width)
            net['rec'], net['rec_hidden_state'] = self.rnn_layer(net['comb_3_rshp'])
            net['out'] = net['rec'].reshape(batch_size * sequence_length, 2, height, width)
        else:
            net['out'] = torch.tanh(net['comb_3'])

        if self.is_trained_with_backward_consistency:
            net['concat_backw'] = torch.cat((net['out1_backw'], *[net['out{}_up'.format(i+1)] for i in range(1, 5)]), 1)
            net['comb_1_backw'] = self.conv6(net['concat_backw'])
            net['comb_2_backw'] = self.conv7(net['comb_1_backw'])
            net['comb_3_backw'] = self.conv8(net['comb_2_backw'])

            if sequence_length > 1:
                # Perform RNN logic:
                net['comb_3_rshp_backw'] = net['comb_3_backw'].reshape(batch_size, sequence_length, 2 * height * width)
                net['rec_backw'], net['rec_hidden_state_backw'] = self.rnn_layer(net['comb_3_rshp_backw'])
                net['out_backw'] = net['rec_backw'].reshape(batch_size * sequence_length, 2, height, width)
            else:
                net['out_backw'] = torch.tanh(net['comb_3_backw'])


        # segmentation branch
        net['conv0ss'] = x_img
        for i in range(5):
            net['conv%dss' % (i + 1)] = self.conv_blocks[i](net['conv%dss' % i])
            net['out%ds' % (i + 1)] = self.convs[i](net['conv%dss' % (i + 1)])
            if i > 0:
                net['out%ds_up' % (i + 1)] = F.interpolate(net['out%ds' % (i + 1)], scale_factor=2 ** i,
                                                           mode='bilinear', align_corners=True)

        net['concats'] = torch.cat((net['out1s'],
                                    net['out2s_up'],
                                    net['out3s_up'],
                                    net['out4s_up'],
                                    net['out5s_up']), 1)
        net['comb_1s'] = self.conv6s(net['concats'])
        net['comb_2s'] = self.conv7s(net['comb_1s'])
        net['outs'] = self.conv8s(net['comb_2s'])
        net['outs_softmax'] = F.softmax(net['outs'], dim=1)
        # net['warped_outs'] = F.grid_sample(net['outs_softmax'], net['grid'], padding_mode='border')

        # segmentation branch for prev image:
        net['conv0ss_prev'] = x_pred
        for i in range(5):
            net['conv%dss_prev' % (i + 1)] = self.conv_blocks[i](net['conv%dss' % i])
            net['out%ds_prev' % (i + 1)] = self.convs[i](net['conv%ds' % (i + 1)])
            if i > 0:
                net['out%ds_up_prev' % (i + 1)] = F.interpolate(net['out%ds_prev' % (i + 1)], scale_factor=2 ** i,
                                                                mode='bilinear', align_corners=True)

        net['concats_prev'] = torch.cat((net['out1s_prev'],
                                         net['out2s_up_prev'],
                                         net['out3s_up_prev'],
                                         net['out4s_up_prev'],
                                         net['out5s_up_prev']), 1)
        net['comb_1s_prev'] = self.conv6s(net['concats_prev'])
        net['comb_2s_prev'] = self.conv7s(net['comb_1s_prev'])
        net['outs_prev'] = self.conv8s(net['comb_2s_prev'])
        net['outs_softmax_prev'] = F.softmax(net['outs_prev'], dim=1)

        if self.is_class_1_only_dynamic:
            net['out_full'] = net['out']
            net['out'] = net['out'] * net['outs_softmax'][:, 1]
            if self.is_trained_with_backward_consistency:
                net['out_backw_full'] = net['out_backw']
                net['out_backw'] = net['out_backw'] * net['outs_softmax_prev'][:, 1]

        net['grid'] = generate_grid(x_img, net['out'])
        net['fr_st'] = F.grid_sample(x_img, net['grid'])
        net['warped_outs'] = F.grid_sample(net['outs_softmax'], net['grid'], padding_mode='border')
        if self.is_trained_with_backward_consistency:
            net['grid_backw'] = generate_grid(x_img, net['out_backw'])
            net['fr_st_backw'] = F.grid_sample(x_pred, net['grid_backw'])
            net['warp_identity'] = F.grid_sample(net['fr_st'], net['grid_backw'])
            net['warp_identity_backw'] = F.grid_sample(net['fr_st_backw'], net['grid'])

        if sequence_length > 1:
            net['out'] = net['out'].reshape(batch_size, sequence_length, 2, height, width)
            net['fr_st'] = net['fr_st'].reshape(batch_size, sequence_length, num_channel, height, width)
            net['outs'] = net['outs'].reshape(batch_size, sequence_length, self.out_ch, height, width)
            net['outs_softmax'] = net['outs_softmax'].reshape(batch_size, sequence_length, self.out_ch, height, width)
            net['outs_prev'] = net['outs_prev'].reshape(batch_size, sequence_length, self.out_ch, height, width)
            net['outs_softmax_prev'] = net['outs_softmax_prev'].reshape(batch_size, sequence_length, self.out_ch, height, width)
            net['warped_outs'] = net['warped_outs'].reshape(batch_size, sequence_length, self.out_ch, height, width)
            if self.is_trained_with_backward_consistency:
                net['out_backw'] = net['out_backw'].reshape(batch_size, sequence_length, self.out_ch, height, width)
                net['fr_st_backw'] = net['fr_st_backw'].reshape(batch_size, sequence_length, num_channel, height, width)
                net['warp_identity'] = net['warp_identity'].reshape(batch_size, sequence_length, num_channel, height, width)
                net['warp_identity_backw'] = net['warp_identity_backw'].reshape(batch_size, sequence_length, num_channel, height, width)


        return net
