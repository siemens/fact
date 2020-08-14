'''
©2020 Siemens Corporation, Corporate Technology
All Rights Reserved

NOTICE: All information contained herein is, and remains, the property of Siemens Corporation, Corporate Technology.
It is proprietary to Siemens and may be covered by patent and copyright laws.

--------------------------
File based on:
--------------------------
PyTorch-Template

MIT License

Copyright (c) 2018 Victor Huang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

--------------------------
Edits made to original file:
--------------------------
> Wrote project-specific losses, reusing the PyTorch-Template data structure.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Tensor

def nll_loss(output, target):
    return F.nll_loss(output, target)


def huber_loss(x):
    x = x.reshape(-1, *x.shape[-3:])
    bsize, csize, height, width = x.size()
    d_x = torch.index_select(x, 3, torch.arange(1, width).cuda()) - torch.index_select(x, 3,
                                                                                       torch.arange(width - 1).cuda())
    d_y = torch.index_select(x, 2, torch.arange(1, height).cuda()) - torch.index_select(x, 2,
                                                                                        torch.arange(height - 1).cuda())
    err = torch.sum(torch.mul(d_x, d_x)) / height + torch.sum(torch.mul(d_y, d_y)) / width
    err /= bsize
    tv_err = torch.sqrt(0.01 + err)
    return tv_err


class MRIFlowLoss(torch.nn.Module):

    __name__ = "flow_warp_loss"

    def __init__(self, factor_flow=1., factor_warp=1.):
        super().__init__()

        zero = Tensor([0.])
        use_warp_loss = factor_warp > 0

        self.flow_criterion = nn.MSELoss()
        self.warp_criterion = nn.MSELoss() if use_warp_loss else lambda x_pred, x_true: zero

        self.factor_flow = factor_flow
        self.factor_warp = factor_warp

    def forward(self, net, truth):

        flow_loss = self.flow_criterion(net['fr_st'], truth['x_pred']) + 0.01 * huber_loss(net['out'])
        warp_loss = self.warp_criterion(net['warped_outs'], net['outs_softmax_prev'])

        loss = self.factor_flow * flow_loss + self.factor_warp * warp_loss
        return loss


class MRIWarpLoss(torch.nn.Module):
    __name__ = "warp_loss"

    def __init__(self, factor=1.):
        super().__init__()

        self.warp_criterion = nn.MSELoss()
        self.factor = factor

    def forward(self, net, truth=None):
        warp_loss = self.warp_criterion(net['warped_outs'], net['outs_softmax_prev'])
        return self.factor * warp_loss


class MRISegmentationLoss(torch.nn.Module):

    __name__ = "seg_loss"

    def __init__(self, factor=0.01):
        super().__init__()

        self.seg_criterion = nn.CrossEntropyLoss()
        self.factor = factor

    def forward(self, net, truth):
        num_classes, h, w = net['outs_softmax'].shape[-3:]

        pred = net['outs_softmax'].reshape(-1, num_classes, h, w)
        truth = truth['gt'].reshape(-1, h, w)
        seg_loss = self.seg_criterion(pred, truth)
        return self.factor * seg_loss


class BackwardFlowConsistencyLoss(torch.nn.Module):
    __name__ = "back_loss"

    def __init__(self, factor=1.):
        super().__init__()

        self.consistency_criterion = nn.MSELoss()
        self.factor = factor

    def forward(self, net, truth=None):

        # WRONG - back_loss = self.consistency_criterion(net['out_backw'], -1. * net['out'])

        back_loss_forward_then_backward_identity = self.consistency_criterion(net['warp_identity'], truth['x'])
        back_loss_backward_then_forward_identity = self.consistency_criterion(net['warp_identity_backw'], truth['x_pred'])
        back_loss = back_loss_forward_then_backward_identity + back_loss_backward_then_forward_identity

        return self.factor * back_loss
