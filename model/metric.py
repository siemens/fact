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
> Wrote project-specific metrics, reusing the PyTorch-Template data structure.
'''

import torch
import cv2
import numpy as np


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


flow_criterion_l1 = torch.nn.MSELoss()


def flow_error_L1(output, target):
    x_pred_c = output['conv0s']
    flow_loss = flow_criterion_l1(output['fr_st'], x_pred_c)
    output['flow_loss'] = flow_loss.item()
    return output['flow_loss']


def opencv_flow_error_L1(output, target):
    x_c, x_pred_c = output['conv0'], output['conv0s']
    c, h, w = x_c.shape[-3:]
    x_c = x_c.reshape(-1, c, h, w)
    batch_x_np, batch_x_pred_np = x_c.permute(0, 2, 3, 1).cpu().numpy(), x_pred_c.permute(0, 2, 3, 1).cpu().numpy()
    batch_x_np, batch_x_pred_np = (batch_x_np * 255).astype(np.uint8), (batch_x_pred_np * 255).astype(np.uint8)
    flows_opencv = []
    warped_opencv = []
    for x_np, x_pred_np in zip(batch_x_np, batch_x_pred_np):
        x_np_gray, x_pred_np_gray = cv2.cvtColor(x_np, cv2.COLOR_RGB2GRAY), cv2.cvtColor(x_pred_np, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            x_np_gray, x_pred_np_gray, None,
            pyr_scale=0.5, levels=1, winsize=5, iterations=9, poly_n=5, poly_sigma=1.1, flags=0)

        # Warp image accordingly:
        flow4warp = -flow
        flow4warp[:, :, 0] += np.arange(w)
        flow4warp[:, :, 1] += np.arange(h)[:, np.newaxis]
        warped_x_np = cv2.remap(x_np, flow4warp, None, cv2.INTER_LINEAR)

        flows_opencv.append(torch.tensor(flow, dtype=torch.float32).cuda())
        warped_opencv.append(torch.tensor(warped_x_np.astype(np.float32) / 255., dtype=torch.float32).cuda())

    output['out_opencv'] = torch.stack(flows_opencv, dim=0).permute(0, 3, 1, 2)
    # output['fr_st_opencv'] = flow_warp(x_c, output['out_opencv'])
    output['fr_st_opencv_remap'] = torch.stack(warped_opencv, dim=0).permute(0, 3, 1, 2)


    # flow_loss_opencv = flow_criterion_l1(output['fr_st_opencv'], x_pred_c)
    flow_loss_opencv = flow_criterion_l1(output['fr_st_opencv_remap'], x_pred_c)
    output['flow_loss_opencv'] = flow_loss_opencv.item()
    return output['flow_loss_opencv']