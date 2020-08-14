'''
©2020 Siemens Corporation, Corporate Technology
All Rights Reserved

NOTICE: All information contained herein is, and remains, the property of Siemens Corporation, Corporate Technology.
It is proprietary to Siemens and may be covered by patent and copyright laws.

--------------------------
File name: visualize.py
Date: 2020-04-28 20:51
Python Version: 3.6
'''

# ==============================================================================
# Imported Modules
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import matplotlib
matplotlib.use('Agg')

import io
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import cv2

from data_loader.dataset_sccd import id2color, color2id
from utils import util_mri

# ==============================================================================
# Constant Definitions
# ==============================================================================

# ==============================================================================
# Function Definitions
# ==============================================================================

def make_border(img, size=2, color=(.0,.0,.0)):

    img_with_border = cv2.copyMakeBorder(
        img,
        top=size,
        bottom=size,
        left=size,
        right=size,
        borderType=cv2.BORDER_CONSTANT,
        value=color
    )
    return img_with_border

def write_legend_in_image(img, text,
                          top_left_offset=(5, 15),
                          font=cv2.FONT_HERSHEY_SIMPLEX,
                          font_scale=.5,
                          font_color=(0, 255, 0)):
    cv2.putText(img, text, top_left_offset, font, font_scale, font_color)
    return img

def convert_pytorch_image_to_opencv(tensor):
    image = np.transpose(tensor.detach().cpu().numpy(), (1, 2, 0))
    image = (image * 255).astype(np.uint8)
    return image


def convert_numpy_mask_to_opencv_image(mask, semantic_map_to_color_fn=id2color):
    mask_image = semantic_map_to_color_fn(np.argmax(mask, axis=0)).convert('RGB')
    # mask_image = np.double(mask_image) / 255.
    mask_image = np.array(mask_image)
    return mask_image


def convert_numpy_mask_to_label_image(mask, semantic_map_to_label_fn=color2id):
    mask_image = semantic_map_to_label_fn(np.argmax(mask, axis=0))
    # mask_image = np.double(mask_image) / 255.
    mask_image = np.asarray(mask_image)
    return mask_image


def convert_pytorch_flow_to_opencv_image(tensor, input_image, mask=None, flow_scale=96):
    try:
        optical_flow = tensor.detach().cpu().numpy()
    except:
        optical_flow = tensor

    if mask is not None:
        optical_flow = optical_flow * mask
    # optical_flow = optical_flow[:, 60:140, 40:120] * 96
    optical_flow = optical_flow * flow_scale

    # X, Y = np.meshgrid(np.arange(0, 80, 2), np.arange(0, 80, 2))
    X, Y = np.meshgrid(np.arange(0, optical_flow.shape[1], 2),
                       np.arange(0, optical_flow.shape[2], 2))

    fig = plt.figure(figsize=(6,6))
    fig.tight_layout(pad=0)
    ax = fig.gca()
    ax.imshow(input_image)
    # mean_scale = np.sqrt(meanu ** 2 + meanv ** 2) * 200
    ax.quiver(X, Y, -optical_flow[0, ::2, ::2], optical_flow[1, ::2, ::2], scale_units='xy', scale=1, color='y')
    ax.margins(0)
    ax.axis('off')
    plt.subplots_adjust(0,0,1,1,0,0)

    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw')
    io_buf.seek(0)
    image_quiver = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                              newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    image_quiver = image_quiver[...,:3]
    io_buf.close()
    plt.close(fig)

    # fig.canvas.draw()
    # image_quiver = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    # print(fig.canvas.get_width_height()[::-1] + (3,))
    # image_quiver = image_quiver.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    image_quiver = cv2.resize(image_quiver, input_image.shape[:2])

    # image_quiver = image_quiver.astype(np.float32) / 255
    return image_quiver


def save_result_as_mat_file(net, index_in_batch, filename_template="./results/pred_result_0001_{}.mat"):

    c, h, w = net['conv0'].shape[-3:]
    num_classes = net['outs_softmax'].shape[-3]
    semantic_mask, optical_flow = (
        net['outs_softmax'].reshape(-1, num_classes, h, w)[index_in_batch].detach().cpu().numpy(),
        (net['out'] * -1.).reshape(-1, 2, h, w)[index_in_batch].detach().cpu().numpy()
    )

    scipy.io.savemat(filename_template.format("optical_flow"), {'optical_flow': optical_flow})
    scipy.io.savemat(filename_template.format("semantic_mask"), {'semantic_mask': semantic_mask})


def save_result_as_image(net, index_in_batch, gt=None,
                         filename_template="./results/pred_result_0001_{}.png",
                         semantic_map_to_color_fn=id2color):

    c, h, w = net['conv0'].shape[-3:]
    num_classes = net['outs_softmax'].shape[-3]

    input_image, input_image_prev, semantic_mask, optical_flow, warped_image, warped_semantic_map, semantic_map_prev = (
        convert_pytorch_image_to_opencv(net['conv0'].reshape(-1, c, h, w)[index_in_batch]),
        convert_pytorch_image_to_opencv(net['conv0s'].reshape(-1, c, h, w)[index_in_batch]),
        net['outs_softmax'].reshape(-1, num_classes, h, w)[index_in_batch].detach().cpu().numpy(),
        net['out'].reshape(-1, 2, h, w)[index_in_batch],
        convert_pytorch_image_to_opencv(net['fr_st'].reshape(-1, c, h, w)[index_in_batch]),
        convert_numpy_mask_to_opencv_image(net['outs_softmax_prev'].reshape(-1, num_classes, h, w)[index_in_batch].detach().cpu().numpy(), semantic_map_to_color_fn=id2color),
        convert_numpy_mask_to_opencv_image(net['warped_outs'].reshape(-1, num_classes, h, w)[index_in_batch].detach().cpu().numpy(), semantic_map_to_color_fn=id2color),

    )

    # Compute relative-error image between warped and expected images:
    diff_image = np.abs(input_image_prev - warped_image)
    diff_image = (diff_image.astype(np.float32) / input_image_prev * 255).astype(np.uint8)
    diff_image[np.isnan(diff_image)] = 255

    # Upscale optical flow results for better interpretation:
    upscale_factor = 3
    optical_flow = (optical_flow.detach().cpu().numpy()) * 25
    optical_flow_upscaled = np.stack(
        [util_mri.upscale_2d_array_with_checkered_pattern(optical_flow[0], N=upscale_factor),
         util_mri.upscale_2d_array_with_checkered_pattern(optical_flow[1], N=upscale_factor)], axis=0) / 255.
    input_image_upscaled = cv2.resize(input_image, optical_flow_upscaled.shape[1:], interpolation=cv2.INTER_AREA)
    input_image_prev_upscaled = cv2.resize(input_image_prev, optical_flow_upscaled.shape[1:], interpolation=cv2.INTER_AREA)
    warped_image_upscaled = cv2.resize(warped_image, optical_flow_upscaled.shape[1:], interpolation=cv2.INTER_AREA)
    diff_image_upscaled = cv2.resize(diff_image, optical_flow_upscaled.shape[1:], interpolation=cv2.INTER_AREA)
    semantic_mask_upscaled = np.stack(
        [cv2.resize(semantic_mask[0], optical_flow_upscaled.shape[1:], interpolation=cv2.INTER_NEAREST),
         cv2.resize(semantic_mask[1], optical_flow_upscaled.shape[1:], interpolation=cv2.INTER_NEAREST)], axis=0)

    image_quiver_upscaled = convert_pytorch_flow_to_opencv_image(optical_flow_upscaled,
                                                                 input_image_upscaled,
                                                                 #semantic_mask_upscaled)
                                                                 None, flow_scale=90)
    semantic_map = convert_numpy_mask_to_opencv_image(semantic_mask, semantic_map_to_color_fn=id2color)

    # Save optical flow results:
    input_image_upscaled_border = write_legend_in_image(make_border(input_image_upscaled, color=[255, 0, 0]), "img t")
    input_image_upscaled_prev_border = write_legend_in_image(make_border(input_image_prev_upscaled, color=[255, 0, 0]), "img t-1")
    image_quiver_border = write_legend_in_image(make_border(image_quiver_upscaled, color=[0, 0, 255]), "pred. flow")
    warped_image_border = write_legend_in_image(make_border(warped_image_upscaled, color=[0, 0, 255]), "pred. img t-1")
    diff_image_border = write_legend_in_image(make_border(diff_image_upscaled, color=[255, 0, 255]), "rel error")
    if 'flow_loss' in net:
        warped_image_border = write_legend_in_image(
            warped_image_border,
            "L1 = {}".format(np.round(net['flow_loss'], 4)),
            top_left_offset=(5, warped_image_border.shape[0] - 15))

    combined_image = np.hstack(
        [input_image_upscaled_border, image_quiver_border, input_image_upscaled_prev_border, warped_image_border, diff_image_border])
    cv2.imwrite(filename_template.format("flow"), combined_image)

    cv2.imwrite(filename_template.format("flow_0_upscaled_input"), input_image_upscaled)
    cv2.imwrite(filename_template.format("flow_1_upscaled_target"), input_image_prev_upscaled)
    cv2.imwrite(filename_template.format("flow_2_upscaled_pred"), warped_image_upscaled)
    image_quiver = convert_pytorch_flow_to_opencv_image(optical_flow,
                                                        input_image,
                                                        semantic_mask[1],
                                                        # None,
                                                        flow_scale=5)
    cv2.imwrite(filename_template.format("flow_3_quiver_masked"), image_quiver)
    image_quiver = convert_pytorch_flow_to_opencv_image(optical_flow,
                                                        input_image,
                                                        # semantic_mask[1],
                                                        None,
                                                        flow_scale=5)
    cv2.imwrite(filename_template.format("flow_3_quiver"), image_quiver)

    # Save semantic segmentation results for image t:
    # image_merged_with_results = input_image * 0.7 + semantic_map * 0.3
    input_image_border = write_legend_in_image(make_border(input_image, color=[255, 0, 0]), "img t")
    semantic_map_border = write_legend_in_image(make_border(semantic_map, color=[0, 0, 255]), "pred.")
    combined_image = [input_image_border, semantic_map_border]
    if gt is not None:
        gt = convert_numpy_mask_to_opencv_image(gt[index_in_batch], semantic_map_to_color_fn=id2color)
        gt_border = write_legend_in_image(make_border(gt, color=[255, 0, 0]), "target")
        combined_image.append(gt_border)
    combined_image = np.hstack(combined_image)
    cv2.imwrite(filename_template.format("segm"), combined_image)

    cv2.imwrite(filename_template.format("segm_0_input"), input_image)
    cv2.imwrite(filename_template.format("segm_2_pred"), semantic_map)
    if gt is not None:
        cv2.imwrite(filename_template.format("segm_1_target"), gt)

    # Save semantic segmentation results for image t-1:
    input_image_prev_border = write_legend_in_image(make_border(input_image_prev, color=[255, 0, 0]), "img t-1")
    semantic_map_prev_border = write_legend_in_image(make_border(semantic_map_prev, color=[0, 0, 255]), "pred.")
    semantic_map_warp_border = write_legend_in_image(make_border(warped_semantic_map, color=[0, 0, 255]), "pred. t warped")
    combined_image = np.hstack([input_image_prev_border, semantic_map_prev_border, semantic_map_warp_border])
    cv2.imwrite(filename_template.format("warp"), combined_image)

    if 'out_backw' in net:
        # Save optical flow results:
        optical_flow_backw = net['out_backw'].reshape(-1, 2, h, w)[index_in_batch]
        optical_flow_backw = (optical_flow_backw.detach().cpu().numpy()) * 25
        warped_image_backw = convert_pytorch_image_to_opencv(net['fr_st_backw'].reshape(-1, c, h, w)[index_in_batch])
        warped_image_upscaled_backw = cv2.resize(warped_image_backw, optical_flow_upscaled.shape[1:], interpolation=cv2.INTER_AREA)

        cv2.imwrite(filename_template.format("flow_1__upscaled_pred_backw"), warped_image_upscaled_backw)
        image_quiver_backw = convert_pytorch_flow_to_opencv_image(optical_flow_backw,
                                                            input_image,
                                                            semantic_mask[1],
                                                            # None,
                                                            flow_scale=5)
        cv2.imwrite(filename_template.format("flow_3_quiver_masked_backw"), image_quiver_backw)
        image_quiver_backw = convert_pytorch_flow_to_opencv_image(optical_flow_backw,
                                                            input_image,
                                                            # semantic_mask[1],
                                                            None,
                                                            flow_scale=5)
        cv2.imwrite(filename_template.format("flow_3_quiver_backw"), image_quiver_backw)

    # if 'out_opencv' in net: # not working, weird...
    try:
        # OpenCV flow results for comparison:
        optical_flow_opencv, warped_image_opencv = (
            net['out_opencv'].reshape(-1, 2, h, w)[index_in_batch],
            # convert_pytorch_image_to_opencv(net['fr_st_opencv'].reshape(-1, c, h, w)[index_in_batch])
            convert_pytorch_image_to_opencv(net['fr_st_opencv_remap'].reshape(-1, c, h, w)[index_in_batch])
        )
        optical_flow_opencv = optical_flow_opencv.detach().cpu().numpy()
        optical_flow_opencv_upscaled = np.stack(
            [util_mri.upscale_2d_array_with_checkered_pattern(optical_flow_opencv[0], N=upscale_factor),
             util_mri.upscale_2d_array_with_checkered_pattern(optical_flow_opencv[1], N=upscale_factor)], axis=0)
        warped_image_opencv_upscaled = cv2.resize(warped_image_opencv, optical_flow_upscaled.shape[1:], interpolation=cv2.INTER_AREA)

        diff_image_opencv = np.abs(input_image_prev - warped_image_opencv) // warped_image * 255
        diff_image_opencv[np.isnan(diff_image_opencv)] = 255
        diff_image_opencv_upscaled = cv2.resize(diff_image_opencv, optical_flow_upscaled.shape[1:], interpolation=cv2.INTER_AREA)

        image_quiver_opencv_upscaled = convert_pytorch_flow_to_opencv_image(
            optical_flow_opencv_upscaled,
            input_image_upscaled,
            # semantic_mask_upscaled,
            None,
            flow_scale=90)

        image_quiver_opencv_border = write_legend_in_image(make_border(image_quiver_opencv_upscaled, color=[0, 0, 255]), "pred. flow")
        warped_image_opencv_border = write_legend_in_image(make_border(warped_image_opencv_upscaled, color=[0, 0, 255]), "pred. img t-1")
        diff_image_opencv_border = write_legend_in_image(make_border(diff_image_opencv_upscaled, color=[255, 0, 255]), "rel error")
        if 'flow_loss_opencv' in net:
            warped_image_opencv_border = write_legend_in_image(
                warped_image_opencv_border,
                "L1 = {}".format(np.round(net['flow_loss_opencv'], 4)),
                top_left_offset=(5, warped_image_opencv_border.shape[0] - 15))

        combined_image = np.hstack(
            [input_image_upscaled_border, image_quiver_opencv_border, input_image_upscaled_prev_border, warped_image_opencv_border, diff_image_opencv_border])
        cv2.imwrite(filename_template.format("flow_opencv"), combined_image)

        cv2.imwrite(filename_template.format("flow_opencv_0_input"), input_image_upscaled_border)
        cv2.imwrite(filename_template.format("flow_opencv_1_target"), input_image_upscaled_prev_border)
        cv2.imwrite(filename_template.format("flow_opencv_2_pred"), warped_image_opencv_border)
        image_quiver = convert_pytorch_flow_to_opencv_image(optical_flow_opencv,
                                                            input_image,
                                                            # semantic_mask,
                                                            None,
                                                            flow_scale=1)
        cv2.imwrite(filename_template.format("flow_opencv_3_quiver"), image_quiver)
    except Exception as e:
        print(e)
        pass
    return

def mask_color_img(img, mask):
    alpha = 0.5
    rows, cols = img.shape
    # Construct a colour image to superimpose
    color_mask = np.zeros((rows, cols, 3))
    color_mask[mask == 1] = [1, 0, 0]  # Red block
    color_mask[mask == 2] = [0, 1, 0]  # Green block
    color_mask[mask == 3] = [0, 0, 1]  # Blue block

    # Construct RGB version of grey-level image
    img_color = np.dstack((img, img, img))
    img_masked = img_color * 0.8 + np.double(color_mask) * 0.3
    return img_masked


def create_prediction_video(save_dir, img, pred, loc, seq_num):
    # create a video with joint prediction of ROI
    mask = np.argmax(pred, axis=1)
    img_mask_bank = []

    for t in range(seq_num):
        img_mask = mask_color_img(img[0, t], mask[t])
        img_mask_bank.append(img_mask)

    mask[mask == 1] = 0
    mask[mask == 3] = 0
    mask[mask == 2] = 1

    mask = np.tile(mask[:, np.newaxis], (1, 2, 1, 1))
    loc = loc * mask
    img_mask_bank = np.array(img_mask_bank)
    flow = loc[:, :, 60:140, 40:120] * 96
    X, Y = np.meshgrid(np.arange(0, 80, 2), np.arange(0, 80, 2))
    for t in range(seq_num):
        # meanu = np.mean(flow[t, 0])
        # meanv = np.mean(flow[t, 1])
        plt.imshow(img_mask_bank[t, 60:140, 40:120])

        # mean_scale = np.sqrt(meanu ** 2 + meanv ** 2) * 200
        plt.quiver(X, Y, -flow[t, 0, ::2, ::2], flow[t, 1, ::2, ::2], scale_units='xy', scale=1, color='y')

        plt.axis('off')
        plt.savefig(os.path.join(save_dir, 'test_%d.png'%t))
        plt.close()
    image_dir = os.path.join(save_dir, 'test_%d.png')
    video_dir = os.path.join(save_dir, 'video.avi')
    #os.system('ffmpeg - f image2 - i {0} - vcodec mpeg4 - b 800k {1}'.format(image_dir, video_dir))
    print("Done: images saved in {}".format(image_dir))


# ==============================================================================
# Main Classes
# ==============================================================================

# ==============================================================================
# Main Function
# ==============================================================================