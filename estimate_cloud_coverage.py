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
> Refactored using trainer method to do evaluation
'''

import os
import argparse
import data_loader.data_loaders as module_data
import data_loader.augmentation as module_aug
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import trainer as module_train
from parse_config import ConfigParser
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional

from utils.io import PROJECT_ROOT
from utils.util import prepare_device


def main(config):
    args = config.args
    logger = config.get_logger('test')

    # build model architecture
    model = config.init_obj('arch', module_arch)
    device, device_ids = prepare_device(config['n_gpu'], logger=logger)
    model = model.to(device)
    # if len(device_ids) > 1:
    #     model = torch.nn.DataParallel(model, device_ids=device_ids)
    logger.info(model)

    # load pretrained weights:
    resume_path = str(config.resume)
    logger.info("Loading checkpoint: {} ...".format(resume_path))
    checkpoint = torch.load(resume_path)
    if checkpoint['config']['arch'] != config.config['arch']:
        logger.warning(
            "Warning: Architecture configuration given in config file is different from that of "
            "checkpoint. This may yield an exception while state_dict is being loaded.")
    model.load_state_dict(checkpoint['state_dict'])
    logger.info("Checkpoint loaded.")

    # load image:
    image_path = str(args.image)
    logger.info("Loading image: {} ...".format(image_path))
    image = Image.open(image_path).convert('RGB')
    image = torchvision.transforms.functional.to_tensor(image).unsqueeze(dim=0).to(device)
    original_height, original_width = image.shape[-2:]
    inference_height, inference_width = config.config["arch"]["args"]['height'], config.config["arch"]["args"]['width']
    image = torch.nn.functional.interpolate(image, size=(inference_height, inference_width), mode="bilinear")
    logger.info("Image loaded.")

    # launch inference
    logger.info("Starting inference.")
    data = [image, image, image]    # we are only predicting cloud coverage, not cloud motion.
                                    # So instead of passing a sequence of image, we pass the single image multiple times.
    with torch.no_grad():
        output = model(*data)
    cloud_class_channel = 1     # channel 0 is for "non-cloud" class.
    cloud_mask = output['outs_softmax']
    cloud_mask = torch.nn.functional.interpolate(cloud_mask, size=(original_height, original_width), mode="nearest")
    cloud_mask = cloud_mask.squeeze()[cloud_class_channel].cpu().numpy()
    logger.info("Inference done.")

    # save results:
    if args.output is None:
        args.output = os.path.splitext(args.image)[0] + "_result.png"
    logger.info("Saving results to: {} ...".format(args.output))
    cloud_image = Image.fromarray(np.uint8(cloud_mask * 255), 'L')
    cloud_image.save(args.output)
    logger.info("Over.")


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Deep-Learning-Based Sky Coverage Estimation')
    args.add_argument('-c', '--config',
                      default=os.path.join(PROJECT_ROOT, 'config', 'config_test.json'),
                      type=str,
                      help='config file path of deep-learning model (default: None).')
    args.add_argument('-r', '--resume',
                      default=os.path.join(PROJECT_ROOT, 'resources', 'saved_models',
                                           'segmotionnet_sccp-data_trained-with-backward-compat.pth'),
                      type=str,
                      help='path to latest trained version of deep-learning model (default: provided pretrained model)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-i', '--image',
                      default=os.path.join(PROJECT_ROOT, 'resources', 'test_images',
                                           'Isabella_TestSamples', 'TestSeq1.png'),
                      type=str,
                      help='path to target sky image (default: image from provided sequence)')
    args.add_argument('-o', '--output',
                      default=None,
                      type=str,
                      help='output path where to save resulting cloud coverage image (default: same folder as input image)')

    config = ConfigParser.from_args(args)
    main(config)
