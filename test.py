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

import argparse
import data_loader.data_loaders as module_data
import data_loader.augmentation as module_aug
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import trainer as module_train
from parse_config import ConfigParser


def main(config):
    logger = config.get_logger('test')

    # Setup data_loader instances, ignoring train data:
    config["data_loader"]["args"]["train_data_folders"] = []
    train_augmentor = None
    valid_augmentor = config.init_obj('augmentations', module_aug, is_test_data=True)
    data_loader = config.init_obj('data_loader', module_data, mode='test',
                                  train_augmentor=train_augmentor, valid_augmentor=valid_augmentor)
    valid_data_loader = data_loader.split_validation()

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    if isinstance(config['loss'], str):
        criterion = getattr(module_loss, config['loss'])
        is_criterion_sequence = False
    elif isinstance(config['loss'], dict):
        criterion = config.init_obj('loss', module_loss)
        is_criterion_sequence = False
    else:
        criterion = []
        is_criterion_sequence = True
        for crit in config['loss']:
            if isinstance(crit, str):
                crit_fn = getattr(module_loss, crit)
            else:
                crit_fn = config.init_obj(crit, module_loss)
            criterion.append(crit_fn)

    metrics = [getattr(module_metric, met) for met in config['metrics']]

    trainer = config.init_obj('trainer', module_train,
                              model=model,
                              criterion=criterion,
                              metric_ftns=metrics,
                              optimizer=None,
                              config=config,
                              data_loader=data_loader,
                              valid_data_loader=valid_data_loader,
                              lr_scheduler=None)
    trainer._valid_epoch(epoch=1)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
