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
> Handling multiple losses
> Adding visualization features
'''

import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker, Tensor
import os
from pathlib import Path
from utils.visualize import save_result_as_image, save_result_as_mat_file
from utils.util_mri import convert_to_1hot


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None, **kwargs):
        super().__init__(model, criterion, metric_ftns, optimizer, config, **kwargs)
        self.config = config
        self.data_loader = data_loader
        if self.optimizer is not None:
            if len_epoch is None:
                # epoch-based training
                self.len_epoch = len(self.data_loader)
            else:
                # iteration-based training
                self.data_loader = inf_loop(data_loader)
                self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.is_criterion_sequence = hasattr(self.criterion, "__getitem__") or hasattr(self.criterion, "__iter__")

        self.loss_names = ['loss']
        if self.is_criterion_sequence:
            self.loss_names += [c.__name__ for c in self.criterion]
        self.train_metrics = MetricTracker(*self.loss_names, *[m.__name__ for m in self.metric_ftns],
                                           writer=self.writer)
        self.valid_metrics = MetricTracker(*self.loss_names, *[m.__name__ for m in self.metric_ftns],
                                           writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        # tqdm_it = tqdm(enumerate(self.data_loader), total=len(self.data_loader))
        for batch_idx, batch in enumerate(self.data_loader):

            x, x_pred, gt = batch
            x, x_pred, gt = x.type(Tensor), x_pred.type(Tensor), gt.type(torch.cuda.LongTensor)

            data = [x, x_pred, x]
            truth = {'gt': gt, 'x': x, 'x_pred': x_pred}

            self.optimizer.zero_grad()
            output = self.model(*data)
            if self.is_criterion_sequence:
                losses = []
                for crit in self.criterion:
                    loss = crit(output, truth)
                    #     loss.backward()
                    #     self.optimizer.step()
                    losses.append(loss)
                    self.train_metrics.update(crit.__name__, loss.item())

                loss = sum(losses)
                loss.backward()
                self.optimizer.step()
                self.train_metrics.update('loss', loss.item())
                losses = [loss] + losses
            else:
                loss = self.criterion(output, truth)
                loss.backward()
                self.optimizer.step()
                self.train_metrics.update('loss', loss.item())
                losses = [loss]

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)

            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, truth))

            if batch_idx % self.log_step == 0:
                losses_strings = ["{name}: {value:.6f}".format(name=self.loss_names[i], value=losses[i].item())
                                  for i in range(len(losses))]
                losses_strings = " | ".join(losses_strings)
                # self.logger.debug('Train Epoch: {} {} {}'.format(
                #     epoch,
                #     self._progress(batch_idx),
                #     losses_strings))
                # tqdm.write(losses_strings + "\t(epoch {})".format(epoch))
                self.logger.debug(losses_strings + "\t(epoch {})".format(epoch))
                self.writer.add_image('input', make_grid(x.cpu(), nrow=8, normalize=True))
                # try:
                gt_images = self.data_loader.train_dataset.convert_gts_to_images(gt)
                self.writer.add_image('target', make_grid(gt_images, nrow=8, normalize=True))
                # except Exception as e: # maybe not images after all...
                #     pass

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()

        with torch.no_grad():
            x = None
            image_num = 0
            # for batch_idx, batch in tqdm(enumerate(self.valid_data_loader), total=len(self.valid_data_loader)):
            for batch_idx, batch in enumerate(self.valid_data_loader):
                if len(batch) == 4:                 # val mode with data paths
                    x_new, gt, x_paths, gt_paths = batch
                else:
                    if self.optimizer is not None:  # val mode without data paths
                        x_new, gt = batch
                        x_paths, gt_paths = None, None
                    elif len(batch) == 2:           # test mode with data paths
                        x_new, x_paths = batch
                        gt, gt_paths = None, None
                    else:                           # test mode without data paths
                        x_new = batch
                        gt, x_paths, gt_paths = None, None, None

                x_new = x_new.type(Tensor)
                if gt is not None:
                    gt = gt.type(torch.cuda.LongTensor)

                x_pred = x if x is not None else x_new
                x = x_new

                data = [x, x_pred, x]
                truth = {'gt': gt, 'x': x, 'x_pred': x_pred}

                output = self.model(*data)
                if self.is_criterion_sequence:
                    losses = []
                    for crit in self.criterion:
                        loss = crit(output, truth)
                        losses.append(loss)
                        self.valid_metrics.update(crit.__name__, loss.item())
                    loss = sum(losses)
                else:
                    loss = self.criterion(output, truth)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, truth))
                self.writer.add_image('input', make_grid(x.cpu(), nrow=8, normalize=True))

                for i in range(0, x.shape[0]):
                    num_classes = output['outs_softmax'].shape[-3]
                    gt_onehot = convert_to_1hot(gt[:, None].cpu().numpy(), num_classes) if gt is not None else None

                    if x_paths is not None:
                        sequence_folder = os.path.dirname(x_paths[i])
                        basename_no_ext = os.path.splitext(os.path.basename(x_paths[i]))[0]
                        mat_results_folder = os.path.join(
                            os.path.dirname(sequence_folder), "DL_OP", "MASK_FLOW", self.config.exper_id)
                        # Path(mat_results_folder).mkdir(parents=True, exist_ok=True)
                        mat_result_filename = os.path.join(mat_results_folder,
                                                           "{}_{}.mat".format(basename_no_ext, "{}"))
                        result_filename = os.path.join(self.config.log_dir,
                                                       "{}_frame{}_{}.png".format(basename_no_ext, image_num, "{}"))
                    else:
                        mat_result_filename = os.path.join(self.config.log_dir,
                                                           "pred_results_{:04}_{}.mat".format(image_num, "{}"))
                        result_filename = os.path.join(self.config.log_dir,
                                                       "pred_results_{:04}_{}.png".format(image_num, "{}"))
                    # save_result_as_mat_file(output, i, mat_result_filename)

                    save_result_as_image(output, i, gt_onehot,
                                         result_filename,
                                         semantic_map_to_color_fn=self.data_loader.val_dataset.id2color)
                    image_num += 1

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
