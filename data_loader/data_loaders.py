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
> Wrote data loader specific to SCCD dataset, partially inspired from example MNIST data loader.
'''

import torch
from torch.utils.data import DataLoader

from data_loader.dataset_sccd import TrainDataset, TestDataset, SCCP_CLOUD_SPLITS, SCCP_IMG_SIZE, id2color, \
    multiple_torch_id2color, SCCP_MODALITY_SUBFOLDERS, color2id, SCCP_DATA_ROOT


class SCCPDataLoader(DataLoader):
    """
    SCCP data loading
    """

    def __init__(self,
                 train_data_folders=SCCP_CLOUD_SPLITS['train'],
                 valid_data_folders=SCCP_CLOUD_SPLITS['val'],
                 sequence_length=1,
                 batch_size=1,
                 shuffle=False,
                 num_workers=2,
                 train_augmentor=None,
                 valid_augmentor=None,
                 train_modality_subfolders=SCCP_MODALITY_SUBFOLDERS,
                 dataset_root=SCCP_DATA_ROOT,
                 mode='train'):

        assert(mode in ['train', 'test'])

        if mode == 'train':
            train_sets = [TrainDataset(data_subfolder,
                                       dataset_root=dataset_root,
                                       sequence_length=sequence_length,
                                       transform=train_augmentor,
                                       modality_subfolders=train_modality_subfolders)
                          for data_subfolder in train_data_folders]
            self.train_data_folders = train_data_folders
            self.train_augmentor = train_augmentor
            self.train_dataset = torch.utils.data.ConcatDataset(train_sets)
            self.train_dataset.convert_gts_to_images = multiple_torch_id2color
            self.train_dataset.id2color = id2color
            self.train_dataset.color2id = color2id
            self.train_modality_subfolders = train_modality_subfolders

            test_mode = 'val'
        else:
            self.train_dataset = None
            test_mode = 'test'

        val_sets = [TestDataset(data_subfolder, dataset_root=dataset_root, mode=test_mode,
                                sequence_length=sequence_length, transform=valid_augmentor, return_filenames=True)
                    for data_subfolder in valid_data_folders]
        self.valid_data_folders = valid_data_folders
        self.valid_augmentor = valid_augmentor
        self.val_dataset = torch.utils.data.ConcatDataset(val_sets)
        self.val_dataset.convert_gts_to_images = multiple_torch_id2color
        self.val_dataset.id2color = id2color
        self.val_dataset.color2id = color2id
        self.val_dataset.mode = test_mode

        self.shuffle = shuffle
        self.sequence_length = sequence_length

        self.batch_idx = 0

        self.train_init_kwargs = {
            'dataset': self.train_dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers
        }
        self.valid_init_kwargs = {
            'dataset': self.val_dataset,
            'batch_size': 1,
            'shuffle': False,
            'num_workers': num_workers
        }

        super().__init__(**self.train_init_kwargs)

        self.original_image_size = SCCP_IMG_SIZE

    def split_validation(self):
        return DataLoader(**self.valid_init_kwargs)

    def split_training_without_augmentation(self, return_filenames=True):
        data_sets = [TestDataset(data_subfolder,
                                 sequence_length=self.sequence_length,
                                 transform=self.valid_augmentor,
                                 return_filenames=return_filenames)
                     for data_subfolder in self.train_data_folders]
        data_sets = torch.utils.data.ConcatDataset(data_sets)
        init_kwargs = {
            'dataset': data_sets,
            'batch_size': self.train_init_kwargs['batch_size'],
            'shuffle': False,
            'num_workers': self.train_init_kwargs['num_workers']
        }
        return DataLoader(**init_kwargs)
