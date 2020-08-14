'''
©2020 Siemens Corporation, Corporate Technology
All Rights Reserved

NOTICE: All information contained herein is, and remains, the property of Siemens Corporation, Corporate Technology.
It is proprietary to Siemens and may be covered by patent and copyright laws.

--------------------------
File name: dataset_sccd.py
Date: 2020-04-07 16:23
Python Version: 3.6
'''


# ==============================================================================
# Imported Modules
# ==============================================================================

import torch.utils.data as data
import numpy as np
import datetime
import os
import logging
import glob
from collections import namedtuple
from PIL import Image
import torch
import torchvision.transforms.functional as TF
import itertools

#################################################################################
# Constant Definitions
#################################################################################

SemanticClass = namedtuple('SemanticClass', [
    'label',                # Name of the semantic class
    'id',                   # ID assigned to the class and used for training to identify the class in the prediction volume.
                            # (IDs should therefore be continuous, starting from 0)
    'single_channel_color', # Single channel color assigned to this class.
])

SCCD_CLASSES = [
    SemanticClass('unlabeled', 0,   0),
    SemanticClass('cloud',     1, 225)
]

SCCP_DATA_ROOT = '/data/SCCP_data'

SCCP_PLACES = ["Isabela", "Princeton"]

SCCP_CLOUD_SPLITS = {'train': ["Isabela_2018-06-09",
                               "Isabela_2018-06-26",
                               "Isabela_2018-07-09"],
                     'val': ["Isabela_2018-05-09",
                             "Isabela_2018-09-30"]}

SCCP_MODALITY_SUBFOLDERS = {"image": "SkyImage", "label": "SkyImageRec"}

SCCP_DATA_PATH = os.path.join(
    # '{root}', '{place}', 'SCCP_VisualAnalytics', 'Figures', '{split}', '{modality}'
    '{root}', '{place}', '{split}'  # Data path for test images
)

SCCP_NUM_CLASSES = 2  # 1
SCCP_IMG_SIZE = (500, 500)

SCCD_PALETTE = list(itertools.chain(*[[sem_class.single_channel_color] * 3 for sem_class in SCCD_CLASSES]))
SCCD_PALETTE += [0] * ((256 - len(SCCD_CLASSES)) * 3)


#################################################################################
# Function Definitions
#################################################################################


def id2color(mask):
    """
    Convert a semantic segmentation mask into a colorized segmentation image.
    """
    image = Image.fromarray(mask.astype(np.uint8)).convert('P')
    image.putpalette(SCCD_PALETTE)
    return image


def color2id(image):
    """
    Convert a palette segmentation image into a semantic segmentation mask
    """
    mask = np.asarray(image)
    mask_copy = mask.copy()
    for semantic_class in SCCD_CLASSES:
        mask_copy[mask == semantic_class.single_channel_color] = semantic_class.id
    return mask_copy


def multiple_torch_id2color(masks):
    """
    Convert PyTorch semantic segmentation masks into colorized segmentation images.
    """
    masks = masks.reshape(-1, *masks.shape[-2:]).cpu().numpy()
    gt_images = []
    for mask in masks:
        gt_image = id2color(mask)
        gt_image = TF.to_tensor(gt_image)
        gt_images.append(gt_image)

    gt_images = torch.stack(gt_images, dim=0)
    return gt_images


def make_dataset(split_subfolder=SCCP_CLOUD_SPLITS["train"][0], mode='train',
                 dataset_root=SCCP_DATA_ROOT,
                 places=SCCP_PLACES,
                 modality_subfolders=SCCP_MODALITY_SUBFOLDERS):
    """
    List all dataset files.
    """
    assert (mode in ['train', 'val', 'test'])

    place, day = split_subfolder.split("_")
    assert (place in places)

    input_folder = SCCP_DATA_PATH.format(
        root=dataset_root, place=place, split=split_subfolder, modality=modality_subfolders["image"])
    input_files_template = os.path.join(input_folder, "*.png")
    images = sorted(glob.glob(os.path.join(input_files_template)))
    assert (len(images) > 0)

    if mode != 'test':
        label_folder = SCCP_DATA_PATH.format(
            root=dataset_root, place=place, split=split_subfolder, modality=modality_subfolders["label"])
        label_files_template = os.path.join(label_folder, "*Rec.png")
        labels = sorted(glob.glob(os.path.join(label_files_template)))
        assert (len(images) == len(labels))

        items = [(img, label) for img, label in zip(images, labels)]
    else:
        items = [(img, None) for img in images]

    logging.info('SCCP CLOUD-{} ({}): {} images'.format(split_subfolder, mode, len(items)))

    return items


#################################################################################
# Class Definitions
#################################################################################

class TrainDataset(data.Dataset):
    def __init__(self, split_subfolder=SCCP_CLOUD_SPLITS["train"][0], transform=None, mode='train',
                 max_skipped_frames=0, sequence_length=1, modality_subfolders=SCCP_MODALITY_SUBFOLDERS,
                 dataset_root=SCCP_DATA_ROOT):
        super(TrainDataset, self).__init__()
        self.split_subfolder = split_subfolder
        self.max_skipped_frames = max_skipped_frames
        self.sequence_length = sequence_length
        self.colorize_mask = id2color
        self.color2id = color2id
        self.modality_subfolders = modality_subfolders
        self.dataset_root = dataset_root

        self.items = make_dataset(split_subfolder, mode,
                                  dataset_root=dataset_root, modality_subfolders=modality_subfolders)

        # data augmentation
        self.transform = transform
        self.convert_gts_to_images = multiple_torch_id2color

        self.num_classes = SCCP_NUM_CLASSES

        self.debug = False

    def __getitem__(self, index):
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

        index = (index % len(self.items))

        images, labels = [], []
        for i in range(self.sequence_length + 1):
            img_path, mask_path = self.items[index]
            img, label = Image.open(img_path).convert('RGB'), Image.open(mask_path).convert('P')
            label = self.color2id(label)
            label = Image.fromarray(label, mode='P')

            images.append(img)
            labels.append(label)

            num_skipped_frames = np.random.randint(0, self.max_skipped_frames + 1)
            index = index + (1 + num_skipped_frames)

        if self.transform:
            images, labels = self.transform(images, labels)

        images = torch.stack(images, dim=0)
        labels = torch.stack(labels, dim=0).squeeze(1)
        img = images[1:]
        img_prev = images[:-1]
        label = labels[1:]

        if self.sequence_length == 1:
            img, img_prev, label = img.squeeze(0), img_prev.squeeze(0), label.squeeze(0)

        return img, img_prev, label

    def __len__(self):
        # return self.sequence_length + 1 # debug
        # return 0 # debug
        return len(self.items) - ((self.max_skipped_frames + 1) * self.sequence_length)


class TestDataset(data.Dataset):
    def __init__(self, split_subfolder=SCCP_CLOUD_SPLITS["val"][0], transform=None, mode='val', sequence_length=1,
                 return_filenames=True, dataset_root=SCCP_DATA_ROOT):
        super(TestDataset, self).__init__()
        self.split_subfolder = split_subfolder
        self.sequence_length = sequence_length
        self.colorize_mask = id2color
        self.color2id = color2id
        self.return_filenames = return_filenames
        self.dataset_root = dataset_root
        self.mode = mode

        self.items = make_dataset(split_subfolder, mode, dataset_root=dataset_root)

        # data augmentation
        self.transform = transform
        self.convert_gts_to_images = multiple_torch_id2color

        self.num_classes = SCCP_NUM_CLASSES

    def __getitem__(self, index):

        index = (index % len(self.items))

        should_load_labels = self.mode != 'test'

        images, labels = [], []
        image_paths, label_paths = [], []
        for i in range(self.sequence_length):
            img_path, mask_path = self.items[index]
            img = Image.open(img_path).convert('RGB')
            if should_load_labels:
                label = Image.open(mask_path).convert('P')
                label = self.color2id(label)
                label = Image.fromarray(label, mode='P')
            else:
                label = None

            images.append(img)
            labels.append(label)
            image_paths.append(img_path)
            label_paths.append(mask_path)

            # num_skipped_frames = np.random.randint(0, self.max_skipped_frames + 1)
            # index = index + (1 + num_skipped_frames)
            index += 1

        if self.transform:
            images, labels = self.transform(images, labels)

        images = torch.stack(images, dim=0)
        if should_load_labels:
            labels = torch.stack(labels, dim=0).squeeze(1)
        # else:
        #     labels = np.empty(images.shape[0])
        #     label_paths = labels

        if self.sequence_length == 1:
            images, labels = images.squeeze(0), labels[0]
            image_paths, label_paths = image_paths[0], label_paths[0]

        if self.return_filenames:
            if should_load_labels:
                return images, labels, image_paths, label_paths
            else:
                return images, image_paths
        else:
            return images, labels if should_load_labels else images

    def __len__(self):
        # return self.sequence_length + 1 # debug
        return len(self.items)
