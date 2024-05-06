from __future__ import division

import logging
import os.path
import pickle
import random
import pydicom
from typing import Any, List

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from glob import glob
import pandas as pd

import utils

logger = logging.getLogger("global_logger")


def build_rsna_dataloader(batch_size, backbone):
    logger.info("building rsna dataset")
    transform = utils.transform_color if backbone == 152 else utils.transform_resnet18

    test_normal_path = glob('./test/normal/*')
    test_anomaly_path = glob('./test/anomaly/*')
    test_path = test_normal_path + test_anomaly_path
    test_label = [0] * len(test_normal_path) + [1] * len(test_anomaly_path)

    train_normal_path = glob('./train/normal/*')
    train_label = [0] * len(train_normal_path)

    normal_path = glob('/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/test/NORMAL/*')
    anomaly_path = glob('/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/test/PNEUMONIA/*')

    shifted_test_path = normal_path + anomaly_path
    shifted_test_label = [0] * len(normal_path) + [1] * len(anomaly_path)

    dataset = RSNA(image_path=train_normal_path, labels=train_label, transform=transform)
    dataset1 = RSNA(image_path=train_normal_path, labels=train_label, transform=utils.Transform())

    dataset_main = RSNA(image_path=test_path, labels=test_label, transform=transform)
    dataset_shifted = ChestX_Ray(image_path=shifted_test_path, labels=shifted_test_label, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2,
                                               drop_last=False)
    train_loader1 = torch.utils.data.DataLoader(dataset1, batch_size=batch_size, shuffle=True, num_workers=2,
                                                drop_last=False)
    test_loader_main = torch.utils.data.DataLoader(dataset_main, batch_size=batch_size, shuffle=False, num_workers=2,
                                                   drop_last=False)
    test_loader_shift = torch.utils.data.DataLoader(dataset_shifted, batch_size=batch_size, shuffle=False, num_workers=2,
                                                    drop_last=False)

    return train_loader, test_loader_main, train_loader1, test_loader_shift


class RSNA(Dataset):
    def __init__(self, image_path, labels, transform=None, count=-1):
        self.transform = transform
        self.image_files = image_path
        self.labels = labels
        if count != -1:
            if count < len(self.image_files):
                self.image_files = self.image_files[:count]
                self.labels = self.labels[:count]
            else:
                t = len(self.image_files)
                for i in range(count - t):
                    self.image_files.append(random.choice(self.image_files[:t]))
                    self.labels.append(random.choice(self.labels[:t]))

    def __getitem__(self, index):
        # Read the DICOM file
        dicom = pydicom.dcmread(self.image_files[index])
        image = dicom.pixel_array

        # Convert to a PIL Image
        image = Image.fromarray(image).convert('RGB')

        # Apply the transform if it's provided
        if self.transform is not None:
            image = self.transform(image)


        return image, self.labels[index]

    def __len__(self):
        return len(self.image_files)


class ChestX_Ray(Dataset):
    def __init__(self, image_path, labels, transform=None, count=-1):
        self.transform = transform
        self.image_files = image_path
        self.labels = labels
        if count != -1:
            if count < len(self.image_files):
                self.image_files = self.image_files[:count]
                self.labels = self.labels[:count]
            else:
                t = len(self.image_files)
                for i in range(count - t):
                    self.image_files.append(random.choice(self.image_files[:t]))
                    self.labels.append(random.choice(self.labels[:t]))

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return image, self.labels[index]

    def __len__(self):
        return len(self.image_files)
