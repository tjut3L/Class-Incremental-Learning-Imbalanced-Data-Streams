################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 20-05-2020                                                             #
# Author: Vincenzo Lomonaco                                                    #
# E-mail: contact@continualai.org                                              #
# Website: continualai.org                                                     #
################################################################################

""" Tiny-Imagenet Pytorch Dataset """

import csv
import os
from pathlib import Path
from typing import Union

from torchvision.datasets.folder import default_loader
from torchvision.transforms import ToTensor

from avalanche.benchmarks.datasets import (
    SimpleDownloadableDataset,
    default_dataset_location,
)
import torch
from torch.utils.data import Dataset
from PIL import Image
import os


class ImagenetSubset():
    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = root_dir
        self.train = train
        self.targets = []
        self.data = []
        self.transform = transform
        self.get_data()

    def get_data(self):
        if self.train:
            path = os.path.join(self.root_dir, 'train.txt')
            with open(path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line_spl = line.split(' ')
                    p1 = os.path.join(self.root_dir, line_spl[0])
                    self.data.append(p1)
                    self.targets.append(int(line_spl[1]))
        else:
            path = os.path.join(self.root_dir, 'test.txt')
            with open(path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line_spl = line.split(' ')
                    if int(line_spl[1])<100:
                        p1 = os.path.join(self.root_dir, line_spl[0])
                        self.data.append(p1)
                        self.targets.append(int(line_spl[1]))

    def pil_loader(self, path: str):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def default_loader(self, path: str):
        return self.pil_loader(path)

    def __len__(self):
        """Returns the length of the set"""
        return len(self.data)

    def __getitem__(self, index):
        """Returns the index-th x, y pattern of the set"""

        path, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = default_loader(path)

        if self.transform is not None:
            img = self.transform(img)
        return img, target


__all__ = ["ImagenetSubset"]
