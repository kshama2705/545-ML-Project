
from typing import Optional

import json
import os
import shutil
import time

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import optim
from torchvision import transforms


class VOC2007Detection(torch.utils.data.Dataset):
    """
    A tiny version of PASCAL VOC 2007 Detection dataset that includes images and
    annotations with small images and no difficult boxes.
    """

    def __init__(
        self,
        dataset_dir: str,
        split: str = "train",
        image_size: int = 224,
    ):
        """
        Args:
            download: Whether to download PASCAL VOC 2007 to `dataset_dir`.
            image_size: Size of imges in the batch. The shorter edge of images
                will be resized to this size, followed by a center crop. For
                val, center crop will not be taken to capture all detections.
        """
        super().__init__()
        self.image_size = image_size

        
        # fmt: off
        voc_classes = [
            "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
            "car", "cat", "chair", "cow", "diningtable", "dog",
            "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"
        ]
        # fmt: on

        # Make a (class to ID) and inverse (ID to class) mapping.
        self.class_to_idx = {
            _class: _idx for _idx, _class in enumerate(voc_classes)
        }
        self.idx_to_class = {
            _idx: _class for _idx, _class in enumerate(voc_classes)
        }

        # Load instances from JSON file:
        self.instances = json.load(
            open(os.path.join(dataset_dir, f"voc07_{split}.json"))
        )
        self.dataset_dir = dataset_dir

        # Define a transformation function for image: Resize the shorter image
        # edge then take a center crop (optional) and normalize.
        _transforms = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
        self.image_transform = transforms.Compose(_transforms)