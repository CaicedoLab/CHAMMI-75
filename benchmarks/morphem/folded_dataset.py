import os
import torch
import skimage.io

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
t = torchvision.transforms.ToTensor()
from collections.abc import Sequence
from torch import Tensor
from typing import Tuple, List, Optional
import math

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
import os
import torch
import skimage.io

import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
t = torchvision.transforms.ToTensor()
from collections.abc import Sequence
from torch import Tensor
from typing import Tuple, List, Optional
import math

import random
from torchvision.transforms import v2

########################################################
## Scale normalization functions for CHAMMI images
########################################################

rrc160 = v2.Compose([
    v2.CenterCrop((92,92)),
    v2.RandomResizedCrop((256,256), scale=(0.8,0.9), ratio=(0.95,1.05), antialias=True)
])
rrc238 = v2.Compose([
    v2.RandomResizedCrop((256,256), scale=(0.95,1.0), ratio=(0.95,1.05), antialias=True)
])
rrc512 = v2.Compose([
    v2.RandomResizedCrop((256,256), scale=(0.95,1.0), ratio=(0.95,1.05), antialias=True)
])

def normalize_scale_for_training(im):
    if im.shape[-2] == 160:
        t = rrc160(im)
    elif im.shape[-2] == 238:
        t = rrc238(im)
    elif im.shape[-2] == 512:
        t = rrc512(im)
    k = random.choice([0, 1, 2, 3])
    t = torch.rot90(t, k=k, dims=(-2, -1))
    return t

def normalize_scale_for_test(im):
    sizes = {160:160, 238:238, 512:512}
    t = transforms.functional.center_crop(im, sizes[im.shape[-2]])
    t = transforms.functional.resize(t, (224,224))
    return t


########################################################
## Re-arrange channels from tape format to stack tensor
########################################################

def fold_channels(image, channel_width, mode="ignore"):
    # Expected input image shape: (h, w * c)
    # Output image shape: (h, w, c)
    output = np.reshape(image, (image.shape[0], channel_width, -1), order="F")

    if mode == "ignore":
        # Keep all channels
        pass
    elif mode == "drop":
        # Drop mask channel (last)
        output = output[:, :, 0:-1]
    elif mode == "apply":
        # Use last channel as a binary mask
        mask = output["image"][:, :, -1:]
        output = output[:, :, 0:-1] * mask

    return t(output)


########################################################
## Dataset Class
########################################################

class SingleCellDataset(Dataset):
    """Single cell dataset."""
    def __init__(self, csv_file, root_dir, target_labels=None, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with metadata.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.metadata = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.target_labels = target_labels

    def __len__(self):
        return len(self.metadata)

    def prepare(self, idx, image, label, norm_func, mixup=False):
        #c = self.metadata.loc[idx, "channel"]
        #image = image[c,...]
        #image = image[np.newaxis,...]
        image = norm_func(image)
 
        if mixup:
            other_idx = np.random.randint(0, self.metadata.shape[0])
            other_img, other_label = self.load_image(other_idx)
            other_img, other_label = self.prepare(other_idx, other_img, other_label, False)
            alpha = 0.5*np.random.random()
            image = (1 - alpha)*image + alpha*other_img
            label = (1 - alpha)*label + alpha*other_label
        return image, label


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.metadata.loc[idx, "file_path"])
        channel_width = self.metadata.loc[idx, 'channel_width']
        image = skimage.io.imread(img_name)
        image = fold_channels(image, channel_width)

        if self.target_labels is not None:
            labels = self.metadata.loc[idx, self.target_labels]
        else:
            labels = None
        
        image, labels = self.prepare(idx, image, labels, normalize_scale_for_test, False)

        if self.transform:
            image = self.transform(image)

        return image, labels


########################################################
## Transformations
########################################################

class Single_cell_centered(torch.nn.Module):
    def __init__(self, size=224):
        super().__init__()
        self.cell_image_size = size

    def forward(self, img):
        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)
        cell_image_size = self.cell_image_size
        if not isinstance(img, torch.Tensor):
            img = torchvision.transforms.ToTensor()(img)
        new_img = img
        img_shape = new_img.permute(1, 2, 0).shape

        if (img_shape[0] >= img_shape[1]) and (img_shape[0] > cell_image_size):
            new_img = transforms.functional.resized_crop(
                new_img,
                0,
                0,
                img_shape[0],
                img_shape[1],
                (int(img_shape[1] * cell_image_size / img_shape[0]), cell_image_size),
                transforms.InterpolationMode("bilinear"),
            )
        elif (img_shape[1] >= img_shape[0]) and (img_shape[1] > cell_image_size):
            new_img = transforms.functional.resized_crop(
                new_img,
                0,
                0,
                img_shape[0],
                img_shape[1],
                (cell_image_size, (int(img_shape[0] * cell_image_size / img_shape[1]))),
                transforms.InterpolationMode("bilinear"),
            )
        img_shape = new_img.permute(1, 2, 0).shape
        pad_border = 0
        upper_pad = int(
            min(max((cell_image_size - (img_shape[0] + pad_border)) / 2, 0), pad_border)
        )
        lower_pad = int(
            min(max((cell_image_size - (img_shape[0] + pad_border)) / 2, 0), pad_border)
        )
        left_pad = int(
            min(max((cell_image_size - (img_shape[1] + pad_border)) / 2, 0), pad_border)
        )
        right_pad = int(
            min(max((cell_image_size - (img_shape[1] + pad_border)) / 2, 0), pad_border)
        )
        new_img = np.pad(
            new_img, ((upper_pad, lower_pad), (left_pad, right_pad), (0, 0)), "constant"
        ).transpose(1, 2, 0)
        img_shape = new_img.shape
        upper_pad = int((cell_image_size - img_shape[0]) / 2)
        lower_pad = max(
            cell_image_size - img_shape[0] - upper_pad,
            int((cell_image_size - img_shape[0]) / 2),
        )
        left_pad = int((cell_image_size - img_shape[1]) / 2)
        right_pad = max(
            cell_image_size - img_shape[1] - left_pad,
            int((cell_image_size - img_shape[1]) / 2),
        )
        padded_img = np.pad(
            new_img, ((upper_pad, lower_pad), (left_pad, right_pad), (0, 0)), "constant"
        )
        img = transforms.ToTensor()(padded_img)
        return img


class RandomResizedCrop(torch.nn.Module):
    def __init__(
        self,
        size,
        scale=(0.8, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation=transforms.InterpolationMode("bilinear"),
    ):
        super().__init__()
        self.size = size
        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(
        img: Tensor, scale: List[float], ratio: List[float]
    ) -> Tuple[int, int, int, int]:
        #         width, height = torchvision.transforms.functional._get_image_size(img)
        width, height = img.shape[1:]
        area = height * width

        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            log_ratio = torch.log(torch.tensor(ratio))
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.
        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """

        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return transforms.functional.resized_crop(
            img, i, j, h, w, self.size, self.interpolation
        )

