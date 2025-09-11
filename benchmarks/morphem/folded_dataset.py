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
