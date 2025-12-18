#### 081623 BK edit from https://github.com/uhlerlab/cross-modal-autoencoders/blob/master/dataloader.py

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os


class ToTensorNormalize(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image_tensor = sample["image"]
        # raw input is uint8 - maximum of 255. Convert it back to 0 and 1 scale, float32
        image_tensor = (image_tensor / 255).astype("float32")
        image_tensor = np.clip(image_tensor, 0, 1)
        N_CH = image_tensor.shape[0]
        return torch.from_numpy(image_tensor).view(N_CH, 64, 64)


class PerImageNormalize(object):
    def __init__(self, eps=1e-7):
        self.eps = eps

    def __call__(self, sample):
        image_tensor = sample["image"]
        image_tensor = torch.from_numpy(image_tensor).float()

        if len(image_tensor.shape) == 4:  # Batch of images [B, C, H, W]
            # Normalize each image in the batch independently
            B, C, H, W = image_tensor.shape
            mean = torch.mean(image_tensor, dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
            std = torch.std(image_tensor, dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        else:  # Single image [C, H, W]
            mean = torch.mean(image_tensor, dim=(1, 2), keepdim=True)
            std = torch.std(image_tensor, dim=(1, 2), keepdim=True)

        normalized_tensor = (image_tensor - mean) / (std + self.eps)
        return normalized_tensor


class CellDataset(Dataset):
    def __init__(
        self, datadir, mode="train", transform=PerImageNormalize(), mask_flag=True
    ):
        self.datadir = datadir  # '../input_data'
        self.mode = mode  # default is train
        self.mask_flag = mask_flag
        self.images = self.load_image_dataframe()  # see below # list of
        self.transform = transform  # take class ToTensorNormalize

    # data can be found ../input_data/cropped_images_combined.pkl
    # Utility function to load images from a HDF5 file that was pickled.
    # image file should contain train/test split label. refer to 'Prepare_Torch_Dataset_070623.ipynb' notebook
    # for prepping single cell crops, metadata organization, dataframe structure
    # dataframe containing mask, image, metadata
    def load_image_dataframe(self):
        # load image_dataframe
        im_df = pd.read_pickle(os.path.join(self.datadir, "CELLPHIE.pkl"))
        if self.mask_flag:  # if mask_flag is True; then mask image
            im_df["image"] = im_df["image"] * im_df["soma_mask"]
        else:
            pass

        # train/test split. Already performed in
        im_df_train = im_df[im_df.train == 1]
        im_df_test = im_df[im_df.train == 0]

        if self.mode == "train":
            return im_df_train
        elif self.mode == "test":
            return im_df_test
        else:
            raise KeyError("Mode %s is invalid, must be 'train' or 'test'" % self.mode)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # retrieve the data sample at given numerical indexing.
        sample = self.images.iloc[idx]  # dataframe of images

        if self.transform:
            # transform the tensor and the particular z-slice
            image_tensor = self.transform(sample)
            return {
                "image_tensor": image_tensor,
                "UMI": sample["UMI"],
                "Gene": sample["Gene"],
                "Time": sample["Time"],
                "Media": sample["Media"],
                "Sample": sample["Sample"],
            }
        return sample


if __name__ == "__main__":
    # test_CellDataset() # just to see if the code runs
    print("Testing complete")
