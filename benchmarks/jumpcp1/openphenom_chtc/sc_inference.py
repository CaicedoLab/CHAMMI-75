import sys
from typing import Sequence, List, Tuple, Optional, Tuple, Union

import torch
import numpy as np
import numbers
import math
import warnings
import argparse
import os

from torchvision import transforms
from torchvision.transforms.functional import _interpolation_modes_from_int, InterpolationMode, get_dimensions, center_crop
from tqdm import tqdm
import sc_dataset_openphenom
from OpenPhenom.huggingface_mae import MAEModel


class self_normalize(object):
    def __call__(self, x):
        m = x.mean((-2, -1), keepdim=True)
        s = x.std((-2, -1), unbiased=False, keepdim=True)
        x -= m
        x /= s + 1e-7
        return x


class SingleCellInferenceTransform(torch.nn.Module):
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.CenterCrop((128,128)),
            transforms.Resize((256,256), interpolation=transforms.InterpolationMode.BICUBIC),
            self_normalize(),
        ])

    def __call__(self, img):
        return self.transform(img)


def main(dataset_location):
    batch_size = 128
    n_channels = 5
    channel_wise_embeddings = True
    huggingface_modelpath = "recursionpharma/OpenPhenom"
    model = MAEModel.from_pretrained(huggingface_modelpath)
    metadata_file = '../sc-metadata.csv'
    device = torch.device('cuda')

    model = MAEModel.from_pretrained(huggingface_modelpath)
    model.return_channelwise_embeddings = channel_wise_embeddings
    model.eval()
    model.to(device)

    transform = SingleCellInferenceTransform()
    lincs_dataset = sc_dataset_openphenom.SingleCellDataset(transform=transform,
        root = dataset_location, metadata_path = metadata_file)

    print(len(lincs_dataset))
    data_loader = torch.utils.data.DataLoader(
        lincs_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        persistent_workers=0
    )

    if channel_wise_embeddings:
        feature_matrix = np.zeros((len(lincs_dataset), 384*n_channels), dtype=np.float32) # TODO: whole feature matrix should not be stored in RAM
    else:
        feature_matrix = np.zeros((len(lincs_dataset), 384), dtype=np.float32) # TODO: whole feature matrix should not be stored in RAM

    bi = 0
    iterations = len(lincs_dataset) // batch_size

    for samples in tqdm(data_loader):
        samples = samples.to(device)
        with torch.no_grad():
            latent = model.predict(samples)
            print(latent.shape)
            feature_matrix[batch_size*bi : batch_size*bi + latent.shape[0], :] = latent.detach().cpu().numpy()

        bi += 1

    plate_name = os.path.basename(os.path.normpath(dataset_location))
    os.system(f'cp sc-metadata.csv /output/single-cells/{plate_name}_metadata.csv')
    np.savez(f'/output/single-cells/{plate_name}.npz', features=feature_matrix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str)
    args = parser.parse_args()
    sys.exit(main(args.dataset_path))

