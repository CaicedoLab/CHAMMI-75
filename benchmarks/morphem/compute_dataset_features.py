import torch
from torch.utils.data import DataLoader
from torch import nn
import sys
sys.path.append('/scr/vidit/Foundation_Models/FoundationModels')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import skimage
import importlib
from tqdm import tqdm

from vision_transformer import vit_small
from torchvision.models import resnet18, ResNet18_Weights
from timm import create_model
import torch.nn.functional as F
import sys

import argparse

from FoundationModels.dataset.dataset import IterableImageArchive
from FoundationModels.dataset import dataset_config
from FoundationModels.dataset.dataset_functions import randomize, split_for_workers, get_proc_split
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision import datasets, transforms


transform = v2.Compose([
    v2.CenterCrop(224),
    v2.ToTensor(),
    v2.Normalize([0.485], [0.229])
])

config = dataset_config.DatasetConfig(
            "/scr/data/foundation_data/CHAMMIv2s.zip", # args.data_path, /scr/data/CHAMMIv2m.zip
            split_fns=[randomize, split_for_workers],
            transform=transform,
            seed=42
            )

dataset = IterableImageArchive(config)
data_loader = DataLoader(dataset=dataset, batch_size=512, num_workers=8, worker_init_fn=dataset.worker_init_fn)

def create_pad(images, patch_width, patch_height): # new method for vit model
    N, C, H, W = images.shape

    new_width = ((W + patch_width - 1) // patch_width) * patch_width
    pad_width = new_width - W

    # Calculate padding amounts for left and right
    pad_left = pad_right = pad_width // 2
    
    if pad_width % 2 != 0:
        pad_right += 1


    new_height = ((H + patch_height - 1) // patch_height) * patch_height
    pad_height = new_height - H
    
    # Calculate padding amounts for top and bottom
    pad_top = pad_bottom = pad_height // 2
    
    if pad_height % 2 != 0:
        pad_bottom += 1
        

    padded_images = F.pad(images, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    
    return padded_images

class ViTClass():
    def __init__(self):
        self.device = f"cuda:2" if torch.cuda.is_available() else 'cpu'

        # Create model with in_chans=1 to match training setup
        self.model = vit_small()
        remove_prefixes = ["module.backbone.", "module.", "module.head."]

        # Load model weights
        student_model = torch.load("/scr/vidit/Foundation_Models/model_weights/LR_0.0005_GuidedCrop/checkpoint.pth")['student']
        # Remove unwanted prefixes
        cleaned_state_dict = {}
        for k, v in student_model.items():
            new_key = k
            for prefix in remove_prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]  # Remove prefix
            if not new_key.startswith("head.mlp") and not new_key.startswith("head.last_layer"):
                cleaned_state_dict[new_key] = v  # Keep only valid keys
        self.model.load_state_dict(cleaned_state_dict, strict=False)
        self.model.eval()
        self.model.to(self.device)

    def get_model(self):
        return self.model

device = torch.device(f"cuda:2" if torch.cuda.is_available() else "cpu")
vit_instance = ViTClass() 
vit_model = vit_instance.get_model()

feature_file = 'pretrained_vit_features.npy'
all_features = []
all_filepaths = []  # New list to store image file names


# Loop through the DataLoader and process each batch.
for images, filepath in tqdm(data_loader, total=len(data_loader)):
    # Get the patch embedding module and extract the convolutional layer.
    images = images.to(torch.float32)
    patch_embed = vit_model.patch_embed
    conv_layer = patch_embed.proj
    patch_size = conv_layer.kernel_size  # This is typically a tuple (patch_height, patch_width)
    patch_height, patch_width = patch_size

    # Pad images so their dimensions are divisible by the patch dimensions.
    images = create_pad(images, patch_width, patch_height)
    
    # For each batch, process each channel separately.
    batch_features = []

    # Select the c-th channel, add a channel dimension, and move to the appropriate device.
    single_channel = images.to(device)
    # Forward pass through the model without tracking gradients.
    with torch.no_grad():
        output = vit_model.forward_features(single_channel)
    # Extract the normalized class token feature from the model output.
    feat = output["x_norm_clstoken"].cpu().numpy()  # shape: [batch_size, feature_dim]
    batch_features.append(feat)
    
    # Concatenate features along the feature dimension (axis=1).
    batch_features = np.concatenate(batch_features, axis=1)

    all_features.append(batch_features)
    all_filepaths.extend(filepath)  # Store the file names


# Concatenate all batch features into a single NumPy array.
all_features = np.concatenate(all_features, axis=0)
all_features = np.squeeze(all_features)  # Remove any singleton dimensions if needed.

# Save the features to a NumPy file.
np.save(feature_file, {"features": all_features, "filepaths": all_filepaths})
print(f"Features saved to {feature_file}")