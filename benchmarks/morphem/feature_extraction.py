import torch
from torch.utils.data import DataLoader
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import skimage
from torchvision import transforms
import importlib
from tqdm import tqdm
from torchvision.transforms import v2
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn.functional as F
import sys
import argparse
import torch
import sys
import folded_dataset
import torch
sys.path.append("../")
from models import ViTClass, MAEModel, DinoV2, get_model
import timm  # If you used timm to define your ViT
import os


def configure_dataset(root_dir, dataset_name, transform=None):
    df_path = f'{root_dir}/{dataset_name}/enriched_meta.csv'
    df = pd.read_csv(df_path)
    dataset = folded_dataset.SingleCellDataset(csv_file=df_path, root_dir=root_dir, target_labels='train_test_split', transform=transform)
    return dataset

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


def get_save_features(feature_dir, root_dir, model_check, gpu, batch_size):
    dataset_names = ['Allen', 'CP', 'HPA']

    if not os.path.exists(args.feat_dir):
        os.makedirs(args.feat_dir, exist_ok=True)
        for dataset_name in dataset_names:
            os.makedirs(f'{feature_dir}/{dataset_name}', exist_ok=True)

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    model_instance = get_model(model_name=model_check, device=device, model_path=args.model_path, model_size=args.model_size)
    model, transform = model_instance.get_model()
    feature_file = model_instance.feature_file
    model = model.to(device)
        
 
    for dataset_name in dataset_names:
        # Post crops and processing getting the transforms
        dataset = configure_dataset(root_dir, dataset_name, transform=transform)
        train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_feat = []
        for images, label in tqdm(train_dataloader, total=len(train_dataloader)):
            patch_height, patch_width = model_instance.get_patch_info()
            images = create_pad(images, patch_width, patch_height)

            batch_feat = model_instance(images)
            all_feat.append(batch_feat)
       
        all_feat = np.concatenate(all_feat)

        if all_feat.ndim == 4:
            all_feat = all_feat.squeeze(2).squeeze(2)
        elif all_feat.ndim == 3:
            all_feat = all_feat.squeeze(2)
        elif all_feat.ndim == 2:
            all_feat = all_feat.squeeze()

        
        feature_path = feature_path = f'{feature_dir}/{dataset_name}/{feature_file}'
        np.save(feature_path, all_feat)
        torch.cuda.empty_cache() # new line
        


def get_parser():
   
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, help="The root directory of the original images", required=True)
    parser.add_argument("--feat_dir", type=str, help="The directory that contains the features", required=True)
    parser.add_argument("--model", type=str, help="The type of model that is being trained and evaluated (mae, openphenom, dinov2 or vit)", required=True) # Choices to get from 'mae', 'vit', 'dinov2', 'dinov3', 'openphenom'
    parser.add_argument("--gpu", type=int, default=0, help="The gpu that is currently available/not in use", required=False)
    parser.add_argument("--batch_size", type=int, default=64, help="Select a batch size that works for your gpu size", required=True)
    parser.add_argument("--model_size", type=str, default="small", help="Size of the ViT model (small or base)", choices=['small', 'base', 'large'])
    parser.add_argument("--model_path", type=str, help="Path to the model weights", required=False)

    return parser

if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()

    root_dir = args.root_dir
    feat_dir = args.feat_dir
    model = args.model
    gpu = args.gpu
    batch_size = args.batch_size

    get_save_features(feat_dir, root_dir, model, gpu, batch_size)

