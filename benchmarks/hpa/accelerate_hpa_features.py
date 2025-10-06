import os
import pandas as pd
import torch
import cv2
import numpy as np
from tqdm import tqdm
from torch import nn
import polars as pl
from torch.utils.data import Dataset
from torchvision.io import decode_image
import matplotlib.pyplot as plt
import sys
from torchvision.transforms import v2
from accelerate import Accelerator
from torchvision import transforms
import argparse
import yaml
sys.path.append("../")
from models import ViTClass, MAEModel, DinoV2, OpenPhenom, get_model

#from vit_pool import ViTPoolModel

import atexit

accelerator = Accelerator()

def cleanup_resources():
    """Clean up resources before exit"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if 'accelerator' in globals():
            accelerator.free_memory()
    except:
        pass

# Register cleanup function
atexit.register(cleanup_resources)

def custom_collate_fn(batch):
    """Custom collate function to handle None values"""
    # Filter out None values
    valid_batch = [(img, row) for img, row in batch if img is not None and row is not None]
    
    if not valid_batch:
        return None, None
    
    images, rows = zip(*valid_batch)
    
    try:
        # Convert to tensors if needed and stack
        tensor_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                tensor_images.append(torch.from_numpy(img).float())
            elif isinstance(img, torch.Tensor):
                tensor_images.append(img.float())
            else:
                print(f"Unexpected image type: {type(img)}")
                continue
            
        images_tensor = torch.stack(tensor_images)
        return images_tensor, list(rows)
        
    except Exception as e:
        print(f"Error in collate function: {e}")
        return None, None


'''
Custom Class to load HPA images for feature extraction.
'''
class UnZippedImageArchive(Dataset):
    """Basic unzipped image arch. This will no longer be used. 
       Remove when unzipped support is added to the IterableImageArchive
    """
    def __init__(self, root_dir: str, transform=None) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.metadata_path = os.path.join(root_dir, 'metadata.csv')
        self.metadata = pl.read_csv(self.metadata_path).rows(named=True)
        self.transform = transform
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        # microtubule fluorescence,  Blue (B) channel
        # endoplasmic reticulum,  Green (G) channel
        # DNA, Red (R) channel
        # Protein of interest, Alpha (A) channel
        # https://virtualcellmodels.cziscience.com/dataset/01933229-3c87-7818-be80-d7e5578bb0b7
        row = self.metadata[idx]
        plate = str(row['if_plate_id'])
        position = row['position']
        sample = str(row['sample'])
        cell_id = str(int(row['cell_id']))
        image_path = os.path.join(self.root_dir, plate, f"{plate}_{position}_{sample}_{cell_id}_cell_image.png")

        # Check if file exists
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return None, None
        
        try:
            # Try to load the image
            image = cv2.imread(image_path, -1)
            
            if image is None:
                print(f"Failed to load image: {image_path}")
                return None, None
                
            # Transpose to (C, H, W) format
            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.float32)  # Ensure image is float32 for transforms
                
            # Apply transforms if provided
            if self.transform:
                # Convert to tensor for transforms
                image_tensor = torch.from_numpy(image)
                image = self.transform(image_tensor)
                
            return image, row
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None, None
        
'''
New extraction feature function to collate all the features from the GPUs on CPUs as when using SubCell Model, the gpu goes out of memory from the previous method!
'''
def extract_features(dataloader: torch.utils.data.DataLoader, model_instance: object, config_path: str = None, model_path: str = None, model_size: str = None, output_folder: str = None):
    """Alternative approach: Save features from each process separately, then combine"""

    all_features = []
    all_rows = []
    idx = 0
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc=f"Extracting features on GPU {accelerator.local_process_index}", disable=not accelerator.is_local_main_process):
            if batch_data[0] is None:
                continue
                
            images, rows = batch_data
            batch_size = images.shape[0]
            num_channels = images.shape[1]
            
            features = model_instance(images)
            
            all_features.append(features)
            all_rows.extend(rows)
    
    # Save this process's data to a temporary file
    if all_features:
        feature_data = np.concatenate(all_features, axis=0)
        # Save both features and metadata for this process
        os.makedirs(output_folder, exist_ok=True)
        process_file = f"{output_folder}/process_{accelerator.process_index}_data.npz"
        np.savez_compressed(
            process_file,
            features=feature_data,
            metadata=np.array(all_rows, dtype=object)
        )
        
        # Wait for all processes to finish saving
        accelerator.wait_for_everyone()
        accelerator.end_training()
        
        # Main process combines all files
        if accelerator.is_main_process:
            all_features_list = []
            all_rows_combined = []
            
            for i in range(accelerator.num_processes):
                process_file = f"{output_folder}/process_{i}_data.npz"
                if os.path.exists(process_file):
                    data = np.load(process_file, allow_pickle=True)
                    features = data['features']
                    metadata = data['metadata']
                    if features.shape[0] > 0:  # Only add non-empty arrays
                        all_features_list.append(features)
                    all_rows_combined.extend(metadata.tolist())
                    # Clean up
                    os.remove(process_file)
            
            # Combine all features
            if all_features_list:
                all_features_combined = np.concatenate(all_features_list, axis=0)
            
            # Save final results
            torch.save((all_rows_combined, torch.from_numpy(all_features_combined)), f"{output_folder}/all_features.pth")
            
            df = pd.DataFrame(all_rows_combined)
            df.to_csv(f"{output_folder}/metadata.csv", index=False)
            print(f"Saved {len(all_rows_combined)} samples with {all_features_combined.shape[1]} features")
            print(f"Saved metadata with shape: {df.shape}")
            
            return all_rows_combined, all_features_combined
        
        return None, None

def main():
    parser = argparse.ArgumentParser(description='Extract features using VIT or subcell model')
    parser.add_argument('--model', type=str, default='vit',
                        help='Model to use for feature extraction (default: vit)') # Choices to get from 'mae', 'subcell', 'vit', 'dinov2', 'dinov3', 'openphenom'
    parser.add_argument('--config_path', type=str, default="/mnt/cephfs/mir/jcaicedo/morphem/dataset/models/subcell_models/all_channels_ViT-ProtS-Pool.yaml",
                        help='Path to config file for subcell model (required when using subcell)')
    parser.add_argument('--image_folder', type=str, default="/scr/data/cell_crops",
                        help='Path to image folder', required=True)
    parser.add_argument('--output_folder', type=str, default="/scr/data/HPA_features",
                        help='Output folder for features', required=True)
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for processing')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--model_path', type=str, default="", help = "Path to where the model is located")
    parser.add_argument('--model_size', type=str, choices=['small', 'base'], default='small')
    args = parser.parse_args()


    # Validate arguments
    if args.model == 'subcell' and args.config_path is None:
        raise ValueError("config_path is required when using subcell model")

    # Print process info
    print(f"Process {accelerator.process_index} of {accelerator.num_processes} started")
    print(f"Using device: {accelerator.device}")

# Initialize model on accelerator device
    model_instance = get_model(model_name=args.model, device=accelerator.device, model_path=args.model_path, model_size=args.model_size)
    model, transform = model_instance.get_model()
    
    if args.model == "channelvit":
        model_instance.set_dataset("mini-HPA", args.model_path)
    
    dataset = UnZippedImageArchive(
        root_dir=args.image_folder, 
        transform=transform
    )

    # Create dataloader - accelerator will handle the distribution
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,  # Reduce num_workers per GPU since we have multiple GPUs
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    
    dataloader = accelerator.prepare(dataloader)

    # Extract features
    rows, feature_data = extract_features(
        dataloader=dataloader, 
        model_instance=model_instance,
        output_folder=args.output_folder,
        config_path=args.config_path,
        model_path=args.model_path,
        model_size=args.model_size
    )
    
    if accelerator.is_main_process:
        print("Feature extraction complete!")
        if rows is not None:
            print(f"Total samples processed: {len(rows)}")
        if feature_data is not None:
            print(f"Feature tensor shape: {feature_data.shape}")
    
    #print(f"Process {accelerator.process_index} finished")

if __name__ == "__main__":
    main()
    sys.exit(0)
