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

# Initialize accelerator at the top
accelerator = Accelerator()

sys.path.append("../morphem")
from vision_transformer import vit_small


'''
Class to call DINOv1 model for feature extraction.
'''
class ViTClass():
    def __init__(self, device):
        self.device = device

        # Create model with in_chans=1 to match training setup
        self.model = vit_small()
        remove_prefixes = ["module.backbone.", "module.", "module.head."]

        # Load model weights
        student_model = torch.load("/scr/vidit/Models/Dino_Small_75ds_Guided/checkpoint.pth", map_location=device)['student']
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


class PerImageNormalize(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize each image independently.
        
        Args:
            x (torch.Tensor): Input tensor of shape (C, H, W) for single image
                              or (N, C, H, W) for batch of images
        
        Returns:
            torch.Tensor: Normalized tensor of the same shape.
        """
        if x.dim() == 3:  # Single image (C, H, W)
            # Compute mean and std across all pixels and channels
            mean = x.mean()
            std = x.std()
            normalized_x = (x - mean) / (std + self.eps)
            return normalized_x
        
        elif x.dim() == 4:  # Batch of images (N, C, H, W)
            batch_size = x.shape[0]
            # Compute mean and std for each image in the batch
            x_flat = x.view(batch_size, -1)
            mean = x_flat.mean(dim=1, keepdim=True)
            std = x_flat.std(dim=1, keepdim=True)
            
            # Reshape for broadcasting
            mean = mean.view(batch_size, 1, 1, 1)
            std = std.view(batch_size, 1, 1, 1)
            
            normalized_x = (x - mean) / (std + self.eps)
            return normalized_x
        
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {x.dim()}D tensor with shape {x.shape}")

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
    def __init__(self, root_dir: str= '/scr/data/cell_crops/', transform=None) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.metadata_path = os.path.join(self.root_dir, 'metadata.csv')
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


def extract_features_hpa(dataloader: torch.utils.data.DataLoader, output_folder: str):
    """Extract features from HPA single-cell crops using multi-GPU"""

    # Initialize model on accelerator device
    vit_instance = ViTClass(accelerator.device) 
    vit_model = vit_instance.get_model()
    vit_model.eval()
    
    # Prepare model and dataloader with accelerator
    vit_model, dataloader = accelerator.prepare(vit_model, dataloader)
    
    all_features = []
    all_rows = []
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc=f"Extracting features on GPU {accelerator.local_process_index}", disable=not accelerator.is_local_main_process):
            if batch_data[0] is None:  # Skip None batches
                continue
                
            images, rows = batch_data
            batch_size = images.shape[0]
            num_channels = images.shape[1]  # Should be 4 for RGBA
            
            # Initialize feature array for this batch
            batch_feat = torch.zeros((batch_size, num_channels * 384), device=accelerator.device)
            
            # Process each channel separately
            for c in range(num_channels):
                # Extract single channel and add channel dimension
                single_channel = images[:, c, :, :].unsqueeze(1).float()
                
                # Forward pass through model
                # Access the underlying model when wrapped in DDP
                if hasattr(vit_model, 'module'):
                    output = vit_model.module.forward_features(single_channel)
                else:
                    output = vit_model.forward_features(single_channel)
                feat_temp = output["x_norm_clstoken"]  # Keep on GPU
                
                # Store features for this channel
                batch_feat[:, c * 384:(c + 1) * 384] = feat_temp
            
            # Collect features and rows
            all_features.append(batch_feat)
            all_rows.extend(rows)
    
    # Concatenate all features on current device
    if all_features:
        feature_data = torch.cat(all_features, dim=0)  # [N, feature_dim]
    else:
        feature_data = torch.empty((0, num_channels * 384), device=accelerator.device)
    
    # Gather results from all processes
    all_features_gathered = accelerator.gather(feature_data)
    
    # For metadata, we need to handle it differently since gather_object might not be available
    # Convert rows to tensors for gathering, then convert back
    if all_rows:
        # Create a simple approach: save locally and combine on main process
        local_save_path = f"{output_folder}/temp_process_{accelerator.process_index}.pkl"
        os.makedirs(output_folder, exist_ok=True)
        
        import pickle
        with open(local_save_path, 'wb') as f:
            pickle.dump(all_rows, f)
    
    # Wait for all processes to save their data
    accelerator.wait_for_everyone()
    
    # Main process collects all the temporary files
    if accelerator.is_main_process:
        all_rows_gathered = []
        for i in range(accelerator.num_processes):
            temp_file = f"{output_folder}/temp_process_{i}.pkl"
            if os.path.exists(temp_file):
                with open(temp_file, 'rb') as f:
                    process_rows = pickle.load(f)
                    all_rows_gathered.extend(process_rows)
                # Clean up temp file
                os.remove(temp_file)
    else:
        all_rows_gathered = None
    
    # Save only on main process
    if accelerator.is_main_process:
        # Move to CPU for saving
        all_features_cpu = all_features_gathered.cpu()
        
        # Create output directory
        os.makedirs(output_folder, exist_ok=True)
        
        # Save in format expected by train_classification.py
        torch.save((all_rows_gathered, all_features_cpu), f"{output_folder}/all_features.pth")
        print(f"Saved {len(all_rows_gathered)} samples with {all_features_cpu.shape[1]} features")
        
        # Also save as numpy for convenience
        np.save(f"{output_folder}/features.npy", all_features_cpu.numpy())
        
        # Convert rows to DataFrame and save
        df = pd.DataFrame(all_rows_gathered)
        df.to_csv(f"{output_folder}/metadata.csv", index=False)
        print(f"Saved metadata with shape: {df.shape}")
    
    # Wait for all processes to complete
    accelerator.wait_for_everyone()
    
    return all_rows_gathered if accelerator.is_main_process else None, \
           all_features_cpu if accelerator.is_main_process else None


if __name__ == "__main__":
    # Configuration
    csv_file = "/scr/data/cell_crops/metadata.csv"  # Your DataFrame with the columns you showed
    image_folder = "/scr/data/cell_crops"
    output_folder = "/scr/data/HPA_features"
    
    # Print process info
    print(f"Process {accelerator.process_index} of {accelerator.num_processes} started")
    print(f"Using device: {accelerator.device}")
    
    # Initialize dataset and dataloader
    dataset = UnZippedImageArchive(
        root_dir=image_folder, 
        transform=transforms.Compose([
            PerImageNormalize(),
            v2.Resize(size=(224, 224), antialias=True)
        ])
    )
    
    # Create dataloader - accelerator will handle the distribution
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=128, 
        shuffle=False, 
        num_workers=10,  # Reduce num_workers per GPU since we have multiple GPUs
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    
    # Extract features
    rows, feature_data = extract_features_hpa(
        dataloader=dataloader, 
        output_folder=output_folder
    )
    
    if accelerator.is_main_process:
        print("Feature extraction complete!")
        if rows is not None:
            print(f"Total samples processed: {len(rows)}")
        if feature_data is not None:
            print(f"Feature tensor shape: {feature_data.shape}")
    
    print(f"Process {accelerator.process_index} finished")