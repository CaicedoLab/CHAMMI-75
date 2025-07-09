import os
import pandas as pd
import torch
import cv2
import numpy as np
from tqdm import tqdm
from torch import nn
import os
import numpy as np
import polars as pl
from torch.utils.data import Dataset
from torchvision.io import decode_image
import matplotlib.pyplot as plt
import torch
import sys
from torchvision.transforms import v2
from accelerate import Accelerator
accelerator = Accelerator()

sys.path.append("../morphem")
from vision_transformer import vit_small


'''
Class to call DINOv1 model for feature extraction.
'''
class ViTClass():
    def __init__(self, gpu):
        self.device = f"cuda:{gpu}" if torch.cuda.is_available() else 'cpu'

        # Create model with in_chans=1 to match training setup
        self.model = vit_small()
        remove_prefixes = ["module.backbone.", "module.", "module.head."]

        # Load model weights
        student_model = torch.load("/scr/vidit/Models/Dino_Small_75ds_Guided/checkpoint.pth")['student']
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
        self.eps = eps  # Small value to avoid division by zero

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize each image in the batch independently.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W)
                              where N = batch size, C = channels, H = height, W = width.
        
        Returns:
            torch.Tensor: Normalized tensor of the same shape.
        """
        # Compute the mean and standard deviation for each image across all channels, height, and width
        mean = x.view(x.size(0), -1).mean(dim=1, keepdim=True).view(x.size(0), 1, 1, 1)
        std = x.view(x.size(0), -1).std(dim=1, keepdim=True).view(x.size(0), 1, 1, 1)

        # Normalize the input tensor
        normalized_x = (x - mean) / (std + self.eps)

        return normalized_x

def custom_collate_fn(batch):
    """Custom collate function to handle None values"""
    # Filter out None value
    images, rows = zip(*batch)
    
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
        # Try to load the image
        image = cv2.imread(image_path, -1)
            
        # Transpose to (C, H, W) format
        image = np.transpose(image, (2, 0, 1))

        image = image.astype(np.float32)  # Ensure image is float32 for transforms
            
        # Apply transforms if provided
        if self.transform:
            # Convert to tensor for transforms
            image_tensor = torch.from_numpy(image)
            image = self.transform(image_tensor)
            
        return image, row


dataset = UnZippedImageArchive(root_dir='/scr/data/cell_crops/')


def extract_features_hpa(dataloader: torch.utils.data.Dataset, output_folder: str):
    """Extract features from HPA single-cell crops"""

    all_features_np = np.zeros((0, 1536))  # Initialize empty array for features # Second dimension is 384*4 = 1536
    all_rows = []
    all_features = []

    device = torch.device(f"cuda:{accelerator.local_process_index}" if torch.cuda.is_available() else "cpu")
    vit_instance = ViTClass(accelerator.local_process_index) 
    vit_model = vit_instance.get_model()
    vit_model.eval()
    
    with torch.no_grad():
        features = np.zeros((len(dataloader.dataset), 1536))  # Initialize empty array for features
        for batch, rows in tqdm(dataloader, desc="Extracting features"):
            images = batch.to(device)
            batch_feat = np.zeros((images.shape[0], 1536))
            for c in range(images.shape[1]):
                # Copy each channel three times
                single_channel = images[:, c, :, :].unsqueeze(1).float().to(device)
    
                output = vit_model.forward_features(single_channel)
                feat_temp = output["x_norm_clstoken"].cpu().detach().numpy()
                batch_feat[:, c * 384:(c + 1) * 384] = feat_temp
            
            features = features[:len(all_features) + len(batch_feat), :]  # Resize to fit new features
            all_rows.extend(rows)

        all_features = np.concatenate(all_features)
        all_rows = np.array(all_rows)
    
    all_features = [torch.from_numpy(f) if isinstance(f, np.ndarray) else f for f in all_features]
    # Concatenate all features
    feature_data = torch.cat(all_features, dim=0)  # [N, feature_dim]
    
    # Save in format expected by train_classification.py
    torch.save((all_rows, feature_data), f"{output_folder}/all_features.pth")
    print(f"Saved {len(df)} samples with {feature_data.shape[1]} features")
    
    return df, feature_data

if __name__ == "__main__":
    # Configuration
    csv_file = "/scr/data/cell_crops/metadata.csv"  # Your DataFrame with the columns you showed
    image_folder = "/scr/data/cell_crops"
    output_folder = "/scr/data/HPA_features"

    device = torch.device(f"cuda:{accelerator.local_process_index}" if torch.cuda.is_available() else "cpu")
    
    # Load metadata
    df = pd.read_csv(csv_file)
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)

    # Initialize dataset and dataloader
    dataset = UnZippedImageArchive(root_dir=image_folder, transform=v2.Resize(size=(224, 224), antialias=True))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=16, collate_fn=custom_collate_fn)
    # Extract features
    df, feature_data = extract_features_hpa(
        dataloader=dataloader, output_folder = output_folder
    )
    
    print("Feature extraction complete!")
    print(f"DataFrame shape: {df.shape}")
    print(f"Feature tensor shape: {feature_data.shape}")