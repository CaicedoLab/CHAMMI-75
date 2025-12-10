import torch
from torch.utils.data import Dataset
import numpy as np
import os
import glob
import skimage.io as io
from torchvision.transforms import v2

class ToTensorNormalize(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, image):
        # raw input is uint8 - maximum of 255. Convert it back to 0 and 1 scale, float32
        image_tensor = (image / 255).astype('float32')
        image_tensor = np.clip(image_tensor, 0, 1)
        tensor = torch.from_numpy(image_tensor)
        
        # Add channel dimension if it's a 2D image [H, W] -> [1, H, W]
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        
        return tensor


class PerImageNormalize(object):
    def __init__(self, eps=1e-7):
        self.eps = eps
    
    def __call__(self, image_tensor):
        # Expects a torch tensor
        if not isinstance(image_tensor, torch.Tensor):
            image_tensor = torch.from_numpy(image_tensor).float()
        
        # Add channel dimension if needed [H, W] -> [1, H, W]
        if image_tensor.ndim == 2:
            image_tensor = image_tensor.unsqueeze(0)
        
        # Now normalize: [C, H, W]
        mean = torch.mean(image_tensor, dim=(1, 2), keepdim=True)
        std = torch.std(image_tensor, dim=(1, 2), keepdim=True)
        
        normalized_tensor = (image_tensor - mean) / (std + self.eps)
        return normalized_tensor


class RBC_Dataloader(Dataset):
    def __init__(self, datadir: str, transform=None):
        self.swiss_image_paths = glob.glob(os.path.join(datadir, "Swiss", "**", "*.ome.tif"), recursive=True)
        self.canadian_image_paths = glob.glob(os.path.join(datadir, "Canadian", "**", "*.ome.tif"), recursive=True)
        
        # Default transform chain
        if transform is None:
            self.transform = lambda img: PerImageNormalize()(ToTensorNormalize()(v2.Resize(224)(img)))
        else:
            self.transform = transform

    def load_image_dataframe(self, image_path):
        img = io.imread(image_path)
        classifier_name = image_path.split("/")[-2]
        if image_path in self.swiss_image_paths:
            label = "Swiss"
        else:
            label = "Canadian"
        
        # Apply transform to image only
        if self.transform:
            img = self.transform(img)
        
        sample = {'image': img, 'label': label, 'classifier_name': classifier_name}
        return sample

    def __len__(self):
        return len(self.swiss_image_paths) + len(self.canadian_image_paths)
    
    def __getitem__(self, idx):
        if idx < len(self.swiss_image_paths):
            image_path = self.swiss_image_paths[idx]
        else:
            image_path = self.canadian_image_paths[idx - len(self.swiss_image_paths)]
        
        sample = self.load_image_dataframe(image_path)
        return sample