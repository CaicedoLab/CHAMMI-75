from .utils import get_resized_dims
import os
import zipfile
import torch
import random
from safetensors.torch import load
from torchvision.transforms.v2.functional import resize
from torchvision import disable_beta_transforms_warning
from pathlib import Path
disable_beta_transforms_warning()

class GuidedCrop(object):
    def __init__(self, crop_size: tuple[int,int], crop_data: str):
        assert len(crop_size) == 2 and isinstance(crop_size[0], int) and isinstance(crop_size[1], int)
        assert isinstance(crop_data, str) and crop_data.endswith('zip')
        
        self.crop_size = crop_size
        crop_data = os.path.abspath(os.path.expanduser(crop_data))
        self.data = zipfile.ZipFile(crop_data)
        self.data_paths = set([file.filename for file in self.data.filelist if not file.is_dir()])

    def __call__(self, sample:torch.Tensor, sample_path:str) -> torch.Tensor: 
        if sample_path in self.data_paths:
            with self.data.open(sample_path) as f:
                image_height, image_width = sample.shape[1], sample.shape[2]
                
                possible_centroids = load(f.read())['data']
                chosen_centroid = possible_centroids[random.randint(0, possible_centroids.shape[0] - 1), :]
                x, y = chosen_centroid[0], chosen_centroid[1]
                crop_height, crop_width = self.crop_size[0], self.crop_size[1]  
                
                # Divide by 2, as we want half the crop size on each size of the center point
                x1, y1, x2, y2 = get_crop_location(crop_height//2, crop_width//2, y, x, image_height, image_width)
                
                cropped_sample = sample[:, y1:y2, x1:x2]
                return cropped_sample
        else:
            raise ValueError("Sample path is not in the guided crop data. Please check why this function was called.")
        
def get_crop_location(crop_height:int, crop_width:int, y_center:int, x_center:int,  image_height: int, image_width: int):
    # subtraction goes up or left, addition goes down or right
    y1 = y_center - crop_height 
    y2 = y_center + crop_height
    x1 = x_center - crop_width
    x2 = x_center + crop_width

    if y1 < 0:
        y2 = y2 - y1 # add into down direction -(-) = +
        y1 = 0
    elif y2 > image_height:
        y1 = y1 - (y2-image_height) # Move y1 up the difference
        y2 = image_height
    
    if x1 < 0:
        x2 = x2 - x1
        x1 = 0
    elif x2 > image_width:
        x1 = x1 - (x2-image_width)
        x2 = image_width
        
    return x1, y1, x2, y2