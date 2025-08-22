import os
from skimage import io
from typing import Tuple
import polars as pl
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from safetensors.torch import load_file
from utils import get_segmentation_crops, get_masked_cells
from torchvision.transforms.v2.functional import resize, center_crop

class FeatureExtractionDataset(Dataset):
    def __init__(self, config: dict, inputs_dir: str):
        config = config
        self.feature_config = config['feature_extraction']
        self.use_masks = False
        meta_abspath = self._get_abs_path(config['metadata'])
        metadata = pl.read_csv(meta_abspath, separator=',')       
        plate = os.path.basename(inputs_dir).split('-')[1]
        metadata = metadata.filter(pl.col("storage.zip") == plate)
        
        self.inputs_dir = inputs_dir
        input_paths = os.listdir(inputs_dir)
        self.mask_paths = set([path for path in input_paths if path.endswith('.tiff')])
        
        # only ever 1 file in the zips so [0]
        safetensors_file = [path for path in input_paths if path.endswith('tensors')][0] 
        self.coords = load_file(os.path.join(self.inputs_dir, safetensors_file))
        self.plate = os.path.basename(inputs_dir) # this plate is the actual zip name w/o the .zip. 

        self.image_groups = self._get_filtered_image_groups(metadata)
        
    def _get_filtered_image_groups(self, metadata: pl.DataFrame):
        images = metadata.select(pl.all().sort_by('imaging.channel').over('imaging.multi_channel_id'))
        image_groups = list(images.group_by('imaging.multi_channel_id'))
        
        # as in image groups where the segmentation channel didn't find anything
        blank_image_groups = set()
        for criteria, group in image_groups: 
            image_names = group['storage.filename']
            found_mask = False
            for image_name in image_names:
                if self._get_st_key(image_name) in self.coords:
                    found_mask = True
            if not found_mask:
                blank_image_groups.add(criteria)
        
        return [group[1] for group in image_groups if group[0] not in blank_image_groups]      
    
    def _get_st_key(self, image_name):
        return f"{self.plate}/{image_name}"  
        
    def _read_images_get_masks(self, image_names):
        images = []
        mask = None
        for image_name in image_names:
            image_path = os.path.join(self.inputs_dir, image_name)
            images.append(read_image(image_path)) # make this async if we wanna be nerds 
            
            # We pre-filter to make sure that one of the image_names are in the masks_paths,
            # so this if will only ever trigger once.
            if self._get_st_key(image_name) in self.coords:
                segmented_image_name = image_name
                
                if self.use_masks:
                    potential_mask_name = f"{image_name[:-3]}tiff"
                    mask_path = os.path.join(self.inputs_dir, potential_mask_name)
                    mask = torch.from_numpy(io.imread(mask_path))
                
        return images, segmented_image_name, mask
        
    def _get_abs_path(self, path:str):
        return os.path.abspath(os.path.expanduser(path))
        
    def __len__(self):
        return len(self.image_groups)

    def __getitem__(self, index) -> Tuple[torch.Tensor, str]:
        group:pl.DataFrame = self.image_groups[index] 
        image_names = list(group['storage.filename'])
        multi_channel_id = group['imaging.multi_channel_id'][0]
        images, segmented_image_name, mask = self._read_images_get_masks(image_names)
        images = torch.concat(images, dim=0)
    
        img_coords = self.coords[self._get_st_key(segmented_image_name)]
    
        # y1, y2, x1, x2 coord order
        if self.use_masks:
            patches = get_masked_cells(images, img_coords, mask)
        else:
            patches = get_segmentation_crops(images, img_coords)
            
        # patches get normalized by each model in the way the model needs. Don't add here please.
        stacked = torch.stack(patches, dim=0)
        
        if stacked.shape[-1] != self.feature_config['crop']:
            stacked = center_crop(stacked, self.feature_config['crop'])
        if stacked.shape[-1] != self.feature_config['resize']:
            stacked = resize(stacked, size = (self.feature_config['resize']), antialias=True)
        
        return stacked, image_names, multi_channel_id
