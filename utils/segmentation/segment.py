import os
import safetensors.numpy
from torch.utils.data import DataLoader
from torchvision import disable_beta_transforms_warning
from torchvision.io import decode_image, read_image
from torch.utils.data import Dataset
from cellpose import models
import polars as pl
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import scipy
import safetensors
import skimage
import cv2
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from skimage import filters, morphology, measure, segmentation
from skimage.segmentation import clear_border

from accelerate import Accelerator
accelerator = Accelerator()
disable_beta_transforms_warning()

OVERRIDES = {'experiment.well':pl.String, 
             'experiment.plate':pl.String, 
             'microscopy.fov': pl.String, 
             'microscopy.magnification': pl.String, 
             'geometry.depth': pl.String,
             'geometry.z_slice': pl.String
             }

class UnZippedImageArchive(Dataset):
    """Basic unzipped image arch. This will no longer be used. 
       Remove when unzipped support is added to the IterableImageArchive
    """
    def __init__(self, output_path: str, overwrite: bool = False) -> None:
        super().__init__()
        self.overwrite = overwrite
        self.output_path = os.path.expanduser(output_path)
        self.configs: pl.DataFrame = pl.read_csv('./config.tsv', separator='\t')
        self.ds10 = self.configs['study'].unique().to_list()
        self.configs = self.configs.to_pandas()
        self.imgs_base = '/scr/data'
        self.meta_path = '/scr/vidit/metadata/75ds_large_meta_fixes.csv'
        self.data = self.get_dataset()
        self.size = self.data['imaging.multi_channel_id'].unique().len()
        
        multi_channel_ids = set(self.data['imaging.multi_channel_id'].unique().to_list())
        self.data = pl.read_csv(self.meta_path, schema_overrides=OVERRIDES).to_pandas().groupby('imaging.multi_channel_id')
        self.data = [data for crit, data in self.data if crit in multi_channel_ids]

    def __len__(self):
        return self.size
        
    def get_dataset(self):
        meta = pl.read_csv(self.meta_path, schema_overrides=OVERRIDES)
        meta = meta.sort('imaging.multi_channel_id').filter(pl.col('experiment.study').is_in(self.ds10))
        if not self.overwrite:
            base_path = self.output_path
            paths = []
            for path, _, files in os.walk(base_path):
                for file in files:
                    file_path = os.path.join(path.replace(base_path, ''), file)
                    paths.append(file_path.replace('.safetensors', '.png')[1:])
        
        paths = set(paths)
        meta = meta.filter(~pl.col('storage.path').is_in(paths))
        return meta
    
    def __getitem__(self, idx):
        data: pd.DataFrame = self.data[idx]
        id = data['imaging.multi_channel_id'].iloc[0]
        
        data = data.sort_values('imaging.channel')
        images_paths = [os.path.join(self.imgs_base, path) for path in data['storage.path'].to_list()]
        
        study = data['experiment.study'].iloc[0]       
        channel_type = ','.join(data.sort_values('imaging.channel_type')['imaging.channel_type'].to_list())
        channel_settings = self.configs[(self.configs['study'] == study) & (self.configs['config'] == channel_type)]
        try:
            col_eq = channel_settings['seg_cfg'].iloc[0]
        except:
            print(channel_settings['seg_cfg'], 'WHY THIS BREAKING', channel_type, study, self.configs['study'], self.configs['config'],flush=True)
        diameter = channel_settings['diameter'].iloc[0]
        images = [read_image(image)[0] for image in images_paths]
    
        image_data = {'id': id, 'study': [path for path in data['storage.path'].to_list()]}
        if col_eq == "classical":
            return images, image_data
        elif col_eq == 'nucleus':
            col_eq = int(data[data['imaging.channel_type'] == 'nucleus']['imaging.channel'].item())
        elif col_eq == 'threshold':
            image_data['config'] = 'threshold'
            return images, image_data
        elif col_eq == 'snake':
            image_data['config'] = 'snake'
            return images, image_data
        elif col_eq == "skip":
            image_data['config'] = col_eq
            return images, image_data

        col_eq = [col_eq] if isinstance(col_eq, int) else col_eq.split(',')
        col_eq = [int(col) for col in col_eq]
        
        channel_axis = 1
        if len(col_eq) == 2:
                image_data['axis'] = channel_axis
                channels_config = [1, 2]
        else:
            channel_axis = -1
            channels_config = [0, 0]
            
        image_data['config'] = channels_config
        image_data['diameter'] = diameter
        try:

            if col_eq[0] != 0:
                cellpose_images = [images[idx-1] for idx in col_eq]
            else:
                cellpose_images = images
        except:
            print(col_eq)
            data['experiment.study']
            print("error goodbye")
            
        return cellpose_images, image_data

def main():
    output_path = '/scr/data/75ds_large_segmentations'
    dataset = UnZippedImageArchive(output_path)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
    
    device = torch.device(f'cuda:{accelerator.local_process_index}' if torch.cuda.is_available() else 'cpu')
    cellpose_model = models.CellposeModel(model_type='cyto3', device=device)
    data_loader = accelerator.prepare(data_loader)
    if accelerator.is_main_process:
        data_loader = tqdm(data_loader)

    for images, metadata in data_loader:     
        centers = []
        sizes = []

        if 'config' in metadata:
            if metadata['config'][0] == 'skip':
                continue

            elif metadata['config'][0] == 'snake':
                # Snake/worm segmentation
                # Assuming single channel image for worms
                image = images[0][0].cpu().numpy()
                
                # Otsu thresholding
                thresh = filters.threshold_otsu(image)
                binary = image < thresh  # Worms are darker than background
                binary = morphology.remove_small_objects(binary, min_size=300)
                binary = morphology.closing(binary, morphology.disk(10))
                binary = morphology.remove_small_holes(binary, area_threshold=200)
                
                # Distance transform for watershed seeds
                distance = ndi.distance_transform_edt(binary)
                coords = peak_local_max(distance, min_distance=50, threshold_abs=25)
                mask = np.zeros(distance.shape, dtype=bool)
                mask[tuple(coords.T)] = True
                markers, _ = ndi.label(mask)
                
                # Watershed segmentation
                labels = segmentation.watershed(-distance, markers, mask=binary)
                
                # Extract centers and sizes for each segmented worm
                num_worms = labels.max()
                if num_worms == 0:
                    continue
                    
                for worm_id in range(1, num_worms + 1):
                    row_indices, col_indices = np.where(labels == worm_id)
                    if len(row_indices) > 0:  # Make sure worm exists
                        avg_x, avg_y = row_indices.mean(), col_indices.mean()
                        center_x = int(round(avg_x))
                        center_y = int(round(avg_y))
                        centers.append((center_y, center_x))
                        
                        # Calculate worm size (number of pixels)
                        worm_size = len(row_indices)
                        sizes.append(worm_size)
            elif metadata['config'][0] == 'threshold':
                # Threshold-based segmentation using edge detection
                image = images[0][0].cpu().numpy()
                
                # Edge detection + morphology
                edges = filters.sobel(image)
                
                # Threshold edges to get strong boundaries
                edge_thresh = filters.threshold_otsu(edges)
                edge_mask = edges > edge_thresh
                
                # Close gaps in edges to form complete boundaries
                edge_mask = morphology.closing(edge_mask, morphology.disk(10))
                
                # Get all connected components
                labeled = measure.label(edge_mask)
                props = measure.regionprops(labeled)
                
                if len(props) == 0:
                    continue
                
                for prop in props:
                    if prop.area > 100:  # Filter small noise
                        centroid_y, centroid_x = prop.centroid
                        centers.append((int(round(centroid_x)), int(round(centroid_y))))
                        sizes.append(prop.area)
            else:
                with torch.no_grad():
                    axis = None if 'axis' not in metadata else metadata['axis']
                    channels = [int(ch) for ch in metadata['config']]
                    diameter = int(metadata['diameter'])
                    cellpose_images = [image[0].cpu().numpy() for image in images]
                    masks, _, _ = cellpose_model.eval(cellpose_images, channels=channels, channel_axis=axis, do_3D=False, diameter=diameter, batch_size=256)
                    mask: np.ndarray = masks[0]
                    num_cells = mask.max()
                    if num_cells == 0:
                        continue
                    for cell in range(num_cells):
                        row_indices, col_indices = np.where(mask == cell)
                        avg_x, avg_y = row_indices.mean(), col_indices.mean()
                        center_x = int(round(avg_x))
                        center_y = int(round(avg_y))
                        centers.append((center_y, center_x))      

                        cell_size = len(row_indices)
                        sizes.append(cell_size)
    
        for path in [path[0] for path in metadata['study']]:
            if len(centers) > 0:
                output_file = os.path.join(output_path, path.replace('.png', '.safetensors'))
                os.makedirs(os.path.dirname(output_file), exist_ok=True)

                save_dict = {'data': np.array(centers)}
                if 'config' in metadata and metadata['config'][0] != 'skip' and metadata['config'][0] != 'snake' and metadata['config'][0] != 'threshold' and len(sizes) > 0:
                    save_dict['sizes'] = np.array(sizes)
                safetensors.numpy.save_file(save_dict, output_file)

    accelerator.wait_for_everyone()
    accelerator.end_training()
        
if __name__ == '__main__':
    main()