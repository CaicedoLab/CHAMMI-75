from .dataset_config import DatasetConfig
from .dataset_transforms import GuidedCrop
from .dataset_functions import get_proc_split
from torch.utils.data import IterableDataset, Dataset
from torchvision.transforms import v2
from torchvision import disable_beta_transforms_warning
from torchvision.io import decode_image, read_image
import torch
import zipfile
import os 
import random
import pandas as pd
from io import StringIO
import json
import polars as pl
import math
import itertools
from collections import defaultdict

disable_beta_transforms_warning()

DS10 = ["hpa0018", "idr0002", "idr0008", "idr0086", "idr0088", "idr0089", "jump0001", "nidr0031", "nidr0032", "wtc0001"]


class ZipFileWrapper:
    """Wrapper for ZipInfo objects to add archive_index attribute"""
    def __init__(self, zipinfo, archive_index):
        self.zipinfo = zipinfo
        self.archive_index = archive_index
        
    @property
    def filename(self):
        return self.zipinfo.filename
        
    @property
    def is_dir(self):
        return self.zipinfo.is_dir()
        
    def __getattr__(self, name):
        # Delegate any other attribute access to the wrapped ZipInfo object
        return getattr(self.zipinfo, name)

class IterableImageArchive(IterableDataset):
    def __init__(self, config: DatasetConfig) -> None:
        super().__init__()

        self.config = config
        self.archive = None 
        self.image_paths = None 
        self.metadata_filename = None
        self.guided_crops: GuidedCrop = None
        self.default_transform = None
        self.metadata_df = None

    def load_archive(self):
        if isinstance(self.config.data_path, list) and len(self.config.data_path) == 2:
            # Handle two data paths
            self.archive = [zipfile.ZipFile(path, "r") for path in self.config.data_path]
            self.image_paths = []
            for i, archive in enumerate(self.archive):
                archive_images = [ZipFileWrapper(file, i) for file in archive.infolist() 
                                if not file.is_dir() and file.filename.endswith(self.config.img_type)]
                self.image_paths.extend(archive_images)
            print(f"Loaded {len(self.image_paths)} images from {self.config.data_path}")
        else:
            # Handle single data path (original behavior)
            self.archive = zipfile.ZipFile(self.config.data_path, "r")
            self.image_paths = [file for file in self.archive.infolist() 
                            if not file.is_dir() and file.filename.endswith(self.config.img_type)]
            print(f"Loaded {len(self.image_paths)} images from {self.config.data_path}")

    def return_sample(self, file_list: list):
        for file_path in file_list:
            # Handle multiple archives
            if isinstance(self.archive, list):
                current_archive = self.archive[file_path.archive_index]
                # Use the wrapped ZipInfo object for reading
                zipinfo_obj = file_path.zipinfo if hasattr(file_path, 'zipinfo') else file_path
                img_bytes = bytearray(current_archive.read(zipinfo_obj.filename))
            else:
                img_bytes = bytearray(self.archive.read(file_path.filename))
                
            try:
                torch_buffer = torch.frombuffer(img_bytes, dtype=torch.uint8)
                image_tensor = decode_image(torch_buffer)
                image_tensor = image_tensor.to(torch.float16)

                dataset = file_path.filename.split(os.sep)[1]
                
                baseline = get_crop_size(dataset)

                # Apply cropping based on baseline
                if baseline != (-1, -1):
                    # Calculate range with proper bounds to avoid randint errors
                    crop_height = random.randint(int(baseline[0] * 0.9), int(baseline[0] * 1.1))
                    crop_width = random.randint(int(baseline[1] * 0.9), int(baseline[1] * 1.1))
                    self.guided_crops.crop_size = (crop_height, crop_width)
                else:
                    self.guided_crops.crop_size = (-1, -1)

                # Apply guided crops if available
                if self.guided_crops.crop_size != (-1, -1) and self.config.guided_crops_path:
                    safetensors_name = file_path.filename[:-4] + ".safetensors"
                    safetensors_name = safetensors_name.replace("CHAMMI-75_train", "CHAMMI-75_guidance")
                    if safetensors_name in self.guided_crops.data_paths:
                        image_tensor = self.guided_crops(image_tensor, safetensors_name)
                    else:
                        pass

                # Apply additional transforms if configured
                if self.config.transform:
                    image_tensor = self.config.transform(image_tensor)
                
                # Yield based on mode
                if self.config.test:
                    yield file_path.filename
                else:
                    yield image_tensor
            except Exception as e:
                print(f"Error processing {file_path.filename}: {e}")
                import traceback
                traceback.print_exc()
            
    def worker_init_fn(self, worker_id):
        worker_info = torch.utils.data.get_worker_info()
        dataset:IterableImageArchive = worker_info.dataset
        dataset.load_archive()
        
        if dataset.config.guided_crops_path:
            dataset.guided_crops = GuidedCrop(dataset.config.guided_crops_size, dataset.config.guided_crops_path)

    def call_splitting_fns(self, data):
        output_buffer = data
        for split_fn in self.config.split_fns:
            output_buffer = split_fn(output_buffer, self.config)
        return output_buffer

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.load_archive()
            
        if self.config.guided_crops_path:
            self.default_transform = v2.RandomResizedCrop(size=self.guided_crops.crop_size, antialias=True)  
        
        worker_data = self.call_splitting_fns(self.image_paths)    
        samples = iter(self.return_sample(worker_data))


        return samples
    
    def __len__(self):
        if not self.image_paths:
            if isinstance(self.config.data_path, list) and len(self.config.data_path) == 2:
                # Handle two data paths
                image_paths = []
                for i, path in enumerate(self.config.data_path):
                    archive = zipfile.ZipFile(path, "r")
                    archive_images = [ZipFileWrapper(file, i) for file in archive.infolist() 
                                    if not file.is_dir() and file.filename.endswith(self.config.img_type)]
                    image_paths.extend(archive_images)
                    archive.close()
                self.image_paths = image_paths
            else:
                # Handle single data path (original behavior)
                archive = zipfile.ZipFile(self.config.data_path, "r")
                image_paths = [file for file in archive.infolist() 
                                if not file.is_dir() and file.filename.endswith(self.config.img_type)] 
                self.image_paths = image_paths
                archive.close()

        if self.config.num_procs > 1:
            return len(get_proc_split(self.image_paths, self.config))
        else:
            return len(self.image_paths)
        
# Required for our dataset config otherwise polars gets confuzzled
OVERRIDES = {'experiment.well':pl.String, 
             'experiment.plate':pl.String, 
             'microscopy.fov': pl.String, 
             'microscopy.magnification': pl.String, 
             'geometry.depth': pl.String,
             'geometry.z_slice': pl.String
             }    

class ChannelViTDataset(IterableImageArchive):
    def __init__(self, config: DatasetConfig) -> None:
        super().__init__(config)
        self.config = config

        if self.config.dataset_config:
            config_path: str = self.config.dataset_config 
        else:
            raise ValueError("dataset_config path to config file must be supplied")

        self.image_paths, self.channels = self.load_dataset_config(config_path)
        self.num_channels = len(self.channels) # for init channel_vit in_chans
        self.channel_map = self.init_channel_map(self.channels, True)
        
    def init_channel_map(self, channels: list, save: bool):
        channel_map = {}
        for idx, channel in enumerate(sorted(channels)):
            channel_map[channel] = idx
        
        if save:
            with open(os.path.join(self.config.output_dir, 'channel_map.json'), 'w') as f: f.write(json.dumps(channel_map))
        return channel_map

    def read_im(self, file_path: str):
        img_bytes = bytearray(self.archive.read(file_path))
        torch_buffer = torch.frombuffer(img_bytes, dtype=torch.uint8)
        image_tensor = decode_image(torch_buffer)
        image_tensor = image_tensor.to(torch.float16)
        return image_tensor
        
    def load_archive(self):
        self.archive = zipfile.ZipFile(self.config.data_path, "r")
    
    def return_sample(self, file_list: list):
        for file_group, channel_types in file_list:
            ims = [self.read_im(im) for im in file_group]
            image_tensor = torch.concat(ims, dim=0)
                    
            if self.guided_crops:
                dataset = file_group[0].split(os.sep)[1]
                baseline = get_crop_size(dataset)

                if baseline != (-1, -1):
                    crop_height = random.randint(int(baseline[0] * 0.9), int(baseline[0] * 1.1))
                    crop_width = random.randint(int(baseline[1] * 0.9), int(baseline[1] * 1.1))
                    self.guided_crops.crop_size = (crop_height, crop_width)
                else:
                    self.guided_crops.crop_size = (-1, -1)
                
                if self.guided_crops.crop_size != (-1, -1) and self.config.guided_crops_path:
                    safetensors_name:str = file_group[0][:-4] + ".safetensors"
                    safetensors_name = "CHAMMI-75_guidance/" + safetensors_name.split('/', maxsplit=1)[1] 
                    if safetensors_name in self.guided_crops.data_paths:
                        image_tensor = self.guided_crops(image_tensor, safetensors_name)
                                    
            if self.config.transform:
                image_tensor = self.config.transform(image_tensor)
            
            if type(image_tensor) == list:
                channel_types = [tuple(channel_types)]*len(image_tensor)
                sample = list(zip(image_tensor, channel_types))
            else:
                channel_types = tuple(channel_types)
                sample = image_tensor, channel_types
            
            # Yield based on mode
            if self.config.test:
                yield file_group
            else:
                yield sample
    
    
    def mask_crop(self, crop: torch.Tensor, max_chans:int):
        c,w,h = crop.shape
        mask = torch.zeros(max_chans, w, h, dtype=torch.float16)
        mask[:c, :w, :h] = crop
        return mask
    
    def collate_crops(self, crops: list[torch.Tensor], max_chans: int):
        crops = [self.mask_crop(crop, max_chans) for crop in crops]
        collated_crops = torch.stack(crops)
        return collated_crops
    
    def collate_fn(self, samples: list[tuple[torch.Tensor, list[str]]]):        
        max_chans = max([sample[0][0].shape[0] for sample in samples])
        num_crops = len(samples[0])
    
        collate_list = [list() for _ in range(num_crops)]
        channel_list = []
        channel_masks = []
        for sample in samples:
            for idx, (image, _) in enumerate(sample):
                collate_list[idx].append(image)
            channel_list.append([self.channel_map[chn] for chn in sample[0][1]])
            num_chans = image.shape[0]
            channel_masks.append([True if idx < num_chans else False for idx in range(max_chans)])
            
        return [self.collate_crops(crops, max_chans) for crops in collate_list], channel_list, channel_masks

    def collate_fn_simclr(self, samples: list):        
        max_chans = max([sample[0].shape[0] for sample in samples])  # sample[0] is the tensor
        
        images = []
        channel_list = []
        channel_masks = []
        
        for sample in samples:
            # sample is (tensor, channels_tuple)
            image, channels = sample  # Unpack the tuple directly
            
            images.append(image)
            
            # Channel info
            sample_channels = [self.channel_map[chn] for chn in channels]
            channel_list.append(sample_channels)
            
            # Create mask based on actual number of channels
            num_chans = image.shape[0]
            channel_mask = [True if idx < num_chans else False for idx in range(max_chans)]
            channel_masks.append(channel_mask)
        
        # Collate into single batch [B, C, H, W]
        batch = self.collate_crops(images, max_chans)
            
        return batch, channel_list, channel_masks
    
    def load_dataset_config(self, config_path):
        proper_path = os.path.abspath(os.path.expanduser(config_path))
        self.dataset_config = pl.read_csv(proper_path, schema_overrides=OVERRIDES)
        
        if self.config.dataset_filter == "allen":
            self.dataset_config = self.dataset_config.filter(pl.col('storage.path').str.contains('/Allen/'))
        elif self.config.dataset_filter == 'cp':
            self.dataset_config = self.dataset_config.filter(pl.col('storage.path').str.contains('/CP/'))
        elif self.config.dataset_filter == 'hpa':
            self.dataset_config = self.dataset_config.filter(pl.col('storage.path').str.contains('/HPA/'))
        elif self.config.dataset_filter == '10ds':
            self.dataset_config = self.dataset_config.filter(pl.col('experiment.study').is_in(DS10))
        
        channels = self.dataset_config['imaging.channel_type'].unique().to_list()
        aggregated = self.dataset_config.sort('imaging.channel').group_by('imaging.multi_channel_id', maintain_order=True).agg(pl.col('storage.path'), pl.col('imaging.channel_type'))
        return list(zip(aggregated['storage.path'].to_list(), aggregated['imaging.channel_type'].to_list())), channels

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.load_archive()
            
        if self.config.guided_crops_path:
            self.default_transform = v2.RandomResizedCrop(size=self.guided_crops.crop_size, antialias=True)  
        
        worker_data = self.call_splitting_fns(self.image_paths)    
        samples = iter(self.return_sample(worker_data))

        return samples
    
def get_crop_size(dataset):
    # Dataset configurations using exact sizes provided
    if dataset == "wtc0001":
        baseline = random.choice([(256, 256), (450, 450)])
    elif dataset == "jump0001":
        baseline = random.choice([(112, 112), (450, 450)])
    elif dataset == "hpa0018":
        baseline = random.choice([(200, 200), (450, 450)])
    elif dataset == "nidr0031":
        baseline = random.choice([(128, 128), (250, 250)])
    elif dataset == "nidr0032":
        baseline = random.choice([(92, 92), (350, 350)])
    elif dataset == "idr0002":
        baseline = random.choice([(114, 114), (350, 350)])
    elif dataset == "idr0088":
        baseline = random.choice([(114, 114), (350, 350)])
    elif dataset == "idr0086" or dataset == "idr0089":
        baseline = (-1, -1)  # keep as is
    elif dataset == "idr0008":
        baseline = random.choice([(224, 224), (512, 512)])  # already defined
    elif dataset == "idr0001":
        baseline = random.choice([(145, 145), (350, 350)])
    elif dataset == "idr0003":
        baseline = random.choice([(72, 72), (140, 140)])
    elif dataset == "idr0006":
        baseline = random.choice([(150, 150), (300, 300)])
    elif dataset == "idr0005":
        baseline = random.choice([(150, 150), (300, 300)])
    elif dataset == "idr0009":
        baseline = random.choice([(150, 150), (450, 450)])
    elif dataset == "idr0010":
        baseline = random.choice([(128, 128), (300, 300)])
    elif dataset == "idr0011":
        baseline = random.choice([(72, 72), (200, 200)])
    elif dataset == "idr0012":
        baseline = random.choice([(128, 128), (200, 200)])
    elif dataset == "idr0013":
        baseline = random.choice([(48, 48), (200, 200)])
    elif dataset == "idr0017":
        baseline = random.choice([(56, 56), (300, 300)])
    elif dataset == "idr0020":
        baseline = random.choice([(70, 70), (200, 200)])
    elif dataset == "idr0022":
        baseline = random.choice([(120, 120), (600, 600)])
    elif dataset == "idr0028":
        baseline = random.choice([(200, 200), (500, 500)])
    elif dataset == "idr0030":
        baseline = random.choice([(150, 150), (300, 300)])
    elif dataset == "idr0033":
        baseline = random.choice([(150, 150), (350, 350)])
    elif dataset == "idr0035":
        baseline = random.choice([(200, 200), (400, 400)])
    elif dataset == "idr0037":
        baseline = random.choice([(100, 100), (400, 400)])
    elif dataset == "idr0056":
        baseline = random.choice([(75, 75), (300, 300)])
    elif dataset == "idr0069":
        baseline = random.choice([(100, 100), (300, 300)])
    elif dataset == "idr0080":
        baseline = random.choice([(200, 200), (400, 400)])
    elif dataset == "idr0093":
        baseline = random.choice([(100, 100), (400, 400)])
    elif dataset == "idr0094":
        baseline = random.choice([(50, 50), (150, 150)])
    elif dataset == "idr0120":
        baseline = random.choice([(200, 200), (600, 600)])
    elif dataset == "idr0123":
        baseline = random.choice([(200, 200), (400, 400)])
    elif dataset == "idr0128":
        baseline = random.choice([(50, 50), (300, 300)])
    elif dataset == "idr0130":
        baseline = random.choice([(20, 20), (150, 150)])
    elif dataset == "idr0133":
        baseline = random.choice([(200, 200), (400, 400)])
    elif dataset == "idr0140":
        baseline = random.choice([(50, 50), (200, 200)])
    elif dataset == "idr0145":
        baseline = random.choice([(100, 100), (300, 300)])
    elif dataset == "nidr0001":
        baseline = random.choice([(300, 300), (500, 500)])
    elif dataset == "nidr0003":
        baseline = (-1, -1)  # keep as is
    elif dataset == "nidr0004":
        baseline = random.choice([(600, 600), (-1, -1)])
    elif dataset == "nidr0005":
        baseline = (-1, -1)  # keep as is
    elif dataset == "nidr0006":
        baseline = random.choice([(128, 128), (300, 300)])
    elif dataset == "nidr0008":
        baseline = random.choice([(84, 84), (400, 400)])
    elif dataset == "nidr0010":
        baseline = random.choice([(64, 64), (250, 250)])
    elif dataset == "nidr0011":
        baseline = random.choice([(140, 140), (450, 450)])
    elif dataset == "nidr0012":
        baseline = random.choice([(45, 45), (400, 400)])
    elif dataset == "nidr0013":
        baseline = random.choice([(92, 92), (350, 350)])
    elif dataset == "nidr0014":
        baseline = random.choice([(140, 140), (350, 350)])
    elif dataset == "nidr0015":
        baseline = random.choice([(140, 140), (350, 350)])
    elif dataset == "nidr0016":
        baseline = random.choice([(140, 140), (250, 250)])
    elif dataset == "nidr0017":
        baseline = random.choice([(140, 140), (350, 350)])
    elif dataset == "nidr0018":
        baseline = (-1, -1)  # keep as is
    elif dataset == "nidr0019":
        baseline = random.choice([(240, 240), (350, 350)])
    elif dataset == "nidr0020":
        baseline = random.choice([(184, 184), (400, 400)])
    elif dataset == "nidr0021" or dataset == "nidr0022":
        baseline = (-1, -1)  # keep as is
    elif dataset == "nidr0023":
        baseline = random.choice([(184, 184), (400, 400)])
    elif dataset == "nidr0024":
        baseline = (-1, -1)  # keep as is
    elif dataset == "nidr0025":
        baseline = random.choice([(250, 250), (400, 400)])
    elif dataset == "nidr0027":
        baseline = random.choice([(200, 200), (-1, -1)])
    elif dataset == "nidr0028":
        baseline = (-1, -1)  # keep as is - you didn't specify configuration
    elif dataset == "nidr0029":
        baseline = (-1, -1)  # keep as is - you didn't specify configuration
    elif dataset == "nidr0030":
        baseline = (-1, -1)  # keep as is - you didn't specify configuration
    elif dataset == "hpa0023":
        baseline = random.choice([(256, 256), (512, 512)])
    else:
        # Default case for undefined datasets
        baseline = (-1, -1)
    return baseline
