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

disable_beta_transforms_warning()

class IterableImageArchive(IterableDataset):
    def __init__(self, config: DatasetConfig) -> None:
        super().__init__()

        self.config = config
        self.archive = None 
        self.image_paths = None 
        self.metadata_filename = None
        self.file_order = None
        self.guided_crops: GuidedCrop = None
        self.default_transform = None
        self.metadata_df = None

    def load_archive(self):
        self.archive = zipfile.ZipFile(self.config.data_path, "r")
        #if self.config.dataset_size == "large":
        #    self.metadata_filename = "75ds_large_metadata.csv"
        #elif self.config.dataset_size == "small":
        #    self.metadata_filename = "75ds_small_metadata.csv"

        #with zipfile.ZipFile(os.path.join(self.config.data_path, self.metadata_filename), "r") as metadata_zip:
            #metadata = metadata_zip.read(self.metadata_filename).decode("utf-8")
            #self.metadata_df = pd.read_csv(StringIO(metadata))
        self.image_paths = [file for file in self.archive.infolist() 
                        if not file.is_dir() and file.filename.endswith(self.config.img_type)]

    def return_sample(self, file_list: list):
        try:
            for file_path in file_list:
                img_bytes = bytearray(self.archive.read(file_path.filename))
                torch_buffer = torch.frombuffer(img_bytes, dtype=torch.uint8)
                image_tensor = decode_image(torch_buffer)
                image_tensor = image_tensor.to(torch.float16)

                


                '''
                dataset = file_path.filename.split(os.sep)[1]
                # Dataset configurations using exact sizes provided
                if dataset == "wtc0001":
                    baseline = random.choice([(256, 256), (450, 450)])
                elif dataset == "jump0001":
                    baseline = random.choice([(112, 112), (450, 450)])
                elif dataset == "hpa0001":
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
                elif dataset == "nidr0026":
                    baseline = (-1, -1)  # keep as is
                elif dataset == "nidr0027":
                    baseline = random.choice([(200, 200), (-1, -1)])
                elif dataset == "nidr0028":
                    baseline = (-1, -1)  # keep as is - you didn't specify configuration
                elif dataset == "nidr0029":
                    baseline = (-1, -1)  # keep as is - you didn't specify configuration
                elif dataset == "nidr0030":
                    baseline = (-1, -1)  # keep as is - you didn't specify configuration
                else:
                    # Default case for undefined datasets
                    baseline = (-1, -1)

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
                    if safetensors_name in self.guided_crops.data_paths:
                        image_tensor = self.guided_crops(image_tensor, safetensors_name)
                    else:
                        pass
                else:
                    pass
                '''

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

        if self.config.test:
            self.file_order = [file.filename for file in worker_data]

        return samples
    
    def __len__(self):
        if not self.image_paths:
            archive = zipfile.ZipFile(self.config.data_path, "r")
            image_paths = [file for file in archive.infolist() 
                            if not file.is_dir() and file.filename.endswith(self.config.img_type)] 
            self.image_paths = image_paths

        if self.config.num_procs > 1:
            return len(get_proc_split(self.image_paths, self.config))
        else:
            return len(self.image_paths)
