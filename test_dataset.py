import torch
from dataset.dataset_config import DatasetConfig
from dataset.dataset import MultiChannelDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.transforms import v2
import accelerate

def main():
    accelerator = accelerate.Accelerator()
    default_transform = v2.RandomResizedCrop(size=56, antialias=True)
    config = DatasetConfig(
                           data_path='~/scratch/CHAMMI-75_small.zip', 
                           dataset_config='~/scratch/CHAMMI-75_small_metadata.csv', 
                           transform=default_transform,
                           num_procs=accelerator.num_processes,
                           proc=accelerator.local_process_index,
                           guided_crops_path='~/scratch/CHAMMI-75_guidance.zip',
                           guided_crops_size=(56,56)
                           )
    
    dataset = MultiChannelDataset(config)
    dl = DataLoader(dataset, batch_size=1, num_workers=1, worker_init_fn=dataset.worker_init_fn, collate_fn=dataset.collate_fn)
    for batch in tqdm(dl):
    # for batch in dl:
        continue

if __name__ == "__main__":
    main()