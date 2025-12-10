import os
import safetensors.torch
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import safetensors
import sys
import sys
target_path = os.path.abspath("../")
sys.path.append(target_path)
from models import get_model, Model
from idr0017_dataloader import FeatureExtractionDataset

def process_image_patches(model: Model, batch):
    """
    Processes a batch of patches and extracts features using a given model.
    
    Parameters:
    model: The model used for feature extraction.
    batch (torch.Tensor) [B, C, H, W]: A batch of patches.
    device (torch.device): The device to run the model on (CPU or GPU).
    """
    if len(batch) == 0:
        return np.array([])
    
    device = "cuda"
    if batch.device != device:
        batch = batch.to(device)

    minibatches = torch.split(batch, args.batch_size, dim=0)
        
    features = None  # Initialize as None
    for minibatch in minibatches:

        if minibatch.dtype == torch.uint8:
            minibatch = minibatch.float() / 255.0  # Normalize to [0,1] range

        minibatch_features = model(minibatch)
        
        if features is None:
            # First iteration - initialize the features tensor
            features = minibatch_features
        else:
            # Subsequent iterations - concatenate
            features = torch.cat([features, minibatch_features], dim=0)

    return features
 

def get_features(model:Model, args: dict, dataset:Dataset, out_folder:str) -> None:    
    
    dataloader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers)
    
    current_plate = None
    image_st = {}
    
    for patches, image_paths, multi_channel_id, plate_name in tqdm(dataloader, desc="Extracting Features", total=len(dataloader)):
        plate_name = plate_name[0]  # Unwrap from batch
        
        # Detect plate change - save previous plate's features
        if current_plate is not None and plate_name != current_plate:
            output_file = os.path.join(out_folder, f"{current_plate}_features.safetensors")
            safetensors.torch.save_file(image_st, output_file)
            print(f"Saved {len(image_st)} features for {current_plate}")
            image_st = {}  # Reset for new plate
        
        current_plate = plate_name
        
        # Process features
        patches = patches.squeeze(0).to("cuda")
        features = process_image_patches(model, patches)
        features = torch.tensor(features).contiguous()
        

        features = features.mean(dim=0)
        
        image_st[multi_channel_id[0]] = features
    
    # Save the last plate
    if current_plate is not None and len(image_st) > 0:
        output_file = os.path.join(out_folder, f"{current_plate}_features.safetensors")
        safetensors.torch.save_file(image_st, output_file)
        print(f"Saved {len(image_st)} features for {current_plate}")


def main(args, image_folder:str , output_folder: str):
    plate = os.path.basename(image_folder)


    model_name = args.model_type
    model_path = args.model_path

    model_instance = get_model(model_name=model_name, model_path=model_path)
    
    if model_name == "channelvit":
        model_instance.set_dataset('idr17', model_path)
    
    model = model_instance
    _, transforms = model_instance.get_model()

    # Do this after I get my transform from the model instance
    dataset = FeatureExtractionDataset(image_folder, transforms=transforms)

    if model is None:
        raise AttributeError("Model is None.")
        
    model.to("cuda")
    get_features(model=model, args=args, dataset=dataset, out_folder = output_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Get features from a given model")    
    parser.add_argument("-s", "--images_folder", type=str, help='Unzipped plate path. Input to segment in snakemake')
    parser.add_argument("-o", "--output_folder", type=str, help='CSV path. Output of segement in snakemake.')
    parser.add_argument('-d', '--device', type=int, help="Cuda device for torch to set to, None is CPU")
    parser.add_argument('--model_path', type=str, help= "Model Path written in this argument")
    parser.add_argument('--model_type', type=str, help = "Tells us what type of model given")
    parser.add_argument('--num_workers', type=int, help="Tells us how many workers helping in feature extraction")
    parser.add_argument('--batch_size', default=4, type=int, help= "Setting the batch size for processing image batch sizes")
    
    args = parser.parse_args()
    
    main(args, args.images_folder, args.output_folder)
