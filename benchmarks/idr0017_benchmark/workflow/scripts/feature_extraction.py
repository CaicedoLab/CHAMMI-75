import os
import safetensors.torch
import torch
import argparse
import numpy as np
from tqdm import tqdm
from config import load_config
from torch.utils.data import DataLoader, Dataset
import safetensors
import sys
import sys
target_path = os.path.abspath("../../")
sys.path.append(target_path)
from models import get_model, Model
from data.dataset.dataset import FeatureExtractionDataset

def process_image_patches(model: Model, batch, cfg):
    """
    Processes a batch of patches and extracts features using a given model.
    
    Parameters:
    model: The model used for feature extraction.
    batch (torch.Tensor) [B, C, H, W]: A batch of patches.
    device (torch.device): The device to run the model on (CPU or GPU).
    """
    # TODO: return some empty safetensors or something, not a numpy array
    if len(batch) == 0:
        return np.array([])
    
    device = cfg['device']
    if batch.device != device:
        batch = batch.to(device)

    minibatches = torch.split(batch, cfg['feature_extraction']['resources']['batch_size'], dim=0)
        
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
 

def get_features(model:Model, cfg: dict, dataset:Dataset, out_folder:str, out_name=str) -> None:    
    feature_config = cfg['feature_extraction']
    feature_resources = feature_config['resources']
    
    # Batch size is 1 because we want to process 1 image at a time, but we are using a 
    # DataLoader so that when we are in the loop, we can async load the patches and whatnot
    # for the next images. Bicc brain
    dataloader = DataLoader(dataset, batch_size=1, num_workers=feature_resources['num_workers']) # Add the transform
    
    image_st = {}
    for patches, image_paths, multi_channel_id in tqdm(dataloader, desc=f"Getting {feature_config['model']} features", total=len(dataloader)):
        patches:torch.Tensor = patches.squeeze(0).to(cfg['device'])
        
        features = process_image_patches(model, patches, cfg)

        features = torch.tensor(features)
        features = features.contiguous()                            

        if feature_config['feature_agg']:
            features = features.mean(dim=0)
            
        image_st[multi_channel_id[0]] = features
    
    output_file = os.path.join(out_folder, f"{out_name}_features.safetensors")
    safetensors.torch.save_file(image_st, output_file)


def main(cfg, snake_in:str , snake_out: str, snake_model:str):
    plate = os.path.basename(snake_in)


    model_name = cfg['feature_extraction']['model'].lower()
    model_mode = cfg['feature_extraction']['model_mode']
    model_path = cfg['feature_extraction']['model_path']
    

    model_instance = get_model(model_name=model_name, model_path=model_path, model_type=model_mode)
    model = model_instance
    _, transforms = model_instance.get_model()

    # Do this after I get my transform from the model instance
    dataset = FeatureExtractionDataset(cfg, snake_in, transform=transforms)

    if model is None:
        raise AttributeError(f"Config {cfg['feature_extraction']['model']['model_mode']} and {cfg['feature_extraction']['model']} are not a valid model_mode, model config (misspelled?).")
        
    model.to(cfg['device'])
    get_features(model=model, cfg=cfg, dataset=dataset, out_folder = snake_out, out_name = plate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Get features from a given model")
    parser.add_argument("-c", "--config", type=str, metavar="DIR", help="path to a config file", default=None, required=True)
    
    parser.add_argument("-s", "--snake_in", type=str, help='Unzipped plate path. Input to segment in snakemake')
    parser.add_argument("-o", "--snake_out", type=str, help='CSV path. Output of segement in snakemake.')
    parser.add_argument("-m", "--snake_model", type=str, help='Torch set directory to pull models from')
    parser.add_argument('-d', '--device', type=int, help="Cuda device for torch to set to, None is CPU")
    
    args = parser.parse_args()
    if args.snake_in and not args.snake_out or args.snake_out and not args.snake_in:
        parser.error("--snake_in and --snake_out must both be given.")
    
    cfg = load_config(args.config, args.device)
    main(cfg, args.snake_in, args.snake_out, args.snake_model)
