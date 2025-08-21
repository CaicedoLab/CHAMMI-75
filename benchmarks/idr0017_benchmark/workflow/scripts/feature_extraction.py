import os
import safetensors.torch
import torch
import argparse
import numpy as np
from tqdm import tqdm
from config import load_config
from torch.utils.data import DataLoader, Dataset
from models import Model
import safetensors
from models import *
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
        
    features = []
    for minibatch in minibatches:
        features.append(model(minibatch))

    return torch.cat(features, dim=0) 

 

def get_features(model:Model, cfg: dict, dataset:Dataset, out_folder:str, out_name=str) -> None:    
    feature_config = cfg['feature_extraction']
    feature_resources = feature_config['resources']
    
    # Batch size is 1 because we want to process 1 image at a time, but we are using a 
    # DataLoader so that when we are in the loop, we can async load the patches and whatnot
    # for the next images. Bicc brain
    dataloader = DataLoader(dataset, batch_size=1, num_workers=feature_resources['num_workers'])
    
    image_st = {}
    for patches, image_paths, multi_channel_id in tqdm(dataloader, desc=f"Getting {feature_config['model']} features", total=len(dataloader)):
        patches:torch.Tensor = patches.squeeze(0).to(cfg['device'])
        
        features = process_image_patches(model, patches, cfg).detach().cpu()
        if features.shape[1] > 1:    
            # this is for single channel models
            for image_path, image_idx in zip(image_paths, range(features.shape[1])):
                single_image_features = features[:,image_idx,:]
                if feature_config['feature_agg']:
                    single_image_features = single_image_features.mean(dim=0)
                image_st[os.path.basename(image_path[0])] = single_image_features.contiguous()
        else:
            # this is for channel adaptive models
            features = features.squeeze(1).contiguous()
            if feature_config['feature_agg']:
                features = features.mean(dim=0)
                
            image_st[multi_channel_id[0]] = features
    
    output_file = os.path.join(out_folder, f"{out_name}_features.safetensors")
    safetensors.torch.save_file(image_st, output_file)


def main(cfg, snake_in:str , snake_out: str, snake_model:str):
    os.environ['TORCH_HOME'] = cfg['models']
    plate = os.path.basename(snake_in)
    dataset = FeatureExtractionDataset(cfg, snake_in)

    model_name = cfg['feature_extraction']['model'].lower()
    model_mode = cfg['feature_extraction']['model_mode']
    if model_mode is not None:
        model_mode = model_mode.lower()
    
    model = None
    if model_name == "dinov2":
        model = DinoV2(cfg)                
    elif model_name == "openphenom":
        if model_mode == 'agg':
            model = OpenPhenom(cfg, 'agg')
        elif model_mode == 'conc':
            model = OpenPhenom(cfg, 'conc')
    elif model_name == "subcell":
        if model_mode == 'mae':
            model = SubCell(cfg, "MAE")
        elif model_mode == 'vit':
            model = SubCell(cfg, "VIT")

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
