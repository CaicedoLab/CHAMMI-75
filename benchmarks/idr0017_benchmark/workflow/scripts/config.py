import yaml
import torch
from torchvision import disable_beta_transforms_warning
disable_beta_transforms_warning()

def _merge(src, dst):
    for k, v in src.items():
        if k in dst:
            if isinstance(v, dict):
                _merge(src[k], dst[k])
        else:
            dst[k] = v


def load_config(config_path:dict, device:int):
    with open(config_path, "r") as cfg:
        config = yaml.full_load(cfg)
    
    if device is not None:    
        config['device'] = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
    else:
        config['device'] = torch.device('cpu')
    return config
