import argparse
import os
import subprocess
import numpy as np
import sys
from omegaconf import OmegaConf
sys.path.append('../../')

from config import *

default_path = '/home/jgpeters3/configs'

def submit_command(command:str):
    result = subprocess.run(
                    command,
                    shell=True,
                    # capture_output=True,
                    check=True,
                    text=True
    )
    
def run_command(config: DINOV1Config):
    conf = OmegaConf.create(config)
    OmegaConf.save(config=conf, f=os.path.join(default_path, f'{config.train.name}.yaml'))
    
    command = ' '.join([
                        "condor_submit", 
                        f"wandb_key={os.environ.get('WANDB_API_KEY')}",
                        f'config_name={config.train.name}.yaml',
                        f"config_path={os.path.join(default_path, f'{config.train.name}.yaml')}",
                        "chtc_job.sh"
                    ])
    submit_command(command=command)

def main():    
    config = DINOV1Config()
    config.train.name = "e51716f_75ds_base"
    
    config.dataset.metadata = '/hdd/jcaicedo/morphem/dataset/sampling/CHAMMI-75_small_metadata.csv'
    config.dataset.guided_cropping = True
    config.dataset.guided_crops_path = '/scratch/CHAMMI-75_guidance.zip'
    config.model.arch = Arch.channelvit_base
    config.optim.batch_size_per_gpu = 7
    
    config.train.data_path = '/scratch/CHAMMI-75_small.zip' 
    
    run_command(config)
    
    # config = DINOV1Config()
    # config.train.name = "e51716f_10ds"
    
    # config.dataset.TEMP_DATASET = "10ds"
    # config.dataset.metadata = '/hdd/jcaicedo/morphem/dataset/sampling/CHAMMI-75_small_metadata.csv'
    # config.dataset.guided_cropping = True
    # config.dataset.guided_crops_path = '/scratch/CHAMMI-75_guidance.zip'
    
    # config.train.data_path = '/scratch/CHAMMI-75_small.zip' 
    
    # run_command(config)
    
    # config = DINOV1Config()
    # config.train.name = "e51716f_75ds"
    
    # config.dataset.metadata = '/hdd/jcaicedo/morphem/dataset/sampling/CHAMMI-75_small_metadata.csv'
    # config.dataset.guided_cropping = True
    # config.dataset.guided_crops_path = '/scratch/CHAMMI-75_guidance.zip'
    
    # config.train.data_path = '/scratch/CHAMMI-75_small.zip' 
    
    # run_command(config)
    
if __name__ == "__main__":
    main()