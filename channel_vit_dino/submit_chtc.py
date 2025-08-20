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
    config.train.name = "eea860c_baseline"
    run_command(config)
    
    
if __name__ == "__main__":
    main()