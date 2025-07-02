from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .dataset_config import DatasetConfig

import typing
import random
from torch.utils.data import get_worker_info

def split_for_workers(data: list, config: DatasetConfig): 
    worker_info = get_worker_info()

    return data[worker_info.id::worker_info.num_workers]

def get_proc_split(data: list, config: DatasetConfig):
    return data[config.proc::config.num_procs]

def randomize(data: list, config: DatasetConfig):
    assert len(data) != 0, "No data was given to randomize"

    if config.seed is not None:
        random.seed(config.seed)
    
    if isinstance(data[0], list):
        for sub_data in data:
            random.shuffle(sub_data)
    else:
        random.shuffle(data)
    
    return data

FUNCTIONS = {
    "split_for_workers": split_for_workers,
    "randomize": randomize,
    "get_proc_split": get_proc_split   
}