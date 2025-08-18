from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum

class Arch(str, Enum):
    channelvit_small = "channelvit_small"
    channelvit_base = "channelvit_base"
    channelvit_large = "channelvit_large"

@dataclass
class ModelConfig:
    """Model parameters"""
    arch: Arch = Arch.channelvit_small
    patch_size: int = 16
    out_dim: int = 65536
    norm_last_layer: bool = True
    momentum_teacher: float = 0.996
    use_bn_in_head: bool = False

@dataclass
class TemperatureConfig:
    """Temperature teacher parameters"""
    warmup_teacher_temp: float = 0.04
    teacher_temp: float = 0.07
    warmup_teacher_temp_epochs: int = 30

@dataclass
class OptimConfig:
    """Training/Optimization parameters"""
    use_fp16: bool = True
    weight_decay: float = 0.04
    weight_decay_end: float = 0.4
    clip_grad: float = 3.0
    batch_size_per_gpu: int = 32
    epochs: int = 100
    freeze_last_layer: int = 1
    lr: float = 0.0005
    warmup_epochs: int = 10
    min_lr: float = 1e-6
    optimizer: str = 'adamw' 
    drop_path_rate: float = 0.1

@dataclass
class MultiCropConfig:
    """Multi-crop parameters"""
    global_crops_scale: List[float] = field(default_factory=lambda: [0.4, 1.0])
    local_crops_number: int = 8
    local_crops_scale: List[float] = field(default_factory=lambda: [0.05, 0.4])

class DatasetSize(str, Enum):
    small = "small"
    large = "large"

@dataclass
class DatasetConfig:
    """Dataset parameters"""
    guided_crops_path: Optional[str] = None
    dataset_size: DatasetSize = DatasetSize.large
    multiscale: bool = False
    guided_cropping: bool = False
    guided_crops_size: Tuple[int, int] = (256, 256)
    small_list_path: Optional[str] = None
    metadata: str = '../../multi_channel_chammi_metadata.csv'

@dataclass
class TrainConfig:
    """Misc parameters"""
    name: str = ""
    data_path: str = '../../chammi_train.zip'
    output_dir: str = "/hdd/jcaicedo/projects/channel_vit_dinov1/models"
    saveckp_freq: int = 20
    seed: int = 42
    num_workers: int = 10
    dist_url: str = "env://"
    local_rank: int = 0
    world_size: int = 0
    gpu: int = 0
    rank: int = 0

@dataclass
class DINOV1Config:
    """Main configuration class"""
    model: ModelConfig = field(default_factory=ModelConfig)
    temp: TemperatureConfig = field(default_factory=TemperatureConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    crops: MultiCropConfig = field(default_factory=MultiCropConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    train: TrainConfig = field(default_factory=TrainConfig)