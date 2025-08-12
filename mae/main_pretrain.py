# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
from PIL import Image
import torch
import torch.nn as nn
import datetime
import json
import numpy as np
import os
import time
import wandb
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
#from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import sys
from timm.optim import create_optimizer_v2
from timm.optim.optim_factory import param_groups_weight_decay
sys.path.append('../')
from dataset.dataset import IterableImageArchive
from dataset import dataset_config
from dataset.dataset_functions import randomize, split_for_workers, get_proc_split
from torch.utils.data import DataLoader
from torchvision.transforms import v2
sys.path.append('utils')

import timm
#assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae

from engine_pretrain import train_one_epoch



class PerImageNormalize(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        # We initialize with num_features=1, but we’ll replace it on-the-fly if needed.
        self.eps = eps
        self.instance_norm = nn.InstanceNorm2d(
            num_features=1,             # Temporary placeholder
            affine=False,               # No learnable parameters
            track_running_stats=False,  # Use per-forward stats (no running mean)
            eps=self.eps
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (N, C, H, W)
        We'll ensure that our instance_norm has the correct number of channels (C).
        """
        # If your input has a dynamic channel size, we need to re-initialize:
        C, _, _ = x.shape
        if self.instance_norm.num_features != C:
            self.instance_norm = nn.InstanceNorm2d(
                num_features=C,
                affine=False,
                track_running_stats=False,
                eps=self.eps
            )

        # Now we can pass x through our InstanceNorm2d layer
        return self.instance_norm(x)
    

class SaturationNoiseInjector(nn.Module):
    def __init__(self, low=200, high=255):
        """
        Initialize the SaturationNoiseInjector module.
        
        Parameters:
            low (int): Lower bound for uniform noise values.
            high (int): Upper bound for uniform noise values.
        """
        super().__init__()
        self.low = low
        self.high = high

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply high-intensity noise injection to saturated pixels in a single-channel image.
        The function expects the input tensor to have the shape (1, H, W) with pixel intensities in the 0-255 range.

        Process:
          - Convert the input tensor to float32.
          - Generate noise drawn uniformly from [low, high] for each pixel.
          - Create a mask for saturated pixels (where the pixel value equals 255).
          - Zero-out saturated pixels and add the masked noise.

        Parameters:
            x (torch.Tensor): Input tensor of shape (1, H, W).
        
        Returns:
            torch.Tensor: The processed tensor with noise injected.
        """
        # Ensure input is in floating point for correct arithmetic
        
        # Since x has one channel, extract the channel as a 2D tensor (H, W)
        channel = x[0]
        
        # Generate noise with values uniformly drawn between self.low and self.high
        noise = torch.empty_like(channel).uniform_(self.low, self.high)
        
        # Create a mask of pixels that are saturated (value == 255)
        mask = (channel == 255).float()
        
        # Apply the mask to the noise to affect only the saturated pixels
        noise_masked = noise * mask
        
        # Remove the saturated pixels by setting them to zero
        channel[channel == 255] = 0
        
        # Add the masked noise to the channel
        channel = channel + noise_masked
        
        # Update the tensor with the modified channel
        x[0] = channel
        
        return x


class TensorAugmentationDINO(object):
    def __init__(self):
        flips = transforms.Compose(
            [
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5)
            ]
        )
        self.common_normalization = transforms.Compose([
            v2.RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(0.9, 1.1), antialias=True),
            v2.ToImageTensor(),
            SaturationNoiseInjector(low=200, high=255),
            PerImageNormalize(),
            flips
        ])

    def __call__(self, image):
        image = self.common_normalization(image)
        return image
    



def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_small_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=5e-5, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/scr/vidit/chammi_train.zip', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true',
                        help='automatically resume from the latest checkpoint in output_dir')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument(
    '--local_rank', '--local-rank',
    dest='local_rank',
    default=-1, type=int,
    help='(alias: --local-rank) this process’s GPU-local rank.'
    )
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--use_fp32', action='store_true', default=True,
                        help='Use float32 for images')
    parser.add_argument('--guided_cropping', default=False, type=bool,
                        help='Use guided cropping for training. If true, guided_crops_path and guided_crops_size must be set.')
    parser.add_argument('--guided_crops_size', default=(256, 256), type=int, nargs=2,
                        help='Size of the guided crops to use for training. If None, no guided cropping is used.')
    parser.add_argument('--guided_crops_path', default=None, type=str,)
    parser.add_argument('--multiscale', default=False, type=bool,)
    parser.add_argument('--dataset_size', default="small", type=str, choices=["small", "large"],)
    parser.add_argument('--save_freq', default=20, type=int,
                        help='frequency of saving checkpoints (in epochs)')
    parser.add_argument('--keep_checkpoints', default=5, type=int,
                        help='number of recent checkpoints to keep (0 to keep all)')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    if args.guided_cropping:
        config = dataset_config.DatasetConfig(
                args.data_path, # args.data_path, /scr/data/CHAMMIv2m.zip
                split_fns=[get_proc_split, randomize, split_for_workers],
                num_procs = misc.get_world_size(), # maybe works? brother needs to check!
                proc = misc.get_rank(), # This is the global rank generally? Print out later? Look at multinode?
                use_fp32 = args.use_fp32,  # Use float32 for images
                guided_crops_path = args.guided_crops_path,
                guided_crops_size = args.guided_crops_size,
                transform=TensorAugmentationDINO(),
                dataset_size=args.dataset_size,
                seed=seed,
                )
    else:
        config = dataset_config.DatasetConfig(
                args.data_path, # args.data_path, /scr/data/CHAMMIv2m.zip
                split_fns=[get_proc_split, randomize, split_for_workers],
                num_procs = misc.get_world_size(), # maybe works? brother needs to check!
                proc = misc.get_rank(), # This is the global rank generally? Print out later? Look at multinode?
                use_fp32 = args.use_fp32,  # Use float32 for images
                transform=TensorAugmentationDINO(),
                dataset_size=args.dataset_size,
                seed=seed,
                )
    dataset_train = IterableImageArchive(config)


    world_size = misc.get_world_size()
    rank       = misc.get_rank()
    if world_size > 1:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
    else:
        sampler_train = None

    print("Sampler_train = %s" % str(sampler_train))

   # if global_rank == 0 and args.log_dir is not None:
        #os.makedirs(args.log_dir, exist_ok=True)
        #log_writer = SummaryWriter(log_dir=args.log_dir)
   #else:
   #     log_writer = None
    data_loader_train = torch.utils.data.DataLoader(
        dataset = dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        persistent_workers=True,
        worker_init_fn=dataset_train.worker_init_fn
    )
    
    # define the model
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = param_groups_weight_decay(model, args.weight_decay)
    optimizer    = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    # Check for existing checkpoints in output directory and auto-resume from latest
    if args.output_dir and not args.resume and args.auto_resume:
        latest_checkpoint = misc.find_latest_checkpoint(args.output_dir)
        if latest_checkpoint:
            args.resume = latest_checkpoint
            print(f"Found existing checkpoint: {latest_checkpoint}. Auto-resuming from it.")

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Checkpoint settings:")
    print(f"  - Auto-resume: {args.auto_resume}")
    print(f"  - Save frequency: every {args.save_freq} epochs")
    print(f"  - Keep checkpoints: {args.keep_checkpoints if args.keep_checkpoints > 0 else 'all'}")
    print(f"  - Output directory: {args.output_dir}")

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if sampler_train is not None:
            sampler_train.set_epoch(epoch)
        #if args.distributed and data_loader_train.sampler is not None:
            #data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler, 
            args=args
        )
        if args.output_dir and (epoch % args.save_freq == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
