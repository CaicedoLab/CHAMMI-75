'''
Main SimCLR training script with original data loader

'''


import random
import os
from re import match
import sys
import datetime
import time
import math
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
from torchvision.transforms.v2 import Transform
import sys
sys.path.append("../../")
from dataset.dataset import IterableImageArchive
from dataset import dataset_config
from dataset.dataset_functions import randomize, split_for_workers, get_proc_split
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import distributed_utils
import argparse
import yaml
from multi_channel_vit import get_multi_channel_vit
from torch.nn.parallel import DistributedDataParallel as DDP
from optimizer import get_optimizer
from diffusers.optimization import get_scheduler
import glob


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
        return self.instance_norm(x).to(torch.float16)
    

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
        x = x.to(torch.float32)
        
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


def find_latest_checkpoint(output_dir):
    """
    Find the latest checkpoint in the output directory.
    
    Args:
        output_dir (str): Directory to search for checkpoints
        
    Returns:
        str or None: Path to the latest checkpoint file, or None if no checkpoints found
    """
    checkpoint_pattern = os.path.join(output_dir, "checkpoint_epoch_*.pt")
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    if not checkpoint_files:
        return None
    
    # Extract epoch numbers and find the latest one
    epoch_numbers = []
    for checkpoint_file in checkpoint_files:
        try:
            # Extract epoch number from filename like "checkpoint_epoch_5.pt"
            filename = os.path.basename(checkpoint_file)
            epoch_num = int(filename.split('_')[-1].split('.')[0])
            epoch_numbers.append((epoch_num, checkpoint_file))
        except (ValueError, IndexError):
            continue
    
    if epoch_numbers:
        # Sort by epoch number and return the latest checkpoint
        latest_checkpoint = max(epoch_numbers, key=lambda x: x[0])[1]
        return latest_checkpoint
    
    return None


def load_checkpoint(checkpoint_path, model, optimizer, lr_scheduler, device):
    """
    Load checkpoint and return the epoch and global_step to resume from.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        model: The model to load state into
        optimizer: The optimizer to load state into
        lr_scheduler: The learning rate scheduler to load state into
        device: Device to load the checkpoint on
        
    Returns:
        tuple: (start_epoch, global_step, last_loss)
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch']
    global_step = checkpoint.get('global_step', 0)
    last_loss = checkpoint.get('loss', 0.0)
    
    print(f"Resumed from epoch {start_epoch}, global step {global_step}, last loss: {last_loss:.4f}")
    
    return start_epoch, global_step, last_loss


def check_and_handle_existing_checkpoints(output_dir, args):
    """
    Check for existing checkpoints and handle them based on user preference.
    
    Args:
        output_dir (str): Directory to check for checkpoints
        args: Training arguments
        
    Returns:
        tuple: (should_resume, checkpoint_path)
    """
    latest_checkpoint = find_latest_checkpoint(output_dir)
    
    if latest_checkpoint is None:
        print(f"No existing checkpoints found in {output_dir}. Starting fresh training.")
        return False, None
    
    print(f"Found existing checkpoint: {latest_checkpoint}")
    
    # Add a command line argument to control this behavior
    if hasattr(args, 'resume') and args.resume:
        print("Resuming training from the latest checkpoint.")
        return True, latest_checkpoint
    elif hasattr(args, 'overwrite') and args.overwrite:
        print("Overwriting existing checkpoints and starting fresh training.")
        return False, None
    else:
        # Interactive prompt (only on main process to avoid multiple prompts)
        if args.gpu == 0:
            while True:
                choice = input("Do you want to (r)esume from checkpoint, (o)verwrite and start fresh, or (a)bort? [r/o/a]: ").strip().lower()
                if choice in ['r', 'resume']:
                    should_resume = True
                    break
                elif choice in ['o', 'overwrite']:
                    should_resume = False
                    break
                elif choice in ['a', 'abort']:
                    print("Training aborted by user.")
                    sys.exit(0)
                else:
                    print("Invalid choice. Please enter 'r', 'o', or 'a'.")
            
            # In distributed training, broadcast the decision to other processes
            if dist.is_initialized():
                # Create a tensor to broadcast the decision
                decision_tensor = torch.tensor([1 if should_resume else 0], dtype=torch.int, device=args.gpu)
                dist.broadcast(decision_tensor, src=0)
        else:
            # Non-main processes wait for the decision
            if dist.is_initialized():
                decision_tensor = torch.tensor([0], dtype=torch.int, device=args.gpu)
                dist.broadcast(decision_tensor, src=0)
                should_resume = bool(decision_tensor.item())
            else:
                should_resume = False
        
        return should_resume, latest_checkpoint if should_resume else None




def get_args_parser():
    parser = argparse.ArgumentParser('SimCLR', add_help=False)

    parser.add_argument('--seed', default=0, type=int, help='Random seed.')

    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")

    parser.add_argument("--local-rank", default=0, type=int, help="Please ignore and do not set this argument.")

    parser.add_argument('--data_path', default='/scr/vidit/chammi_train.zip', type=str, help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir', type=str, help='path where to save, empty for no saving')

    parser.add_argument('--lr', default=5e-5, type=float, help='learning rate')

    parser.add_argument('--warmup_epoch', default=10, type=int, help='number of warmup epochs')

    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='number of gradient accumulation steps')

    parser.add_argument('--batch_size_per_gpu', default=256, type=int, help='Per-GPU batch-size')

    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')

    parser.add_argument('--num_workers', default=6, type=int, help='number of data loading workers per GPU')

    parser.add_argument('--dataset_size', default='small', type=str, help='Size of the dataset to use: small/medium/full')

    parser.add_argument('--guided_cropping', default=False, type=bool, help='Whether to use guided cropping based on segmentation masks')

    parser.add_argument('--guided_crops_path', default='/scr/vidit/chammi_segmentations.zip', type=str, help='Path to the guided crops segmentation masks zip file')

    parser.add_argument('--guided_crops_size', default=(256, 256), type=int, nargs=2,
        help="""Size of the guided crops. Only used if --guided_cropping is True.
        Should be a tuple of two integers (height, width).""")

    parser.add_argument('--multiscale', default=False, type=bool, help='Whether to use multiscale training')

   # Checkpoint handling arguments
    parser.add_argument('--resume', action='store_true', help='Automatically resume from the latest checkpoint if available')
    
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing checkpoints and start fresh training')
    return parser



class SimCLRBatchTransform(object):
    """
    Simple SimCLR transform to apply in your training loop.
    Takes a batch [B, C, H, W] and returns [2*B, C, H, W] with SimCLR ordering.
    """
    
    def __init__(self, image_size=(224, 224), kernel_size=11):
        """
        Args:
            image_size (tuple): Target image size (height, width)
            kernel_size (int): Kernel size for Gaussian blur
        """
        self.image_size = image_size
        self.kernel_size = kernel_size
        
        # Create augmentation pipeline
        self.augmentation_pipeline = v2.Compose([
            v2.RandomResizedCrop(
                size=self.image_size, 
                scale=(0.2, 1.0),
                interpolation=v2.InterpolationMode.BICUBIC,
                antialias=True
            ),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomApply([v2.GaussianBlur(kernel_size=self.kernel_size, sigma=(0.1, 2.0))], p=0.5),
        ])
    
    def __call__(self, batch):
        """
        Apply SimCLR transformations to a batch.
        
        Args:
            batch (torch.Tensor): Input batch [B, C, H, W]
            
        Returns:
            torch.Tensor: Output batch [2*B, C, H, W] ordered as:
                         [img1_view1, img2_view1, ..., img1_view2, img2_view2, ...]
        """
        # Normalize to [0, 1] if input is uint8
        if batch.dtype == torch.uint8:
            batch = batch.float() / 255.0
        elif batch.dtype == torch.float16:
            batch = batch.float()  # Convert float16 to float32
        
        batch_size = batch.shape[0]
        
        # Generate first views
        view1_list = []
        for i in range(batch_size):
            view1 = self.augmentation_pipeline(batch[i])
            view1_list.append(view1)
        
        # Generate second views
        view2_list = []
        for i in range(batch_size):
            view2 = self.augmentation_pipeline(batch[i])
            view2_list.append(view2)
        
        # Stack in SimCLR order: all view1s first, then all view2s
        all_views = view1_list + view2_list
        return torch.stack(all_views, dim=0)

def train_simclr(args):
    distributed_utils.init_distributed_mode(args)
    distributed_utils.fix_random_seeds(args.seed)

    # Check for existing checkpoints and handle them
    should_resume, checkpoint_path = check_and_handle_existing_checkpoints(args.output_dir, args)

    config = dataset_config.DatasetConfig(
                args.data_path, # args.data_path, /scr/data/CHAMMIv2m.zip
                split_fns=[get_proc_split, randomize, split_for_workers],
                num_procs = distributed_utils.get_world_size(), # maybe works? brother needs to check!
                proc = torch.distributed.get_rank(), # This is the global rank generally? Print out later? Look at multinode?
                transform=transforms.Compose([SaturationNoiseInjector(low=200, high=255), PerImageNormalize(), v2.Resize((224,224))]),
                dataset_size=args.dataset_size,
                seed=42,
                use_fp32=True
        )
    
    # If guided cropping is enabled, we add the guided crops path and size to the config
    if args.guided_cropping:
        print("WHY IN GUIDANCE?")
        config = dataset_config.DatasetConfig(
                args.data_path, # args.data_path, /scr/data/CHAMMIv2m.zip
                split_fns=[get_proc_split, randomize, split_for_workers],
                num_procs = distributed_utils.get_world_size(), # maybe works? brother needs to check!
                proc = torch.distributed.get_rank(), # This is the global rank generally? Print out later? Look at multinode?
                guided_crops_path = args.guided_crops_path,
                guided_crops_size = args.guided_crops_size,
                transform=transforms.Compose([SaturationNoiseInjector(low=200, high=255), PerImageNormalize(), v2.Resize((224,224))]),
                dataset_size=args.dataset_size,
                seed=42,
                use_fp32=True
                )
    
    # Setup the num_epochs as 100
    dataset = IterableImageArchive(config)
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size_per_gpu, num_workers=args.num_workers, worker_init_fn=dataset.worker_init_fn, drop_last=True, prefetch_factor=8, pin_memory=True, persistent_workers=True)

    simclr_transform = SimCLRBatchTransform(image_size=(224, 224))
    
    with open("model_config.yaml", "r") as f:
        model_cfg = yaml.safe_load(f)

    model_cfg["in_chans"] = 1
    model = get_multi_channel_vit(**model_cfg).to(args.gpu)

    # Calculate training steps - CRITICAL for proper scheduler setup
    num_update_steps_per_epoch = math.ceil(len(data_loader) / args.gradient_accumulation_steps)
    num_warmup_steps = num_update_steps_per_epoch * args.warmup_epoch  # 10 epochs of warmup
    total_training_steps = args.epochs * num_update_steps_per_epoch



    ddp_model = DDP(model, device_ids=[args.gpu])

    channel_ids_list = None  # [0] * b  ## list of channel ids for each image in the batch, used for channelViT simclr
    channel_masks = None
    labels = None
    bag_of_channels_mode = True  ## treat each channel as a separate image

    optimizer = get_optimizer(
        params_to_optimize=[{"params": ddp_model.parameters(), "lr": args.lr}],
        learning_rate=args.lr
    )
    lr_scheduler = get_scheduler(
        name = "cosine",
        optimizer = optimizer,
        num_warmup_steps = num_warmup_steps,
        num_training_steps = total_training_steps
    )

    # Initialize training variables
    start_epoch = 0
    global_step = 0

    # Load checkpoint if resuming
    if should_resume and checkpoint_path:
        start_epoch, global_step, last_loss = load_checkpoint(
            checkpoint_path, ddp_model.module, optimizer, lr_scheduler, args.gpu
        )

    ddp_model.train()
    
    # Calculate total steps before training starts
    total_steps_per_epoch = len(data_loader) // args.gradient_accumulation_steps
    total_steps = total_steps_per_epoch * args.epochs

    # Setup the learning rate warmup
   # Training loop
    global_step = 0
    for epoch in range(start_epoch, args.epochs):
        print(f"Starting epoch {epoch + 1}/{args.epochs}")
        epoch_start_time = time.time()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, data in enumerate(data_loader):
            
            # Apply SimCLR transformations
            simclr_data = simclr_transform(data)
            
            # Forward pass
            output = ddp_model(
                simclr_data,
                channel_ids_list=channel_ids_list,
                channel_masks=channel_masks,
                y=labels,
                bag_of_channels_mode=bag_of_channels_mode,
            )
            
            loss = output["loss"]
            
            # Simplified training step for high batch sizes (no gradient accumulation needed)
            if args.gradient_accumulation_steps == 1:
                # Simple case: update every batch
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Enhanced logging - every 10 steps
                if args.gpu == 0 and (global_step % 10 == 0 or global_step < 10):
                    current_lr = lr_scheduler.get_last_lr()[0]
                    elapsed_time = time.time() - epoch_start_time
                    steps_in_epoch = batch_idx + 1
                    if steps_in_epoch > 0:
                        time_per_step = elapsed_time / steps_in_epoch
                        eta_epoch = time_per_step * (total_steps_per_epoch - steps_in_epoch)
                        print(f"Epoch {epoch + 1}, Step {global_step}/{total_steps}: "
                            f"Loss = {loss.item():.4f}, LR = {current_lr:.6f}, "
                            f"ETA Epoch: {eta_epoch/60:.1f}min")
            else:
                # Gradient accumulation for memory-limited setups
                loss = loss / args.gradient_accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                        global_step += 1
                        
                        # Enhanced logging - every 10 steps
                        if args.gpu == 0 and (global_step % 10 == 0 or global_step < 10):
                            current_lr = lr_scheduler.get_last_lr()[0]
                            elapsed_time = time.time() - epoch_start_time
                            if global_step > 0:
                                time_per_step = elapsed_time / (global_step - (epoch * total_steps_per_epoch))
                                eta_epoch = time_per_step * (total_steps_per_epoch - (global_step % total_steps_per_epoch))
                                print(f"Epoch {epoch + 1}, Step {global_step}/{total_steps}: "
                                    f"Loss = {loss.item() * args.gradient_accumulation_steps:.4f}, "
                                    f"LR = {current_lr:.6f}, ETA Epoch: {eta_epoch/60:.1f}min")
            
            epoch_loss += loss.item() * args.gradient_accumulation_steps
            num_batches += 1

        # End of epoch logging with timing
        avg_epoch_loss = epoch_loss / num_batches
        if args.gpu == 0:
            epoch_duration = time.time() - epoch_start_time
            current_lr = lr_scheduler.get_last_lr()[0]
            
            # Estimate total training time remaining
            epochs_remaining = args.epochs - (epoch + 1)
            eta_total = (epoch_duration * epochs_remaining) / 60  # in minutes
            
            print(f"Epoch {epoch + 1} completed in {epoch_duration/60:.1f} minutes: "
                f"Average Loss = {avg_epoch_loss:.4f}, LR = {current_lr:.6f}")
            
            if epochs_remaining > 0:
                print(f"ETA for training completion: {eta_total:.1f} minutes ({eta_total/60:.1f} hours)")
            
            print("-" * 50)
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:  # Save every 5 epochs
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': ddp_model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'loss': avg_epoch_loss,
                'global_step': global_step
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser('SimCLR', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_simclr(args)

