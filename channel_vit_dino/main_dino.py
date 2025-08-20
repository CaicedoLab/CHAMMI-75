# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import random
import os
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
sys.path.append("../")
from dataset.dataset import ChannelViTDataset
from dataset import dataset_config
from dataset.dataset_functions import randomize, split_for_workers, get_proc_split
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from omegaconf import OmegaConf
from config import DINOV1Config

import torch
import torchvision.transforms.v2.functional as func
from torchvision.transforms.functional import to_pil_image

import utils
import vision_transformer as vits
from vision_transformer import DINOHead
#os.makedirs("/scratch/cache", exist_ok=True)
#torch.hub.set_dir("/scratch/cache") 

from typing import Union

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

import wandb

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))


def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)
    parser.add_argument('-c', '--config', required=True, help="Path to config file")
    parser.add_argument('--local-rank') # don't use. It's just for silly torchrun things
    return parser


def train_dino(cfg: DINOV1Config):
    utils.init_distributed_mode(cfg)
    utils.fix_random_seeds(cfg.train.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, v) for k, v in sorted(cfg.items())))
    cudnn.benchmark = True

    if utils.is_main_process():
        wandb.init(
                project="channelvit-dino",
                config=OmegaConf.to_container(cfg),
                name=cfg.train.name,
                id=cfg.train.name,
                resume = "allow"
            )

    # ============ preparing data ... ============

     # PREV TRANSFORM FROM DINO
    transform = TensorAugmentationDINO(
        cfg.crops.global_crops_scale,
        cfg.crops.local_crops_scale,
        cfg.crops.local_crops_number,
    )

    # transform = CellAugmentationDino(
    #     is_train=True,
    #     local_crops_number=cfg.crops.local_crops_number,
    #     brightness=True,   
    # )

    config = dataset_config.DatasetConfig(
                cfg.train.data_path, # args.data_path, /scr/data/CHAMMIv2m.zip
                split_fns=[get_proc_split, randomize, split_for_workers],
                num_procs = utils.get_world_size(), # maybe works? brother needs to check!
                proc = torch.distributed.get_rank(), # This is the global rank generally? Print out later? Look at multinode?
                transform=transform,
                dataset_size=cfg.dataset.dataset_size,
                small_list_path = cfg.dataset.small_list_path,
                seed=42,
                dataset_config=cfg.dataset.metadata
        )
    
    # If guided cropping is enabled, we add the guided crops path and size to the config
    if cfg.dataset.guided_cropping:
        config = dataset_config.DatasetConfig(
                cfg.train.data_path, # args.data_path, /scr/data/CHAMMIv2m.zip
                split_fns=[get_proc_split, randomize, split_for_workers],
                num_procs = utils.get_world_size(), # maybe works? brother needs to check!
                proc = torch.distributed.get_rank(), # This is the global rank generally? Print out later? Look at multinode?
                guided_crops_path = cfg.crops.guided_crops_path,
                guided_crops_size = cfg.crops.guided_crops_size,
                transform=transform,
                dataset_size=cfg.dataset.dataset_size,
                small_list_path = cfg.dataset.small_list_path,
                seed=42
                )

    dataset = ChannelViTDataset(config)
    #args.batch_size_per_gpu
    data_loader = DataLoader(dataset=dataset, batch_size=cfg.optim.batch_size_per_gpu, num_workers=cfg.train.num_workers, worker_init_fn=dataset.worker_init_fn, collate_fn=dataset.collate_fn, drop_last=True)
    
    print(f"Data loaded: there are {len(data_loader)} images.")
    # ============ building student and teacher networks ... ============
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if cfg.model.arch in vits.__dict__.keys():
        student = vits.__dict__[cfg.model.arch](
            patch_size=cfg.model.patch_size,
            drop_path_rate=cfg.optim.drop_path_rate,  # stochastic depth
            in_chans = dataset.num_channels # dataset generates this live from the data itself. 
        )
        teacher = vits.__dict__[cfg.model.arch](patch_size=cfg.model.patch_size, in_chans = dataset.num_channels)
        embed_dim = student.embed_dim
    else:
        raise AttributeError("The specified model architecture was not found in the vision_transfomers file.")

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(student, DINOHead(
        embed_dim,
        cfg.model.out_dim,
        use_bn=cfg.model.use_bn_in_head,
        norm_last_layer=cfg.model.norm_last_layer,
    ))
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, cfg.model.out_dim, cfg.model.use_bn_in_head),
    )
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[cfg.train.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[cfg.train.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {cfg.model.arch} network.")

    channel_map = {}
    for idx, channel in enumerate(sorted(dataset.channels)):
        channel_map[channel] = idx

    with open(os.path.join(cfg.train.output_dir, 'channel_map.json'), 'w') as f: f.write(json.dumps(channel_map))
    
    print(f"Channel VIT channel map initialized. We have {len(channel_map.keys())} different channels in the dataset.")
        
    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        cfg.model.out_dim,
        cfg.crops.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        cfg.temp.warmup_teacher_temp,
        cfg.temp.teacher_temp,
        cfg.temp.warmup_teacher_temp_epochs,
        cfg.optim.epochs,
    ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    len(params_groups)
    if cfg.optim.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    else:
        return ValueError("adamw is the only supported optimizer")
     
    fp16_scaler = None
    if cfg.optim.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        cfg.optim.lr,  # linear scaling rule
        cfg.optim.min_lr,
        cfg.optim.epochs, len(data_loader),
        warmup_epochs=cfg.optim.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        cfg.optim.weight_decay,
        cfg.optim.weight_decay_end,
        cfg.optim.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(cfg.model.momentum_teacher, 1,
                                               cfg.optim.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(cfg.train.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting DINO training !")
    for epoch in range(start_epoch, cfg.optim.epochs):

        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
            data_loader, optimizer, channel_map, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, cfg)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': OmegaConf.to_container(cfg),
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(cfg.train.output_dir, 'checkpoint.pth'))
        if cfg.train.saveckp_freq and epoch % cfg.train.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(cfg.train.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(cfg.train.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, channel_map, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, cfg: DINOV1Config):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, cfg.optim.epochs)
    for it, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        if(it == len(lr_schedule)):
            break
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]            
            
        # loss = 0.0
        for channels, minibatch in batch.items():
            extra_tokens = {
                    "channels": [channel_map[chan] for chan in channels]
            }

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                global_crops = minibatch['global_crops'].cuda(non_blocking=True)
                local_crops = minibatch['local_crops'].cuda(non_blocking=True)
                teacher_output = teacher(global_crops, extra_tokens=extra_tokens)
                
                student_output = student([global_crops, local_crops], extra_tokens=extra_tokens)        

        loss = dino_loss(student_output, teacher_output, epoch)
            
        # loss = loss / len(batch)
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if cfg.optim.clip_grad:
                param_norms = utils.clip_gradients(student, cfg.optim.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              cfg.optim.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if cfg.optim.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, cfg.optim.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              cfg.optim.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        if utils.is_main_process():
            stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
            wandb.log(stats)
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

class ChangeBrightness(Transform):
    def __init__(self, p=0.2):
        """
        p is the std dev used in torch.randn() * p + 1.
        We clamp the factor to a minimum of 0.5.
        """
        super().__init__()
        self.p = p

    def _transform(self, inpt, params):
        """
        Expects `inpt` to be a single grayscale image of shape (H, W).

        Returns a single tensor of shape (H, W) with brightness adjusted.
        """
        if inpt.max() == 0:
            return inpt

        factor = (torch.randn(1, device=inpt.device) * self.p + 1).clamp_(min=0.5).item()
        print("Brightness factor:", factor)
        # Unsqueeze channel dim -> (1, H, W)
        out = func.adjust_brightness(inpt.unsqueeze(0), factor)
        # Squeeze back -> (H, W)
        return out.squeeze(0)


class ChangeContrast(Transform):
    def __init__(self, p=0.2):
        """
        p is the std dev used in torch.randn() * p + 1.
        We clamp the factor to a minimum of 0.5.
        """
        super().__init__()
        self.p = p

    def _transform(self, inpt, params):
        """
        Expects `inpt` to be a single grayscale image of shape (H, W).

        Returns a single tensor of shape (H, W) with contrast adjusted.
        """
        if inpt.max() == 0:
            return inpt

        factor = (torch.randn(1, device=inpt.device) * self.p + 1).clamp_(min=0.5).item()
        print("Contrast factor:", factor)
        out = func.adjust_contrast(inpt.unsqueeze(0), factor)
        return out.squeeze(0)


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


class TensorAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flips = transforms.Compose(
            [
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5)
            ]
        )
        self.common_normalization = transforms.Compose([
            v2.ToImageTensor(),
            PerImageNormalize()
            # SaturationNoiseInjector(low=200, high=255),
        ])
        
        augmentation_pipeline = transforms.Compose([
            flips,
            # ChangeBrightness(p=0.4),
            # ChangeContrast(p=0.4),
            # safe_color_jitter
            # random_invert
        ])


        # first global crop
        self.global_transfo1 = transforms.Compose([ #224 for global, 96 for local
            v2.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC, antialias=True),
            augmentation_pipeline
            #color_jittering,
        ])
        

        # second global crop
        self.global_transfo2 = transforms.Compose([
            v2.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC, antialias=True),
            augmentation_pipeline
            #v2.ToImageTensor(),
            #PerImageNormalize()
            #color_jittering,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            v2.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC, antialias=True),
            augmentation_pipeline
            #v2.ToImageTensor(),
            #color_jittering
        ])
    def crop_image(self, image, size=None, width=None, height=None):
        if width is None and height is None:
            image = transforms.RandomCrop(size)(image)
        else:
            image = transforms.RandomCrop((width, height))(image)
        return image


    def __call__(self, image):
        # Apply the common normalization to the input image

        '''
        if (image.size()[1] == 160): # Jump-CP
            size = random.choice([140, 92, 48])
            image = self.crop_image(image, random.randint(int(size*0.9), int(size*1.1)))
        elif (image.size()[1] == 512): # HPA
            size = random.choice([465, 224])
            image = self.crop_image(image, random.randint(int(size*0.9), int(size*1.1)))
        elif (image.size()[1] == 238): # WTC
            size_w = random.choice([215, 120])
            size_h = random.choice([300, 150])
            image = self.crop_image(image, width = random.randint(int(size_w*0.9), int(size_w*1.1)),  height = random.randint(int(size_h*0.9), int(size_h*1.1)))
        '''
        image = self.common_normalization(image)
        crops = []

        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops

class CellAugmentationDino(object):
    def __init__(
        self,
        is_train: bool,
        local_crops_number: int,
        global_resize: int = 224,
        local_resize: int = 96,
        normalization_mean: list[float] = [0.4914, 0.4822, 0.4465],
        normalization_std: list[float] = [0.2023, 0.1994, 0.2010],
        brightness: bool = False,
        use_channel_shuffle: bool = False,
        use_coarse_dropout: bool = True,
        max_channels: int = -1,
    ):
        """
        MulticropAugmentation strategy, as developed by M. Caron
        https://arxiv.org/pdf/2006.09882.pdf.
        ASSUMES images are from the distribution N(0,I).
        global_crops_scale: List[float]
            List of (a, b) that defines the scale, sampled uniformly, at which
            to crop the image for the global crop. For instance, (.8, 1.0) will mean that each
            global crop will shrink the original image to be x ~ Uniform([.8, 1.])
            % of the original size.
        local_crops_scale: List[float]
            List of (a, b) that defines the scale, sampled uniformly, at which
            to crop the image for the local crop. For instance, (.6, .8) will mean that each
            local crop will shrink the original image to be x ~ Uniform([.6, .8])
            % of the original size.
        n_local_crops_per_image : int
            number of of local crops per image in the original pair.
            n_local_crops_per_image==0 implies just a single pair of
            reference images (global crops only), whereas n_local_crops_per_image>0
            (as in DINO) implies applying a local crop to each image n_local_crops_per_image
            times.
        global_resize: int
            After cropping image to be of global_crops_scale size of the original size,
            will resize to this value. 224 by default.
        local_resize: int
            After cropping image to be of local_crops_scale size of the original size,
            will resize to this value. 96 by default.
        """
        flip_rotate = A.OneOf(
            [
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.Rotate(90),
                A.Rotate(180),
                A.Rotate(270),
            ]
        )

        if brightness:
            print("Apply brightness change after flip and rotate")
            flip_rotate = A.Compose([flip_rotate, A.RandomBrightness()])

        mean_div_255 = [m / 255.0 for m in normalization_mean]
        std_div_255 = [s / 255.0 for s in normalization_std]
        normalize = A.Compose([ToTensorV2()])
#A.Normalize(mean_div_255, std_div_255), 
        self.is_train = is_train
        self.normalize = normalize

        # global crop
        if use_coarse_dropout:
            coarse_dropout = A.CoarseDropout(max_holes=10, max_height=10, max_width=10)
        else:
            coarse_dropout = A.NoOp()

        self.global_transform1 = A.Compose(
            [
                RandomPadCrop(global_resize),
                flip_rotate,
                A.Defocus(radius=(1, 3)),
                coarse_dropout,
                normalize
            ]
        )

        self.global_transform2 = A.Compose(
            [
                RandomPadCrop(global_resize),
                flip_rotate,
                A.Defocus(radius=(1, 5)),
                coarse_dropout,
                normalize
            ]
        )
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transform = A.Compose(
            [
                RandomPadAndCropCenter(local_resize),
                flip_rotate,
                A.Defocus(radius=(1, 3)),
                normalize
            ]
        )

        self.use_channel_shuffle = use_channel_shuffle
        self.num_channels = len(mean_div_255)
        self.max_channels = max_channels

    def __call__(self, image: torch.Tensor) -> Union[list[torch.Tensor], torch.Tensor]:
        """
        Takes as input two images as the reference pair,
        and outputs:
        [global_transformed(img1), global_transformed(img2),
         n_local_crops_per_image local_transformed(img1),
         n_local_crops_per_image local_transformed(img2)]
        """
        image = image.numpy()
        if self.is_train:
            crops = []
            crops.append(self.global_transform1(image=image)["image"])
            crops.append(self.global_transform2(image=image)["image"])

            for _ in range(self.local_crops_number):
                crops.append(self.local_transform(image=image)["image"])

            if self.use_channel_shuffle:
                # shuffle the channels for each view in the same way
                idx = torch.randperm(self.num_channels)
                crops = [c[idx, :, :] for c in crops]

            if self.max_channels != -1:
                # subsample channels during training
                idx = torch.randperm(self.num_channels)[: self.max_channels]
                crops = [c[idx, :, :] for c in crops]

            return crops
        else:
            return self.normalize(image=image)["image"]
        
def RandomPadCrop(size):
    """
    Crops image to range of `scale` inputs and resize to `size`
    """
    return A.Compose(
        [
            A.PadIfNeeded(
                min_width=256,
                min_height=256,
                position="random",
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
            ),
            A.RandomCrop(width=size, height=size),
        ]
    )
    
def RandomPadAndCropCenter(size):
    """
    Crops image to range of `scale` inputs and resize to `size`
    """
    return A.Compose(
        [
            A.PadIfNeeded(
                min_width=320,
                min_height=320,
                position="random",
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
            ),
            A.CenterCrop(width=size, height=size),
            # A.ChannelDropout(p=0.2, channel_drop_range=(1, 3)),
        ]
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    config_path = os.path.abspath(os.path.expanduser(args.config))
    cfg = OmegaConf.load(config_path)

    default_config:DINOV1Config = OmegaConf.structured(DINOV1Config)
    cfg: DINOV1Config = OmegaConf.merge(default_config, cfg)
    output_dir = os.path.abspath(os.path.expanduser(cfg.train.output_dir))
    cfg.train.output_dir = os.path.join(output_dir, cfg.train.name)
    
    Path(cfg.train.output_dir).mkdir(parents=True, exist_ok=True)
    train_dino(cfg)
    