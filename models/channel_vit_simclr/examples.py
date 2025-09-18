###########################################################
########################  EXAMPLE USAGE ###################
###########################################################
import os
import yaml
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from multi_channel_vit import get_multi_channel_vit


def setup(rank, world_size):
    dist.init_process_group(
        backend="nccl",  # use "gloo" if you're on CPU only
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


############ Example 1: BoC SimCLR
def run_BoC_demo(rank, world_size):
    setup(rank, world_size)

    with open("model_config.yaml", "r") as f:
        model_cfg = yaml.safe_load(f)

    model_cfg["in_chans"] = 1
    model = get_multi_channel_vit(**model_cfg).to(rank)

    ddp_model = DDP(model, device_ids=[rank])

    b, c, h, w = 6, 1, 224, 224
    images = torch.randn(b, c, h, w, device=rank)
    ##### batch consists of 3 images, with 2 views by this order: ## [image_1_view1, image_2_view1, image_3_view1, image_1_view2, image_2_view2, image_3_view2]

    channel_ids_list = None  # [0] * b  ## list of channel ids for each image in the batch, used for channelViT simclr
    channel_masks = None
    labels = None
    bag_of_channels_mode = True  ## treat each channel as a separate image

    ddp_model.train()
    output = ddp_model(
        images,
        channel_ids_list=channel_ids_list,
        channel_masks=channel_masks,
        y=labels,
        bag_of_channels_mode=bag_of_channels_mode,
    )

    if rank == 0:
        print("Output keys:", output.keys())
        print("Output shape:", output["output"].shape)
        print("Loss:", output["loss"].item())

    cleanup()


def BoC_main():
    world_size = torch.cuda.device_count()
    mp.spawn(run_BoC_demo, args=(world_size,), nprocs=world_size, join=True)


############ Example 2: Multi-channel ViT SimCLR
def run_multi_channel_vit_demo(rank, world_size):
    setup(rank, world_size)

    with open("model_config.yaml", "r") as f:
        model_cfg = yaml.safe_load(f)

    model_cfg["in_chans"] = 25  # multi-channel input
    model = get_multi_channel_vit(**model_cfg).to(rank)

    ddp_model = DDP(model, device_ids=[rank])

    ## assume we have 6 images (3 images, each has 2 views), and
    ## maximum 3 channels per image in this batch
    b, c, h, w = 6, 3, 224, 224
    ##### batch consists of 3 images, with 2 views by this order: ## [image_1_view1, image_2_view1, image_3_view1, image_1_view2, image_2_view2, image_3_view2]
    images = torch.randn(b, c, h, w, device=rank)

    channel_ids_list = [[0, 1], [13, 3, 4], [1, 2, 3]] * 2
    channel_masks = torch.tensor([[True, True, False], [True, True, True], [True, True, True]] * 2)
    labels = None
    bag_of_channels_mode = False

    ddp_model.train()
    output = ddp_model(
        images,
        channel_ids_list=channel_ids_list,
        channel_masks=channel_masks,
        y=labels,
        bag_of_channels_mode=bag_of_channels_mode,
    )

    if rank == 0:
        print("Output keys:", output.keys())
        print("Output shape:", output["output"].shape)
        print("Loss:", output["loss"].item())

    cleanup()


def multi_channel_vit_main():
    world_size = torch.cuda.device_count()
    mp.spawn(run_multi_channel_vit_demo, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    ###### command to run the examples ######
    ## cd CHAMMI-75/models
    ## torchrun -m simclr.examples --nproc_per_node=1

    print(">>>>>> running BoC simclr example ...")
    BoC_main()

    print("\n>>>>>> running multi-channel ViT simclr example ...")
    multi_channel_vit_main()

    ###### sample output ######
    """
    $ torchrun -m simclr.examples --nproc_per_node=1
    >>>>>> running BoC simclr example ...
    Output keys: dict_keys(['output', 'proxy_loss', 'supcon_loss', 'simclr_loss', 'loss'])
    Output shape: torch.Size([6, 384])
    Loss: 1.5402511358261108

    >>>>>> running multi-channel ViT simclr example ...
    Output keys: dict_keys(['output', 'proxy_loss', 'supcon_loss', 'simclr_loss', 'loss'])
    Output shape: torch.Size([6, 384])
    Loss: 1.6703466176986694
    """
