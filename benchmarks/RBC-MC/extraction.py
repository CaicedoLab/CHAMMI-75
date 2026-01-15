from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import argparse
from accelerate import Accelerator
import torch.nn.functional as F

from rbc_dataloader import RBC_Dataloader
import sys
from pathlib import Path

# Get the parent directory of the current script
script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent

# Add the parent directory to sys.path
sys.path.insert(0, str(parent_dir))

from models import get_model
import os


def create_pad(images, patch_width, patch_height):  # new method for vit model
    N, C, H, W = images.shape

    new_width = ((W + patch_width - 1) // patch_width) * patch_width
    pad_width = new_width - W

    # Calculate padding amounts for left and right
    pad_left = pad_right = pad_width // 2

    if pad_width % 2 != 0:
        pad_right += 1

    new_height = ((H + patch_height - 1) // patch_height) * patch_height
    pad_height = new_height - H

    # Calculate padding amounts for top and bottom
    pad_top = pad_bottom = pad_height // 2

    if pad_height % 2 != 0:
        pad_bottom += 1

    padded_images = F.pad(
        images, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0
    )

    return padded_images


def extract_embeddings(dataloader, model, accelerator):
    embeddings = []

    # Debug: Print how many samples each GPU will process
    print(
        f"GPU {accelerator.local_process_index} will process {len(dataloader)} samples"
    )

    # Only show progress on main process to avoid multiple progress bars
    if accelerator.is_main_process:
        iterator = tqdm(dataloader, desc="Extracting embeddings")
    else:
        iterator = tqdm(dataloader, desc="Extracting embeddings", disable=True)

    for data in iterator:
        image_tensor = data["image"]
        batch_size = image_tensor.shape[0]

        patch_height, patch_width = model.get_patch_info()

        image_tensor = create_pad(
            image_tensor, patch_width, patch_height
        )  # [N_CH, H_pad, W_pad]
        image_embedding = model(image_tensor.to(accelerator.device))  # [N_CH, D]
        # Unbatch: iterate through each sample in the batch
        for i in range(batch_size):
            # Extract embedding for single sample
            single_embedding = image_embedding[i]  # [N_CH, D]

            # Extract metadata for single sample
            single_metadata = {k: data[k][i] for k in data if k != "image"}

            embeddings.append(
                {"embedding": single_embedding, "metadata": single_metadata}
            )

    return embeddings


def main():
    parser = argparse.ArgumentParser(
        description="Extract features using VIT or subcell model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="vit",
        help="Model to use for feature extraction (default: vit)",
    )  # Choices to get from 'mae', 'vit', 'dinov2', 'dinov3', 'openphenom', 'subcell'
    parser.add_argument(
        "--config_path",
        type=str,
        default="/mnt/cephfs/mir/jcaicedo/morphem/dataset/models/subcell_models/all_channels_ViT-ProtS-Pool.yaml",
        help="Path to config file for subcell model (required when using subcell)",
    )
    parser.add_argument(
        "--image_folder", type=str, default="", help="Path to image folder"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="/scr/vidit/label-free-features/iclr_model",
        help="Output folder for features",
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of workers for data loading"
    )
    parser.add_argument(
        "--model_path", type=str, default="", help="Path to where the model is located"
    )
    args = parser.parse_args()

    accelerator = Accelerator()

    # Create datasets first

    dataloader = RBC_Dataloader(datadir=args.image_folder, transform=None)

    print(f"Total samples in dataset: {len(dataloader)}")

    # Create DataLoaders
    dataloader = DataLoader(
        dataloader,
        batch_size=256,
        shuffle=True,
        num_workers=args.num_workers,
    )

    # Prepare dataloaders for multi-GPU distribution
    dataloader = accelerator.prepare(dataloader)

    # Initialize model
    model_instance = get_model(
        model_name=args.model,
        device=accelerator.device,
        model_path=args.model_path,
    )
    model_instance.to(accelerator.device)

    if args.model == "channelvit":
        model_instance.set_dataset("rbc-mc", args.model_path)

    # Extract embeddings
    embeddings = extract_embeddings(dataloader, model_instance, accelerator)

    # Ensure output folder exists
    os.makedirs(args.output_folder, exist_ok=True)
    embeddings_path = os.path.join(args.output_folder, "embeddings.pkl")

    # Gather all embeddings from all processes
    if accelerator.num_processes > 1:
        # Gather embeddings from all GPUs
        all_embeddings = accelerator.gather_for_metrics(embeddings)

        # Only save on main process to avoid duplicate files
        if accelerator.is_main_process:
            print(f"Gathered {len(all_embeddings)} embeddings from all GPUs")

            with open(embeddings_path, "wb") as f:
                pickle.dump(all_embeddings, f)
    else:
        # Single GPU case
        with open(embeddings_path, "wb") as f:
            pickle.dump(embeddings, f)

    if accelerator.is_main_process:
        print("Embedding extraction complete!")


if __name__ == "__main__":
    main()
