import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import sys
import argparse
import sc_dataset

sys.path.append("../")
from models import get_model
import os


def configure_dataset(root_dir, plate_name, transform=None):
    root_dir = f"{root_dir}/{plate_name}"
    metadata_path = "sc-metadata.csv"
    dataset = sc_dataset.SingleCellDataset(
        root=root_dir, metadata_path=metadata_path, transform=transform
    )
    return dataset


def get_save_features(
    feature_dir, root_dir, model_check, gpu, batch_size, model_path, model_size
):
    plates_for_fe = [
        "BR00117010",
        "BR00117011",
        "BR00117012",
        "BR00117013",
        "BR00117024",
        "BR00117025",
        "BR00117026",
    ]  # CP-JUMP1 compound plates
    if not os.path.exists(args.feat_dir):
        os.makedirs(args.feat_dir, exist_ok=True)

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    model_instance = get_model(
        model_name=model_check,
        device=device,
        model_path=model_path,
        model_size=model_size,
    )
    model, transform = model_instance.get_model()
    model = model.to(device)

    for plate in plates_for_fe:
        # Post crops and processing getting the transforms
        dataset = configure_dataset(root_dir, plate, transform=transform)
        train_dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, drop_last=False
        )

        all_feat = []
        for images in tqdm(train_dataloader, total=len(train_dataloader)):
            batch_feat = model_instance(images)
            all_feat.append(batch_feat)

        all_feat = np.concatenate(all_feat)
        feature_path = feature_path = f"{feature_dir}/{plate}"
        np.save(feature_path, all_feat)
        torch.cuda.empty_cache()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        help="The root directory of the original images",
        required=True,
    )
    parser.add_argument(
        "--feat_dir",
        type=str,
        help="The directory that contains the features",
        required=True,
    )
    parser.add_argument(
        "--model",
        type=str,
        help="The type of model that is being trained and evaluated (mae, openphenom, dinov2 or vit)",
        required=True,
    )  # Choices to get from 'mae', 'vit', 'dinov2', 'dinov3', 'openphenom'
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="The gpu that is currently available/not in use",
        required=False,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Select a batch size that works for your gpu size",
        required=True,
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="small",
        help="Size of the ViT model (small or base)",
        choices=["small", "base", "large"],
    )
    parser.add_argument(
        "--model_path", type=str, help="Path to the model weights", required=False
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    get_save_features(
        args.feat_dir,
        args.root_dir,
        args.model,
        args.gpu,
        args.batch_size,
        args.model_path,
        args.model_size,
    )
