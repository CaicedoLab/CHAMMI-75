from abc import ABC, abstractmethod
import torchvision.transforms.v2.functional as F
import sys
import os

# Add the parent directory (CHAMMI-75) to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feat_models.vision_transformer import vit_small, vit_base, vit_large
from feat_models.models_mae import (
    mae_vit_base_patch16,
    mae_vit_small_patch16,
    mae_vit_large_patch16,
)
from feat_models.multi_channel_vit import get_multi_channel_vit
from feat_models.vit_pool import ViTPoolModel
from feat_models.channelvit.vision_transformer import channelvit_base, channelvit_small
import numpy as np
from transformers import AutoModel
import torch
import os
import yaml
import json
import torchvision
from torchvision.transforms import v2

torchvision.disable_beta_transforms_warning()

import torch.nn as nn


class Model(ABC):
    @abstractmethod
    def __call__(self, patches: torch.Tensor) -> torch.Tensor:
        """The forward pass for your model. The output masks of the dataset will be passed this way
        Tensors are in b,c,h,w"""
        pass

    @abstractmethod
    def to(self, device):
        """Use this to set any models to the passed in device. I.e., just call to ony our model."""
        pass

    @abstractmethod
    def get_model(self):
        """Return the model and any transforms that need to be applied to the input before passing to the model."""
        pass

    @abstractmethod
    def get_patch_info(self):
        """Return the patch height and width that the model was trained on."""
        pass


# Noise Injector transformation
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


# Self Normalize transformation
class PerImageNormalize(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        # We initialize with num_features=1, but we’ll replace it on-the-fly if needed.
        self.eps = eps
        self.instance_norm = nn.InstanceNorm2d(
            num_features=1,  # Temporary placeholder
            affine=False,  # No learnable parameters
            track_running_stats=False,  # Use per-forward stats (no running mean)
            eps=self.eps,
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
                num_features=C, affine=False, track_running_stats=False, eps=self.eps
            )

        # Now we can pass x through our InstanceNorm2d layer
        return self.instance_norm(x)


class HuggingFaceCHAMMI75(Model):
    def __init__(self, device):
        # Load model from Hugging Face
        self.model = AutoModel.from_pretrained("CaicedoLab/CHAMMI-75", trust_remote_code=True)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device).eval()
        print(f"✓ Model loaded on {device}")
        self.feature_file = "pretrained_huggingface_chammi75_features.npy"
        self.device = device
        self.transform = torchvision.transforms.Compose(
            [
                SaturationNoiseInjector(),
                PerImageNormalize(),
                v2.Resize(size=(224, 224), antialias=True),
            ]
        )
    def get_model(self):
        return self.model, self.transform
    
    def get_patch_info(self):
        patch_embed = self.model.patch_embed
        # Access the Conv2d layer within PatchEmbed
        conv_layer = patch_embed.proj

        # Extract kernel size (patch size)
        patch_size = conv_layer.kernel_size
        patch_height, patch_width = patch_size
        return patch_height, patch_width
    
    def to(self, device):
        self.model = self.model.to(device)
    
    def __call__(self, images):
        with torch.no_grad():
            batch_feat = []
            images = images.to(self.device)
            for c in range(images.shape[1]):
                single_channel = images[:, c, :, :].unsqueeze(1)
                output = self.model.forward_features(single_channel)
                feat_temp = output["x_norm_clstoken"].cpu().detach().numpy()

                batch_feat.append(feat_temp)
        return np.concatenate(batch_feat, axis=1)
    




class ViTClass:
    def __init__(self, weights_path: str, model_size: str, device):
        self.device = device
        self.feature_file = "pretrained_vit_features.npy"
        # Create model with in_chans=1 to match training setup
        if model_size == "small":
            self.model = vit_small()
        elif model_size == "base":
            self.model = vit_base()
        elif model_size == "large":
            self.model = vit_large()
        else:
            raise ValueError(
                f"Models of base and small are supported, not {model_size}"
            )

        remove_prefixes = ["module.backbone.", "module.", "module.head."]

        # Load model weights
        student_model = torch.load(weights_path, weights_only=False)["student"]
        # Remove unwanted prefixes
        cleaned_state_dict = {}
        for k, v in student_model.items():
            new_key = k
            for prefix in remove_prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]  # Remove prefix
            if not new_key.startswith("head.mlp") and not new_key.startswith(
                "head.last_layer"
            ):
                cleaned_state_dict[new_key] = v  # Keep only valid keys
        self.model.load_state_dict(cleaned_state_dict, strict=False)
        self.model.eval()
        self.transform = torchvision.transforms.Compose(
            [
                SaturationNoiseInjector(),
                PerImageNormalize(),
                v2.Resize(size=(224, 224), antialias=True),
            ]
        )

    def get_model(self):
        return self.model, self.transform

    def get_patch_info(self):
        patch_embed = self.model.patch_embed
        # Access the Conv2d layer within PatchEmbed
        conv_layer = patch_embed.proj

        # Extract kernel size (patch size)
        patch_size = conv_layer.kernel_size
        patch_height, patch_width = patch_size
        return patch_height, patch_width

    def to(self, device):
        self.model = self.model.to(device)

    def __call__(self, images):
        with torch.no_grad():
            batch_feat = []
            images = images.to(self.device)
            for c in range(images.shape[1]):
                single_channel = images[:, c, :, :].unsqueeze(1)

                output = self.model.forward_features((single_channel))
                feat_temp = output["x_norm_clstoken"].cpu().detach().numpy()

                batch_feat.append(feat_temp)

        return np.concatenate(batch_feat, axis=1)


class MAEModel:
    def __init__(self, device, weights_path, model_size):
        self.device = device
        self.feature_file = "pretrained_mae_features.npy"
        if model_size == "small":
            self.model = mae_vit_small_patch16()
        elif model_size == "base":
            self.model = mae_vit_base_patch16()
        elif model_size == "large":
            self.model = mae_vit_large_patch16()
        else:
            raise ValueError(
                f"Only small, base, and large sized models are supported, not {model_size}"
            )

        state_dict = torch.load(
            weights_path,
            map_location=f"{self.device}" if torch.cuda.is_available() else "cpu",
            weights_only=False,
        )
        self.model.load_state_dict(state_dict["model"], strict=False)
        self.model.eval()
        self.transform = torchvision.transforms.Compose(
            [
                SaturationNoiseInjector(),
                PerImageNormalize(),
                v2.Resize(size=(224, 224), antialias=True),
            ]
        )

    def get_model(self):
        return self.model, self.transform

    def to(self, device):
        self.model = self.model.to(device)

    def get_patch_info(self):
        patch_embed = self.model.patch_embed
        # Access the Conv2d layer within PatchEmbed
        conv_layer = patch_embed.proj

        # Extract kernel size (patch size)
        patch_size = conv_layer.kernel_size
        patch_height, patch_width = patch_size
        return patch_height, patch_width

    def __call__(self, images):
        batch_feat = []
        for c in range(images.shape[1]):
            single_channel = images[:, c, :, :].unsqueeze(1).to(self.device)

            feat_temp = (
                self.model.get_features((single_channel).to(self.device))
                .cpu()
                .detach()
                .numpy()
            )

            batch_feat.append(feat_temp)

        return np.concatenate(batch_feat, axis=1)


class DinoV2(Model):
    def __init__(self, device):
        super().__init__()

        model_name = "dinov2_vits14_reg"
        model_repo = "facebookresearch/dinov2"
        self.model = torch.hub.load(model_repo, model_name)
        # self.model = self.model.transformer
        self.model.eval()
        self.model = self.model.to(device)  # Move model to GPU
        self.device = device

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.feature_file = "pretrained_dinov2_features.npy"

    def _to_rgb(self, patches: torch.Tensor):
        return patches.expand(-1, 3, -1, -1)

    def _normalize(self, patches: torch.Tensor):
        normalized_tensor = F.normalize(patches, mean=self.mean, std=self.std)
        return normalized_tensor

    def _scale(self, patches: torch.Tensor):
        max_vals = torch.amax(patches.to(torch.float32), dim=(1, 2, 3)).view(
            patches.shape[0], 1, 1, 1
        )
        normalized_patches = patches / max_vals
        return normalized_patches

    def to(self, device):
        self.model = self.model.to(device)

    def get_model(self):
        return self.model, v2.Resize(size=(224, 224), antialias=True)

    def get_patch_info(self):
        patch_embed = self.model.patch_embed
        # Access the Conv2d layer within PatchEmbed
        conv_layer = patch_embed.proj

        # Extract kernel size (patch size)
        patch_size = conv_layer.kernel_size
        patch_height, patch_width = patch_size
        return patch_height, patch_width

    def __call__(self, patches: torch.Tensor):
        with torch.no_grad():
            patches = self._scale(patches)
            patches = patches.to(self.device)  # Move input to GPU
            features = []
            batch_feat = []
            for image_idx in range(patches.shape[1]):
                single_channel_patches = self._to_rgb(
                    patches[:, image_idx, :, :].unsqueeze(1)
                )
                single_channel_patches = self._normalize(single_channel_patches)
                batch_feat.append(
                    self.model.forward_features(single_channel_patches)[
                        "x_norm_clstoken"
                    ]
                    .detach()
                    .cpu()
                )

            return np.concatenate(batch_feat, axis=1)


class DINOv3(Model):
    def __init__(
        self,
        device,
        repo_dir: str = "/scr/vidit/dinov3",
        weights: str = "/scr/vidit/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    ):
        super().__init__()

        self.repo_dir = repo_dir
        self.weights = weights
        self.model = torch.hub.load(
            self.repo_dir, "dinov3_vits16", source="local", weights=self.weights
        )
        self.model.eval()
        self.model = self.model.to(device)  # Move model to GPU
        self.device = device

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.feature_file = "pretrained_dinov3_features.npy"

    def _to_rgb(self, patches: torch.Tensor):
        return patches.expand(-1, 3, -1, -1)

    def _normalize(self, patches: torch.Tensor):
        normalized_tensor = F.normalize(patches, mean=self.mean, std=self.std)
        return normalized_tensor

    def _scale(self, patches: torch.Tensor):
        max_vals = torch.amax(patches.to(torch.float32), dim=(1, 2, 3)).view(
            patches.shape[0], 1, 1, 1
        )
        normalized_patches = patches / max_vals
        return normalized_patches

    def to(self, device):
        self.model = self.model.to(device)

    def get_model(self):
        return self.model, v2.Resize(size=(224, 224), antialias=True)

    def get_patch_info(self):
        patch_embed = self.model.patch_embed
        # Access the Conv2d layer within PatchEmbed
        conv_layer = patch_embed.proj

        # Extract kernel size (patch size)
        patch_size = conv_layer.kernel_size
        patch_height, patch_width = patch_size
        return patch_height, patch_width

    def __call__(self, patches: torch.Tensor):
        with torch.no_grad():
            patches = self._scale(patches)
            patches = patches.to(self.device)  # Move input to GPU
            features = []
            batch_feat = []
            for image_idx in range(patches.shape[1]):
                single_channel_patches = self._to_rgb(
                    patches[:, image_idx, :, :].unsqueeze(1)
                )
                single_channel_patches = self._normalize(single_channel_patches)
                batch_feat.append(
                    self.model.forward_features(single_channel_patches)[
                        "x_norm_clstoken"
                    ]
                    .detach()
                    .cpu()
                )

            return np.concatenate(batch_feat, axis=1)


class OpenPhenom(Model):
    def __init__(self, device, mode: str = "agg"):
        super().__init__()
        huggingface_modelpath = "recursionpharma/OpenPhenom"
        self.model = AutoModel.from_pretrained(
            huggingface_modelpath, trust_remote_code=True
        )
        self.model.eval()
        self.mode = "conc"  # 'agg' or 'conc'
        self.feature_file = f"pretrained_openphenom_{self.mode}_features.npy"
        self.device = device

    def _scale(self, patches: torch.Tensor):
        max_vals = torch.amax(patches.to(torch.float32), dim=(1, 2, 3), keepdim=True)
        return patches / max_vals

    def _normalize(self, patches: torch.Tensor):
        instance_norm = nn.InstanceNorm2d(patches.shape[1])
        return instance_norm(patches)

    def _predict_agg(self, patches: torch.Tensor) -> torch.Tensor:
        X = self.model.encoder.vit_backbone.forward_features(
            patches
        )  # 3D tensor N x num_tokens x dim
        return X[:, 1:, :].mean(dim=1)  # Aggregate features

    def _predict_conc(self, patches: torch.Tensor) -> torch.Tensor:
        X = self.model.encoder.vit_backbone.forward_features(
            patches
        )  # 3D tensor N x num_tokens x dim
        N, _, d = X.shape
        num_channels = patches.shape[1]
        X_reshaped = X[:, 1:, :].view(N, num_channels, -1, d)
        pooled_segments = X_reshaped.mean(dim=2)  # (N, num_channels, d)
        return pooled_segments.view(N, num_channels * d).contiguous()

    def _predict(self, patches: torch.Tensor) -> torch.Tensor:
        patches = self._scale(patches)
        patches = self._normalize(patches)
        return (
            self._predict_agg(patches)
            if self.mode == "agg"
            else self._predict_conc(patches)
        )

    def to(self, device):
        self.model = self.model.to(device)

    def get_patch_info(self):
        patch_height, patch_width = self.model.patch_size, self.model.patch_size
        return patch_height, patch_width

    def get_model(self):
        return self.model, None

    def __call__(self, patches: torch.Tensor):
        patches = patches.to(self.device)
        with torch.no_grad():
            embeddings = self._predict(patches)
            return (
                embeddings.detach().cpu()
            )  # Add dim so `get_features` knows this is not for 384 channels


class SubCell_Neuron_Feat(Model):
    def __init__(self, device, config_path):
        self.device = device
        # Load config for subcell model
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        # Load subcell model
        self.model = self.get_subcell_model(
            self.config, config_path.replace(".yaml", ".pth")
        )
        self.model.eval()
        self.model.to(self.device)

    def get_patch_info(self):
        """
        Get the patch size information for the model. Passed in manually
        """
        return 16, 16

    def get_subcell_model(self, config, model_path=None):
        model = ViTPoolModel(
            config["model_config"]["vit_model"], config["model_config"]["pool_model"]
        )
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        return model

    def preprocess_input_subcell(self, images, per_channel=False):
        if per_channel:
            # Normalize each channel independently
            min_val = torch.amin(images, dim=(2, 3), keepdims=True)
            max_val = torch.amax(images, dim=(2, 3), keepdims=True)
        else:
            # Normalize globally across all channels and spatial dims
            min_val = torch.amin(images, dim=(1, 2, 3), keepdims=True)
            max_val = torch.amax(images, dim=(1, 2, 3), keepdims=True)

        images = (images - min_val) / (max_val - min_val + 1e-6)
        return images

    def get_model(self):
        return self.model, None

    def to(self, device):
        self.model = self.model.to(device)
        self.device = device  # Update device reference

    def _to_2chan(self, patches: torch.Tensor):
        return patches.expand(-1, 2, -1, -1)

    def __call__(self, patches: torch.Tensor):
        with torch.no_grad():
            patches = self.preprocess_input_subcell(patches)
            patches = patches.to(self.device)
            batch_feat = []

            # Use first channel (nucleus) as the fixed channel
            nucleus_channel = patches[:, 0, :, :].unsqueeze(
                1
            )  # Shape: [batch, 1, H, W]

            # Combine nucleus with each of the other 13 channels
            for channel_idx in range(1, patches.shape[1]):
                other_channel = patches[:, channel_idx, :, :].unsqueeze(
                    1
                )  # Shape: [batch, 1, H, W]
                # Concatenate nucleus and other channel to make 2-channel input
                two_channel_input = torch.cat(
                    [nucleus_channel, other_channel], dim=1
                )  # Shape: [batch, 2, H, W]
                batch_feat.append(self.model(two_channel_input).feature_vector.cpu())

            # Stack the features and return
            return torch.stack(batch_feat, dim=1)


class SimCLR(Model):
    def __init__(self, device, weights_path, model_size):
        self.device = device
        simclr_config_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "simclr", "model_config.yaml"
        )
        with open(simclr_config_path, "r") as f:
            model_cfg = yaml.safe_load(f)
        model_cfg["in_chans"] = 1  # single channel
        self.model = get_multi_channel_vit(**model_cfg)

        state_dict = torch.load(
            weights_path,
            map_location=f"{self.device}" if torch.cuda.is_available() else "cpu",
            weights_only=False,
        )
        self.model.load_state_dict(state_dict["model_state_dict"], strict=False)
        self.model.eval()
        self.transform = torchvision.transforms.Compose(
            [
                SaturationNoiseInjector(),
                PerImageNormalize(),
                v2.Resize(size=(224, 224), antialias=True),
            ]
        )
        self.feature_file = "pretrained_simclr_features.npy"

    def get_model(self):
        return self.model, self.transform

    def to(self, device):
        self.model = self.model.to(device)

    def get_patch_info(self):
        patch_embed = self.model.patch_embed
        # Access the Conv2d layer within PatchEmbed
        conv_layer = patch_embed.proj

        # Extract kernel size (patch size)
        patch_size = conv_layer.kernel_size
        # patch_height, patch_width = patch_size
        return patch_size[1], patch_size[2]

    def __call__(self, images):
        batch_feat = []
        for c in range(images.shape[1]):
            single_channel = images[:, c, :, :].unsqueeze(1).to(self.device)

            feat_temp = self.model(single_channel)
            feat_temp = feat_temp["output"].cpu().detach().numpy()

            batch_feat.append(feat_temp)

        return np.concatenate(batch_feat, axis=1)


class ChannelVITSimCLR(Model):
    def __init__(self, model_path, model_size, device):
        self.device = device
        self.dataset_channels = None  # will be a list
        simclr_config_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "models",
            "channel_vit_simclr",
            "model_config.yaml",
        )
        with open(simclr_config_path, "r") as f:
            model_cfg = yaml.safe_load(f)
        self.model_path = model_path

        with open(
            os.path.join(os.path.dirname(model_path), "channel_map.json"), "r"
        ) as f:
            channel_map_file = f.read()
        self.channel_map = json.loads(channel_map_file)

        model_cfg["in_chans"] = len(self.channel_map)  # single channel
        self.model = get_multi_channel_vit(**model_cfg)

        state_dict = torch.load(
            model_path,
            map_location=f"{self.device}" if torch.cuda.is_available() else "cpu",
            weights_only=False,
        )
        self.model.load_state_dict(state_dict["model_state_dict"], strict=False)
        self.model.eval()
        self.transform = torchvision.transforms.Compose(
            [
                SaturationNoiseInjector(),
                PerImageNormalize(),
                v2.Resize(size=(224, 224), antialias=True),
            ]
        )

        self.feature_file = "pretrained_chanvit_simclr_features.npy"
        # Create model with in_chans=1 to match training setup
        self.model.eval()
        self.model.to(self.device)

    # Add to config
    def to(self, device):
        self.model = self.model.to(device)

    def set_dataset(self, dataset_name, model_path):
        if dataset_name == "Allen":
            if "75ds" in model_path or "10ds" in model_path:
                self.dataset_channels = ["nucleus", "cell body", "protein"]
            else:
                self.dataset_channels = ["nucleus", "membrane", "protein"]
        elif dataset_name == "CP":
            if "75ds" in model_path or "10ds" in model_path:
                self.dataset_channels = [
                    "nucleus",
                    "endoplasmic reticulum",
                    "RNA",
                    "golgi body",
                    "mitochondria",
                ]
            else:
                self.dataset_channels = ["nucleus", "cp2", "er", "cp4", "cp5"]
        elif dataset_name == "HPA":
            if "75ds" in model_path or "10ds" in model_path:
                self.dataset_channels = [
                    "microtubules",
                    "protein",
                    "nucleus",
                    "endoplasmic reticulum",
                ]
            else:
                self.dataset_channels = ["microtubules", "protein", "nucleus", "er"]
        else:
            raise ValueError(
                "Dataset name supplied is not supported. This class only supports CHAMMIv1 benchmarking."
            )

    # Add to config
    def get_model(self):
        return self.model, self.transform

    # Add to config
    def get_patch_info(self):
        patch_embed = self.model.patch_embed
        # Access the Conv2d layer within PatchEmbed
        conv_layer = patch_embed.proj

        # Extract kernel size (patch size)
        patch_size = conv_layer.kernel_size
        return patch_size[1], patch_size[2]

    def __call__(self, images):
        channel_ids = [
            [
                self.channel_map[chan] if chan in self.dataset_channels else 0
                for chan in self.dataset_channels
            ]
        ] * len(images)
        channel_masks = [[True for _ in range(images.shape[1])]] * len(images)
        channel_ids_tensor = torch.tensor(
            channel_ids, dtype=torch.long, device=self.device
        )
        channel_masks_tensor = torch.tensor(
            channel_masks, dtype=torch.bool, device=self.device
        )
        with torch.no_grad():
            images = images.to(self.device)
            output = self.model(
                images,
                channel_ids_list=channel_ids_tensor,
                channel_masks=channel_masks_tensor,
            )
            return output["output"].cpu().detach().numpy()


class ChannelVITMAE(Model):
    def __init__(self, model_path, model_size, device):
        self.device = device
        self.dataset_channels = None  # will be a list
        simclr_config_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "models",
            "channel_vit_mae",
            "model_config.yaml",
        )
        with open(simclr_config_path, "r") as f:
            model_cfg = yaml.safe_load(f)
        self.model_path = model_path

        with open(
            os.path.join(os.path.dirname(model_path), "channel_map.json"), "r"
        ) as f:
            channel_map_file = f.read()
        self.channel_map = json.loads(channel_map_file)

        model_cfg["in_chans"] = len(self.channel_map)  # single channel
        self.model = get_multi_channel_vit(**model_cfg)

        state_dict = torch.load(
            model_path,
            map_location=f"{self.device}" if torch.cuda.is_available() else "cpu",
            weights_only=False,
        )
        self.model.load_state_dict(state_dict["model_state_dict"], strict=False)
        self.model.eval()
        self.transform = torchvision.transforms.Compose(
            [
                SaturationNoiseInjector(),
                PerImageNormalize(),
                v2.Resize(size=(224, 224), antialias=True),
            ]
        )

        self.feature_file = "pretrained_chanvit_mae_features.npy"
        # Create model with in_chans=1 to match training setup
        self.model.eval()
        self.model.to(self.device)

    # Add to config
    def to(self, device):
        self.model = self.model.to(device)

    def set_dataset(self, dataset_name, model_path):
        if dataset_name == "Allen":
            if "75ds" in model_path or "10ds" in model_path:
                self.dataset_channels = ["nucleus", "cell body", "protein"]
            else:
                self.dataset_channels = ["nucleus", "membrane", "protein"]
        elif dataset_name == "CP":
            if "75ds" in model_path or "10ds" in model_path:
                self.dataset_channels = [
                    "nucleus",
                    "endoplasmic reticulum",
                    "RNA",
                    "golgi body",
                    "mitochondria",
                ]
            else:
                self.dataset_channels = ["nucleus", "cp2", "er", "cp4", "cp5"]
        elif dataset_name == "HPA":
            if "75ds" in model_path or "10ds" in model_path:
                self.dataset_channels = [
                    "microtubules",
                    "protein",
                    "nucleus",
                    "endoplasmic reticulum",
                ]
            else:
                self.dataset_channels = ["microtubules", "protein", "nucleus", "er"]
        else:
            raise ValueError(
                "Dataset name supplied is not supported. This class only supports CHAMMIv1 benchmarking."
            )

    # Add to config
    def get_model(self):
        return self.model, self.transform

    # Add to config
    def get_patch_info(self):
        patch_embed = self.model.patch_embed
        # Access the Conv2d layer within PatchEmbed
        conv_layer = patch_embed.proj

        # Extract kernel size (patch size)
        patch_size = conv_layer.kernel_size
        return patch_size[1], patch_size[2]

    def __call__(self, images):
        channel_ids = [
            [
                self.channel_map[chan] if chan in self.dataset_channels else 0
                for chan in self.dataset_channels
            ]
        ] * len(images)
        channel_masks = [[True for _ in range(images.shape[1])]] * len(images)
        channel_ids_tensor = torch.tensor(
            channel_ids, dtype=torch.long, device=self.device
        )
        channel_masks_tensor = torch.tensor(
            channel_masks, dtype=torch.bool, device=self.device
        )
        with torch.no_grad():
            images = images.to(self.device)
            output = self.model(
                images,
                channel_ids_list=channel_ids_tensor,
                channel_masks=channel_masks_tensor,
            )
            return output["output"].cpu().detach().numpy()


class ChannelVIT:
    def __init__(self, model_path, model_size, device):
        self.device = device
        self.dataset_channels = None  # will be a list
        self.model_path = model_path

        with open(
            os.path.join(os.path.dirname(model_path), "channel_map.json"), "r"
        ) as f:
            channel_map_file = f.read()
        self.channel_map = json.loads(channel_map_file)

        self.feature_file = "pretrained_vit_features.npy"
        # Create model with in_chans=1 to match training setup
        if model_size == "base":
            self.model = channelvit_base(in_chans=len(self.channel_map))
        elif model_size == "small":
            self.model = channelvit_small(in_chans=len(self.channel_map))
        else:
            raise ValueError(
                f"Models of base and small are supported, not {model_size}"
            )

        remove_prefixes = ["module.backbone.", "module.", "module.head."]

        # Load model weights
        student_model = torch.load(model_path, weights_only=False)["student"]
        # Remove unwanted prefixes
        cleaned_state_dict = {}
        for k, v in student_model.items():
            new_key = k
            for prefix in remove_prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]  # Remove prefix
            if not new_key.startswith("head.mlp") and not new_key.startswith(
                "head.last_layer"
            ):
                cleaned_state_dict[new_key] = v  # Keep only valid keys
        self.model.load_state_dict(cleaned_state_dict, strict=False)
        self.model.eval()
        self.model.to(self.device)

        self.transform = torchvision.transforms.Compose(
            [PerImageNormalize(), v2.Resize(size=(224, 224), antialias=True)]
        )

    def to(self, device):
        self.model = self.model.to(device)

    def get_patch_info(self):
        patch_embed = self.model.patch_embed
        # Access the Conv2d layer within PatchEmbed
        conv_layer = patch_embed.proj

        # Extract kernel size (patch size)
        patch_size = conv_layer.kernel_size
        patch_height, patch_width = patch_size[1:]
        return patch_height, patch_width

    def get_model(self):
        return self.model, self.transform

    def set_dataset(self, dataset_name, model_path):
        if dataset_name == "Allen":
            if "_75ds" in model_path or "_10ds" in model_path:
                self.dataset_channels = ["nucleus", "cell body", "protein"]
            else:
                self.dataset_channels = ["nucleus", "membrane", "protein"]
        elif dataset_name == "CP":
            if "_75ds" in model_path or "_10ds" in model_path:
                self.dataset_channels = [
                    "nucleus",
                    "endoplasmic reticulum",
                    "RNA",
                    "golgi body",
                    "mitochondria",
                ]
            else:
                self.dataset_channels = ["nucleus", "cp2", "er", "cp4", "cp5"]
        elif dataset_name == "HPA":
            if "_75ds" in model_path or "_10ds" in model_path:
                self.dataset_channels = [
                    "microtubules",
                    "protein",
                    "nucleus",
                    "endoplasmic reticulum",
                ]
            else:
                self.dataset_channels = ["microtubules", "protein", "nucleus", "er"]
        elif dataset_name == "neuron":
            self.dataset_channels = [
                "nucleus",
                "protein",
                "protein",
                "protein",
                "protein",
                "protein",
                "protein",
                "protein",
                "protein",
                "protein",
                "protein",
                "RNA",
                "endoplasmic reticulum",
                "golgi body",
            ]
        elif dataset_name == "idr17":
            self.dataset_channels = ["nucleus", "cytoskeleton"]
        elif dataset_name == "mini-HPA":
            self.dataset_channels = [
                "microtubules",
                "endoplasmic reticulum",
                "nucleus",
                "protein",
            ]
        elif dataset_name == "rbc-mc":
            self.dataset_channels = ["bright-field"]
        else:
            raise ValueError(
                "Dataset name supplied is not supported. This class only supports CHAMMIv1 benchmarking."
            )

    def __call__(self, images):
        channel_ids = [
            [
                self.channel_map[chan] if chan in self.dataset_channels else 0
                for chan in self.dataset_channels
            ]
        ] * len(images)
        channel_masks = [[True for _ in range(images.shape[1])]] * len(images)
        with torch.no_grad():
            images = images.to(self.device)
            return self.model(images, channel_ids, channel_masks).cpu().detach().numpy()


"""
Uniform get model function which can be called from any of the benchmarks!
"""


def get_model(
    model_name: str = None,
    device: torch.device = None,
    model_path: str = None,
    model_size: str = "small",
    model_type: str = "conc",
):
    if model_name == "mae":
        model = MAEModel(device=device, weights_path=model_path, model_size=model_size)
    elif model_name == "vit":
        model = ViTClass(device=device, weights_path=model_path, model_size=model_size)
    elif model_name == "dinov2":
        model = DinoV2(device=device)
    elif model_name == "openphenom":
        model = OpenPhenom(device=device, mode=model_type)
    elif model_name == "dinov3":
        model = DINOv3(device=device)
    elif model_name == "subcell":
        model = SubCell_Neuron_Feat(
            device=device,
            config_path="/mnt/cephfs/mir/jcaicedo/morphem/dataset/models/subcell_models/DNA-Protein_ViT-ProtS-Pool.yaml",
        )
    elif model_name == "channelvit":
        model = ChannelVIT(device=device, model_path=model_path, model_size=model_size)
    elif model_name == "simclr":
        model = SimCLR(device=device, weights_path=model_path, model_size=model_size)
    elif model_name == "chanvit_simclr":
        model = ChannelVITSimCLR(
            model_path=model_path, model_size=model_size, device=device
        )
    elif model_name == "chanvit_mae":
        model = ChannelVITMAE(
            model_path=model_path, model_size=model_size, device=device
        )
    elif model_name == "huggingface_chammi75":
        model = HuggingFaceCHAMMI75(device=device)
    else:
        raise ValueError(
            "Model not recognized. Please use 'mae', 'vit', 'dinov2', 'openphenom', 'simclr', 'chanvit_simclr' or 'chanvit_mae'."
        )
    return model
