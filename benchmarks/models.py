
from abc import ABC, abstractmethod
import torchvision.transforms.v2.functional as F
import sys
import os
# Add the parent directory (CHAMMI-75) to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feat_models.vision_transformer import vit_small, vit_base, vit_large
from feat_models.models_mae import mae_vit_base_patch16, mae_vit_small_patch16, mae_vit_large_patch16
from feat_models.vit_pool import ViTPoolModel
import numpy as np
from transformers import AutoModel
import torch
import os
import yaml
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
        student_model = torch.load(os.path.join(weights_path, "checkpoint.pth"), weights_only=False)["student"]
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
        self.transform = torchvision.transforms.Compose([SaturationNoiseInjector(), PerImageNormalize(), v2.Resize(size=(224, 224), antialias=True)])

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
            os.path.join(weights_path, "checkpoint-latest.pth"),
            map_location=f"{self.device}" if torch.cuda.is_available() else "cpu",
            weights_only=False
        )
        self.model.load_state_dict(state_dict["model"], strict=False)
        self.model.eval()
        self.transform = torchvision.transforms.Compose([SaturationNoiseInjector(), PerImageNormalize(), v2.Resize(size=(224, 224), antialias=True)])

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
        max_vals = torch.amax(patches.to(torch.float32), dim=(1,2,3)).view(patches.shape[0], 1, 1 ,1)
        normalized_patches = patches/max_vals        
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
                single_channel_patches = self._to_rgb(patches[:,image_idx,:,:].unsqueeze(1))    
                single_channel_patches = self._normalize(single_channel_patches)
                batch_feat.append(self.model.forward_features(single_channel_patches)['x_norm_clstoken'].detach().cpu())
        
            return np.concatenate(batch_feat, axis=1)


class DINOv3(Model):
    def __init__(self, device, repo_dir: str = "/scr/vidit/dinov3", weights: str = "/scr/vidit/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"):
        super().__init__()

        self.repo_dir = repo_dir
        self.weights = weights
        import torch
        self.model = torch.hub.load(self.repo_dir, 'dinov3_vits16', source='local', weights=self.weights)
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
        max_vals = torch.amax(patches.to(torch.float32), dim=(1,2,3)).view(patches.shape[0], 1, 1 ,1)
        normalized_patches = patches/max_vals        
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
                single_channel_patches = self._to_rgb(patches[:,image_idx,:,:].unsqueeze(1))    
                single_channel_patches = self._normalize(single_channel_patches)
                batch_feat.append(self.model.forward_features(single_channel_patches)['x_norm_clstoken'].detach().cpu())
        
            return np.concatenate(batch_feat, axis=1)



class OpenPhenom(Model):
    def __init__(self, device, mode: str = "agg"):
        super().__init__()
        huggingface_modelpath = "recursionpharma/OpenPhenom"
        self.model = AutoModel.from_pretrained(huggingface_modelpath, trust_remote_code=True)
        self.model.eval()
        self.mode = 'conc'  # 'agg' or 'conc'
        self.feature_file = f"pretrained_openphenom_{self.mode}_features.npy"
        self.device = device
    
    def _scale(self, patches: torch.Tensor): 
        max_vals = torch.amax(patches.to(torch.float32), dim=(1,2,3), keepdim=True)
        return patches / max_vals
    
    def _normalize(self, patches: torch.Tensor):
        instance_norm = nn.InstanceNorm2d(patches.shape[1])
        return instance_norm(patches)


    
    def _predict_agg(self, patches: torch.Tensor) -> torch.Tensor:
        X = self.model.encoder.vit_backbone.forward_features(patches)  # 3D tensor N x num_tokens x dim
        return X[:, 1:, :].mean(dim=1)  # Aggregate features
    
    def _predict_conc(self, patches: torch.Tensor) -> torch.Tensor:
        X = self.model.encoder.vit_backbone.forward_features(patches)  # 3D tensor N x num_tokens x dim
        N, _, d = X.shape
        num_channels = patches.shape[1]
        X_reshaped = X[:, 1:, :].view(N, num_channels, -1, d)
        pooled_segments = X_reshaped.mean(dim=2)  # (N, num_channels, d)
        return pooled_segments.view(N, num_channels * d).contiguous()
    
    def _predict(self, patches: torch.Tensor) -> torch.Tensor:
        patches = self._scale(patches)
        patches = self._normalize(patches)
        return self._predict_agg(patches) if self.mode == "agg" else self._predict_conc(patches)
    
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
            return embeddings.detach().cpu()  # Add dim so `get_features` knows this is not for 384 channels
        
class SubCell_Neuron_Feat(Model):
    def __init__(self, device, config_path):
        self.device = device
        # Load config for subcell model
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        # Load subcell model
        self.model = self.get_subcell_model(self.config, config_path.replace('.yaml', '.pth'))
        self.model.eval()
        self.model.to(self.device)
    
    def get_patch_info(self):
        """
        Get the patch size information for the model. Passed in manually
        """
        return 16, 16
    
    def get_subcell_model(self, config, model_path=None):
        model = ViTPoolModel(config["model_config"]["vit_model"], config["model_config"]["pool_model"])
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
            for image_idx in range(patches.shape[1]):
                single_channel_patches = self._to_2chan(patches[:, image_idx, :, :].unsqueeze(1))
                batch_feat.append(self.model(single_channel_patches).feature_vector.cpu())
            
            # Stack the features and return
            return torch.stack(batch_feat, dim=1)


class SubCell(Model):

    def __init__(self, config: dict, encoder:str = "MAE"):
        self.config = config
        self.encoder = encoder
        self.channel_config = config["feature_extraction"]["subcell_config"]
        self.channel_order = self._get_channel_order()
        self.model = self._load_model()
        self.model.eval()
        

    def to(self, device):
        self.model = self.model.to(device)



    def min_max_standardize(self, im):
        min_val = torch.amin(im, dim=(1, 2, 3), keepdims=True)
        max_val = torch.amax(im, dim=(1, 2, 3), keepdims=True)

        im = (im - min_val) / (max_val - min_val + 1e-6)
        return im
    
    def _get_channel_order(self):
        channel_order = [self.channel_config["mt"], self.channel_config["er"], self.channel_config["nucleus"], self.channel_config["protein"]]
        channel_order = [i for i in channel_order if i is not None]
        channel_order = [i-1 for i in channel_order]
        return channel_order


    def get_model_name(self):

        # Model Name Tail
        if self.encoder == "MAE":
            feat_ext_name_suffix = "DNA-Protein_MAE-CellS-ProtS-Pool.pth"
            classifier_name_suffix = "DNA-Protein_ViT_MLP_classifier"

        elif self.encoder == "VIT":
            feat_ext_name_suffix = "DNA-Protein_ViT-ProtS-Pool.pth"
            classifier_name_suffix = "DNA-Protein_MAE_MLP_classifier"

        else:
            raise ValueError("Encoder not recognized. Please use 'mae' or 'vit'.")

        # Channels
        if (self.channel_config['nucleus'] is None) or (self.channel_config['protein'] is None):
            raise ValueError("Nucleus and protein channels must be provided in the channel map.")
        
        if self.channel_config['er'] is not None:
            feat_ext_name_suffix = "ER-" + feat_ext_name_suffix
            classifier_name_suffix = "ER-" + classifier_name_suffix


        if self.channel_config['mt'] is not None:
            feat_ext_name_suffix = "MT-" + feat_ext_name_suffix
            classifier_name_suffix = "MT-" + classifier_name_suffix

        feat_ext_name_suffix = feat_ext_name_suffix.replace("MT-ER-DNA-Protein", "all_channels")
        classifier_name_suffix = classifier_name_suffix.replace("MT-ER-DNA-Protein", "all_channels")

        return feat_ext_name_suffix, classifier_name_suffix
            
    
    def _load_model(self):


        '''
        Function to load the model for inference
        Args:
        1. config: dict: The configuration for the model to be used for inference
        
        Returns:
        The model to be used for inference
        '''

        # LOAD MODEL CONFIG
        feat_ext_name, classifier_name = self.get_model_name()

        feat_ext_path = os.path.join(self.config["models"], "subcell_models", feat_ext_name)
        config_file_path = os.path.join(self.config["models"], "subcell_models",feat_ext_name.replace(".pth", ".yaml"))
        classifier_paths = [os.path.join(self.config["models"], "subcell_models", classifier_name, classifier_name + "_seed_0.pth")]


        with open(config_file_path, "r") as config_buffer:
            model_config_file = yaml.safe_load(config_buffer)

        # LOAD THE MODEL
        model = ViTPoolClassifier(model_config_file)
        model.load_model_dict(feat_ext_path, classifier_paths)
        model.eval()
        
        return model

    def __call__(self, patches: torch.Tensor):

        # Reshuffle the Channels as per the channel order (Dont use permute as it will not allow to duplicate channels)
        input_patches = torch.zeros(patches.shape[0], len(self.channel_order), patches.shape[2], patches.shape[3], device=patches.device)
        for i, channel_idx in enumerate(self.channel_order):
            if channel_idx >= 0:
                input_patches[:, i, :, :] = patches[:, channel_idx, :, :]
        
        # Standardize the Patches
        input_patches = self.min_max_standardize(input_patches)

        # Forwardfeed the model
        with torch.no_grad():
            output = self.model(input_patches)
            features = output.pool_op.cpu()

        # Add Channel dimensions
        features = features.unsqueeze(1)
        return features
    
    @torch.no_grad()
    def run_model(self, cell_crop):
        cell_crop = self.min_max_standardize(cell_crop)
        output = self.model(cell_crop)
        embedding = output.pool_op[0].cpu().numpy()
        # save_attention_map(output.pool_attn, (cell_crop.shape[2], cell_crop.shape[3]), output_path)
        return np.array(embedding)


'''
Uniform get model function which can be called from any of the benchmarks!
'''

def get_model(model_name: str = None, device: torch.device = None, model_path: str = None, model_size: str = "small", model_type: str = "conc"):
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
        model = SubCell_Neuron_Feat(device=device, config_path="/mnt/cephfs/mir/jcaicedo/morphem/dataset/models/subcell_models/DNA-Protein_ViT-ProtS-Pool.yaml")
    else:
        raise ValueError("Model not recognized. Please use 'mae', 'vit', 'dinov2', or 'openphenom'.")
    return model
