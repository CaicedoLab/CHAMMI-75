
from abc import ABC, abstractmethod
import torchvision.transforms.v2.functional as F
from dinov2_local.models.vision_transformer import DinoVisionTransformer
from subcell_local.vit_model import ViTPoolClassifier
import numpy as np
# import huggingface_hub
from transformers import AutoModel
import torch
import os
import yaml


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

class DinoV2(Model):
    def __init__(self, config: dict):
        super().__init__()
        
        model_name = "dinov2_vits14_reg"
        model_repo = "facebookresearch/dinov2"
        self.model:DinoVisionTransformer = torch.hub.load(model_repo, model_name)
        # self.model = self.model.transformer
        self.model.eval()
        
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
    
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
    
    def __call__(self, patches: torch.Tensor):
        with torch.no_grad():
            patches = self._scale(patches)
            features = []
            for image_idx in range(patches.shape[1]):
                single_channel_patches = self._to_rgb(patches[:,image_idx,:,:].unsqueeze(1))    
                single_channel_patches = self._normalize(single_channel_patches)
                features.append(self.model.forward_features(single_channel_patches)['x_norm_clstoken'].detach().cpu())
        
            return torch.stack(features, dim=1)

class OpenPhenom(Model):
    def __init__(self, config: dict, mode: str = "agg"):
        super().__init__()
        huggingface_modelpath = "recursionpharma/OpenPhenom"
        self.model = AutoModel.from_pretrained(huggingface_modelpath, cache_dir=config['models'], trust_remote_code=True)
        self.model.eval()
        self.mode = mode  # 'agg' or 'conc'
    
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
    
    def __call__(self, patches: torch.Tensor):
        with torch.no_grad():
            embeddings = self._predict(patches)
            return embeddings.unsqueeze(1)  # Add dim so `get_features` knows this is not for 384 channels
        

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





        

    
    
        
