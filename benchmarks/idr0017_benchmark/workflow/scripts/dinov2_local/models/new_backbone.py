import torch
from dinov2_local import models, configs


def new_backbone(old_model, config_path, new_chans = None):
    config = configs.load_and_merge_config(config_path)
    if new_chans is not None:
        config.student.in_chans = new_chans

    old_params = {}

    for name, param in old_model.named_parameters():
        old_params[name] = param.clone()

    new_shape = list(old_params['patch_embed.proj.weight'].shape)
    new_shape[1] = config.student.in_chans
    new_tensor = torch.zeros(new_shape)

    for i in range(config.student.in_chans):
        new_tensor[:, i, :, :] = torch.mean(old_params['patch_embed.proj.weight'], dim=1)
    #new_tensor[:, -3:, :, :] = old_params['patch_embed.proj.weight']
    old_params['patch_embed.proj.weight'] = new_tensor

    student_backbone, embed_dim = models.build_model_from_cfg(config)
    student_backbone.load_state_dict(old_params)
    return student_backbone, embed_dim
