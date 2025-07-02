import torch
from collections import defaultdict
from typing import Optional
import os
from torchvision.io import decode_image

def can_tile_tensor(tensor: torch.Tensor, tile_width: int, tile_height: int):
    tensor_shape = tensor[0].shape
    assert tensor_shape[1] % tile_width == 0, "Tile width is not divisible by image width"
    assert tensor_shape[0] % tile_height == 0, "Tile height is not divisible by image width"
    
def resize_image_tensor(tensor: torch.Tensor, tile_height: int, tile_width: int):
    orig_h, orig_w = tensor.shape[1:]
    new_height, new_width = get_resized_dims(tensor, tile_height, tile_width)
    if orig_w != new_width or orig_h != new_height:
        tensor = tensor[:, :new_height, :new_width]
    
    return tensor

def get_resized_dims(tensor: torch.Tensor, tile_height: int, tile_width: int):
    image_height, image_width = tensor.shape[1:]
    while image_height % tile_height != 0: image_height -= 1
    while image_width % tile_width   != 0: image_width  -= 1
    
    return image_height, image_width

def get_all_tiles(tensor: torch.Tensor, tile_width: int, tile_height: int, resize: bool= False):
    """ Returns a tensor in format batch, tile_number, tile_height, tile_width
    tensor: A torch tensor in with shape: batch, width, height
    """
    if resize:
        tensor = resize_image_tensor(tensor, tile_height, tile_width)
    else:
        can_tile_tensor(tensor, tile_width, tile_height)
    
    tiles = []
    rows = tensor.split(tile_height, dim=1)
    for row in rows: tiles.extend(row.split(tile_width, dim=2))
    
    return torch.stack(tiles, dim=1)

def get_tile(tensor: torch.Tensor, tile: int, tile_width: int, tile_height: int, resize: bool= False):
    """ Returns an individual 0-based indexed tile from an image tensor
    tensor: A torch tensor in with shape: batch, width, height
    Would be weird to use this with a batch, but I want to keep the API consistent.
    """
    if resize:
        tensor = resize_image_tensor(tensor, tile_height, tile_width)
    else:
        can_tile_tensor(tensor, tile_width, tile_height)
     
    num_horizontal_tiles = tensor.shape[2] // tile_width
    tile_col_pos = tile % num_horizontal_tiles
    tile_row_pos = tile // num_horizontal_tiles

    return tensor[:, tile_row_pos*tile_height:(tile_row_pos+1)*tile_height, tile_col_pos*tile_width:(tile_col_pos+1)*tile_width]
