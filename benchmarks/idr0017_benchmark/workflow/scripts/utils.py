import torch
import numpy as np
import pandas as pd
import csv
import yaml
from tqdm import tqdm
import shutil
import zipfile
from pathlib import Path

def print_cfg_differences(curr, saved):
    differences = {}
    for key in curr:
        if key not in saved or curr[key] != saved[key]:
            differences[key] = {'current': curr[key], 'saved': saved.get(key, 'Key not found')}
    for key in saved:
        if key not in curr:
            differences[key] = {'current': 'Key not found', 'saved': saved[key]}
    
    for key, diff in differences.items():
        print(f"Difference in key '{key}':")
        print(f"  Current: {yaml.dump(diff['current'], default_flow_style=False)}")
        print(f"  Saved: {yaml.dump(diff['saved'], default_flow_style=False)}")
        print()

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print(f"=> Saving checkpoint {filename}")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def get_accuracy(scores, targets):
    ## scores are raw output from model, targets are the indices of each class after sorting
    preds = torch.argmax(scores, dim=1)
    correct = torch.eq(preds, targets)
    num_correct = torch.sum(correct).item()
    accuracy = num_correct / len(preds)
    return accuracy


def calculate_patch_coords(img_shape, patch_size: int):
    """
    Calculate patch coordinates for a given image shape.

    Args:
        img_shape (tuple): Shape of the image tensor [B, C, H, W], [C, H, W], or [H, W].
        patch_size (int): Size of the patches.

    Returns:
        np.ndarray: Array of patch coordinates.
    """
    if len(img_shape) == 4:
        _, _, img_h, img_w = img_shape
    elif len(img_shape) == 3:
        img_h, img_w = img_shape[1], img_shape[2]
    elif len(img_shape) == 2:
        img_h, img_w = img_shape[0], img_shape[1]
    else:
        raise ValueError("Unsupported tensor shape")

    patch_h, patch_w = patch_size, patch_size

    # Number of patches
    n_patches_x = np.uint16(np.ceil(img_w / patch_w))
    n_patches_y = np.uint16(np.ceil(img_h / patch_h))

    # Total remainders
    remainder_x = n_patches_x * patch_w - img_w
    remainder_y = n_patches_y * patch_h - img_h

    # Set up remainders per patch
    remainders_x = np.uint16(np.ones((n_patches_x - 1)) * np.floor(remainder_x / (n_patches_x - 1)))
    remainders_y = np.uint16(np.ones((n_patches_y - 1)) * np.floor(remainder_y / (n_patches_y - 1)))
    remainders_x[0:np.remainder(remainder_x, n_patches_x - 1)] += 1
    remainders_y[0:np.remainder(remainder_y, n_patches_y - 1)] += 1

    # Initialize array of patch coordinates: rows = [start_y, end_y, start_x, end_x]
    patch_coords = np.zeros((n_patches_x * n_patches_y, 4), np.uint16)

    # Fill in patch coordinates array (in order of L->R, top->bottom)
    k = 0
    y = 0
    for i in range(n_patches_y):
        x = 0
        for j in range(n_patches_x):
            patch_coords[k] = [y, y + patch_h, x, x + patch_w]
            k += 1
            if j < (n_patches_x - 1):
                x = x + patch_w - remainders_x[j]
        if i < (n_patches_y - 1):
            y = y + patch_h - remainders_y[i]

    return patch_coords

def patchify_tensor(img: torch.Tensor, patch_coords, patch_size):
    """
    Create a new tensor using the calculated patch coordinates.

    Args:
        img (torch.Tensor): Input tensor of shape [B, C, H, W].
        patch_size (int): Size of the patches.

    Returns:
        torch.Tensor: Output tensor of shape [B * patches, C, patch_size, patch_size].
    """
    B, C, H, W = img.shape
    if len(patch_coords) == 0:
        return torch.zeros((B, C, patch_size, patch_size))
    
    # Pad the image if necessary
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    if pad_h > 0 or pad_w > 0:
        img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h))

    patches = []
    for b in range(B):
        for coord in patch_coords:
            y1, y2, x1, x2 = coord
            patch = img[b, :, y1:y2, x1:x2]
            patches.append(patch)

    return torch.stack(patches)


# give an img as torch.Tensor/np.array (C, H, W)
# can also handle torch.Tensor/np.array (H, W) as input for a single channel image
def patchify(img: torch.Tensor, patch_size: int):
    if len(img.shape) == 4:  # Batch of images
        B, C, H, W = img.shape
        patches = []
        patch_coords = []

        for b in range(B):
            single_img_patches, single_img_coords = patchify(img[b], patch_size)
            patches.append(single_img_patches)
            patch_coords.append(single_img_coords)

        return patches, patch_coords
    
    if len(img.shape) == 3:
        img_h, img_w = img.size()[1], img.size()[2]
        patch_h, patch_w = patch_size, patch_size

        # # Make sure image not too small (center it in 0s in either direction if it is)
        if img_h < patch_h or img_w < patch_w:
            background = torch.zeros((img.size()[0], max(img_h, patch_h), max(img_w, patch_w)))
            start_y = (patch_h - img_h) // 2 if img_h < patch_h else 0
            start_x = (patch_w - img_w) // 2 if img_w < patch_w else 0
            background[:, start_y:start_y + img_h, start_x:start_x + img_w] = img
            img = background
            img_h, img_w = img.size()[1], img.size()[2]

        # Number of patches
        n_patches_x = np.uint16(np.ceil(img_w / patch_w))
        n_patches_y = np.uint16(np.ceil(img_h / patch_h))

        # Total remainders
        remainder_x = n_patches_x * patch_w - img_w
        remainder_y = n_patches_y * patch_h - img_h

        # Set up remainders per patch
        remainders_x = np.uint16(np.ones((n_patches_x - 1)) * np.floor(remainder_x / (n_patches_x - 1)))
        remainders_y = np.uint16(np.ones((n_patches_y - 1)) * np.floor(remainder_y / (n_patches_y - 1)))
        remainders_x[0:np.remainder(remainder_x, n_patches_x - 1)] += 1
        remainders_y[0:np.remainder(remainder_y, n_patches_y - 1)] += 1

        # Initialize array of patch coordinates: rows = [start_x, start_y, end_x, end_y]
        patch_coords = np.zeros((n_patches_x * n_patches_y, 4), np.uint16)

        # fill in patch coordinates array (in order of L->R, top->bottom)
        k = 0
        y = 0
        for i in range(n_patches_y):
            x = 0
            for j in range(n_patches_x):
                # patch_coords[k] = [x, y, x + patch_w, y + patch_h]
                patch_coords[k] = [y, y + patch_h, x, x + patch_w]
                k += 1
                if j < (n_patches_x - 1):
                    x = x + patch_w - remainders_x[j]
            if i < (n_patches_y - 1):
                y = y + patch_h - remainders_y[i]

        patches = list()
        for patch in patch_coords:
            patches.append(img[:, patch[0]: patch[1], patch[2]: patch[3]])

        return patches, patch_coords
    if len(img.shape) == 2:
        img_h, img_w = img.size()[0], img.size()[1]
        patch_h, patch_w = patch_size, patch_size

        # # Make sure image not too small (center it in 0s in either direction if it is)
        if img_h < patch_h or img_w < patch_w:
            background = torch.zeros((max(img_h, patch_h), max(img_w, patch_w)))
            start_y = (patch_h - img_h) // 2 if img_h < patch_h else 0
            start_x = (patch_w - img_w) // 2 if img_w < patch_w else 0
            background[start_y:start_y + img_h, start_x:start_x + img_w] = img
            img = background
            img_h, img_w = img.size()[0], img.size()[1]

        # Number of patches
        n_patches_x = np.uint16(np.ceil(img_w / patch_w))
        n_patches_y = np.uint16(np.ceil(img_h / patch_h))

        # Total remainders
        remainder_x = n_patches_x * patch_w - img_w
        remainder_y = n_patches_y * patch_h - img_h

        # Set up remainders per patch
        remainders_x = np.uint16(np.ones((n_patches_x - 1)) * np.floor(remainder_x / (n_patches_x - 1)))
        remainders_y = np.uint16(np.ones((n_patches_y - 1)) * np.floor(remainder_y / (n_patches_y - 1)))
        remainders_x[0:np.remainder(remainder_x, n_patches_x - 1)] += 1
        remainders_y[0:np.remainder(remainder_y, n_patches_y - 1)] += 1

        # Initialize array of patch coordinates: rows = [start_x, start_y, end_x, end_y]
        patch_coords = np.zeros((n_patches_x * n_patches_y, 4), np.uint16)

        # fill in patch coordinates array (in order of L->R, top->bottom)
        k = 0
        y = 0
        for i in range(n_patches_y):
            x = 0
            for j in range(n_patches_x):
                # patch_coords[k] = [x, y, x + patch_w, y + patch_h]
                patch_coords[k] = [y, y + patch_h, x, x + patch_w]
                k += 1
                if j < (n_patches_x - 1):
                    x = x + patch_w - remainders_x[j]
            if i < (n_patches_y - 1):
                y = y + patch_h - remainders_y[i]

        patches = list()
        for patch in patch_coords:
            patches.append(img[patch[0]: patch[1], patch[2]: patch[3]])

        return patches, patch_coords


def read_seg_coord_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        list_of_lists = []
        for row in reader:
            list_of_tuples = [eval(item) for item in row]
            list_of_lists.append(list_of_tuples)
    return list_of_lists


# give an image as torch.Tensor/np.array (C, H, W) and coords as output from 'get_segmentation_coord_list'
# can also handle torch.Tensor/np.array (H, W) as input for a single channel image
def get_segmentation_crops(image: torch.Tensor, coord_list: list[tuple[int, int, int, int]]) -> list[list[torch.Tensor, int]]:
    crops = []
    for i, coord in enumerate(coord_list):
        y_min, y_max, x_min, x_max = coord
        if len(image.shape) == 3:
            crop_img = image[:, y_min:y_max, x_min:x_max]
        elif len(image.shape) == 2:
            crop_img = image[y_min:y_max, x_min:x_max]
        crops.append(crop_img)
    return crops


def get_masked_cells(image: torch.Tensor, coord_list: list[tuple[int,int,int,int]], masks: torch.Tensor) -> list[torch.Tensor]:
    masked_cells = []
    for coord in coord_list:
        y1, y2, x1, x2 = coord
        
        cell_mask = masks[y1:y2,x1:x2]                
        mask_h, mask_w = cell_mask.shape
        cell = cell_mask[(mask_h//2)-1, (mask_w//2)-1].item() #-1 to for 1 to 0 indexing
        
        bool_mask = (cell_mask == cell).unsqueeze(0)
        bool_mask = bool_mask.expand((image.shape[0], -1, -1))
        
        masked_cell = torch.zeros_like(bool_mask)
        cropped_image = image[:, y1:y2, x1:x2]
        
        masked_cell = torch.where(bool_mask, cropped_image, masked_cell)
        masked_cells.append(masked_cell)
        
    return masked_cells


def get_single_segmentation_crop(image: torch.Tensor, coord_list: list[tuple[int, int, int, int]], index: int) -> list[list[torch.Tensor, int]]:
    y1, y2, x1, x2 = coord_list[index]
    if len(image.shape) == 3:
        crop_img = image[:, y1:y2, x1:x2]
    elif len(image.shape) == 2:
        crop_img = image[y1:y2, x1:x2]
    
    #crop_img.sum() workaround for cell size; could still do single channel intensity as a proportion of total intensity
    #todo: change i to cell size by thresholding maybe?
    return crop_img, crop_img.sum()


def get_single_masked_cell(image: torch.Tensor, coord_list: list[tuple[int,int,int,int]], masks: np.ndarray, index: int) -> list[list[torch.Tensor, int]]:
    y1, y2, x1, x2 = coord_list[index]
            
    cell = masks == index + 1
    masked_cell = image * cell
    if len(masked_cell.shape) == 3:
        masked_cell = masked_cell[:,y1:y2,x1:x2]
    elif len(masked_cell.shape) == 2:
        masked_cell = masked_cell[y1:y2,x1:x2]
    return masked_cell, cell.sum()


def get_feature_cols(columns: pd.Index):
    new_columns = []
    first_col = -1

    for i, col in enumerate(columns):
        try:
            new_columns.append(int(col))
            if first_col < 0:
                first_col = i
        except ValueError:
            new_columns.append(col)
    num_features = sum([type(x)==int for x in new_columns])
    return first_col, num_features


def get_indices(seg_coords):
    '''
    Given list of lists of coordinates (images in rows, tuples of coordinates in columns),
    return dataset indices to access each segmented/masked cell.

    Ignores images where no cells are found.
    '''
    dataset_indices = []
    for image_index, segmented_cell_coords in enumerate(seg_coords):
        dataset_indices.extend([[image_index, cell_index] for cell_index, coord in enumerate(segmented_cell_coords) if coord])
    return np.array(dataset_indices)


def load_bool_csv(file_path):
    bool_list = []
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            bool_list.extend([value == 'True' for value in row])
    bool_tensor = torch.tensor(bool_list, dtype=torch.bool)
    return bool_tensor


def copy_image_zips(cfg):
    # copy and unzip images if needed
    study_path = Path(cfg['study'])
    desired_plates = [f"{cfg['study']}-{plate}-converted.zip" for plate in cfg['split_df']['local_plate'].unique()]
    if not study_path.exists():
        study_path.mkdir(parents=True, exist_ok=True)
        for file in tqdm([cfg['paths']['images_dir']/study_path/plate for plate in desired_plates], desc="Copying image .zip files"):
            shutil.copy(file, study_path)

        for file in tqdm(list(study_path.glob('*.zip')), desc="Unzipping image files"):
            with zipfile.ZipFile(file, 'r') as zip_ref:
                zip_ref.extractall(study_path / file.stem)
            file.unlink()


def disable_warnings():
    import warnings
    import logging
    warnings.filterwarnings("ignore", message="xFormers is available")
    logging.getLogger('cellpose').setLevel(logging.CRITICAL)