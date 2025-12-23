import os
import torch
from typing import Tuple
from torch.utils.data import Dataset
from torchvision.io import read_image
import polars as pl
from safetensors.torch import load_file


# give an image as torch.Tensor/np.array (C, H, W) and coords as output from 'get_segmentation_coord_list'
# can also handle torch.Tensor/np.array (H, W) as input for a single channel image
def get_segmentation_crops(
    image: torch.Tensor, coord_list: list[tuple[int, int, int, int]]
) -> list[list[torch.Tensor, int]]:
    crops = []
    for i, coord in enumerate(coord_list):
        y_min, y_max, x_min, x_max = coord
        if len(image.shape) == 3:
            crop_img = image[:, y_min:y_max, x_min:x_max]
        elif len(image.shape) == 2:
            crop_img = image[y_min:y_max, x_min:x_max]
        crops.append(crop_img)
    return crops


class FeatureExtractionDataset(Dataset):
    def __init__(self, inputs_dir: str, transforms=None):
        self.inputs_dir = inputs_dir

        self.metadata = pl.read_csv(os.path.join(self.inputs_dir, "idr0017_meta.csv"))
        # Gives us all plate names, using list dir
        self.plate_names = [
            entry
            for entry in os.listdir(inputs_dir)
            if os.path.isdir(os.path.join(inputs_dir, entry))
        ]

        self.coords = {}
        self._load_all_coords()

        self.image_groups = self._get_all_image_groups()
        self.transform = transforms

    def _load_all_coords(self):
        for plate_name in self.plate_names:
            plate_dir = os.path.join(self.inputs_dir, plate_name)
            safetensor_file = [
                f for f in os.listdir(plate_dir) if f.endswith(".safetensors")
            ]

            if not safetensor_file:
                raise FileNotFoundError(f"No safetensors file found in {plate_dir}")

            if len(safetensor_file) > 1:
                raise ValueError(
                    f"Multiple safetensors files found in {plate_dir}, expected only one."
                )

            # Assuming one per plate
            plate_coords = load_file(os.path.join(plate_dir, safetensor_file[0]))

            if not plate_coords:
                raise ValueError(
                    f"Failed to load coordinates from {safetensor_file[0]} in {plate_dir}"
                )

            # Don't add plate prefix since keys already have the correct format
            for key, value in plate_coords.items():
                self.coords[key] = value

    """
    Get all image groups from the metadata. Filters images without a key in the coords dictionary
    """

    def _get_all_image_groups(self):
        images = self.metadata.select(
            pl.all().sort_by("imaging.channel").over("imaging.multi_channel_id")
        )

        all_image_groups = []

        for plate_name in self.plate_names:
            plate_metadata = images.filter(pl.col("storage.zip") == plate_name)

            if plate_metadata.height == 0:
                raise ValueError(f"No metadata entries found for plate {plate_name}")

            # Group by multi_channel_id within this plate
            plate_image_groups = list(
                plate_metadata.group_by("imaging.multi_channel_id")
            )

            # Filter check for every sub group
            for criteria, group in plate_image_groups:
                image_names = group["storage.filename"].to_list()
                found_mask = False

                # check if any image in this group has seg coordinates
                for image_name in image_names:
                    st_key = self._get_st_key(image_name, plate_name)
                    if st_key in self.coords:
                        found_mask = True
                        break
                if found_mask:
                    group_with_plate = group.with_columns(
                        pl.lit(plate_name).alias("plate_name")
                    )
                    all_image_groups.append(group_with_plate)
                else:
                    continue

        return all_image_groups

    def _get_st_key(self, image_name: str, plate_name: str = None) -> str:
        if plate_name is None:
            if "/" in image_name:
                return image_name
            else:
                raise ValueError(
                    "Plate name must be provided if image_name does not contain plate info."
                )

        # Clean up - remove plate prefix if it exists
        clean_image_name = image_name.replace(f"{plate_name}/", "")

        # Match the actual coordinate key format: idr0017-{plate_name}-converted/{image_name}
        key = f"idr0017-{plate_name}-converted/{clean_image_name}"
        return key

    """
    Read images and find the coordinates for a given plate
    """

    def _read_images_get_masks(self, image_names: list, plate_name: str):
        images = []
        segmented_image_name = None

        for image_name in image_names:
            # Construct full path - image_name might already include plate prefix
            if image_name.startswith(plate_name):
                # Remove plate prefix for file path construction
                clean_image_name = image_name.replace(f"{plate_name}/", "")
                image_path = os.path.join(self.inputs_dir, plate_name, clean_image_name)
            else:
                image_path = os.path.join(self.inputs_dir, plate_name, image_name)

            images.append(read_image(image_path))

            # Check if image has segmentation coords (check each image)
            st_key = self._get_st_key(image_name, plate_name)
            if st_key in self.coords and segmented_image_name is None:
                segmented_image_name = image_name

        # Make sure we found at least one image with segmentation coordinates
        if segmented_image_name is None:
            # Debug: print available keys to understand the mismatch
            available_keys = [k for k in self.coords.keys() if plate_name in k]
            # print(f"Available coordinate keys for {plate_name}: {available_keys}")
            # print(f"Looking for keys: {[self._get_st_key(name, plate_name) for name in image_names]}")
            raise ValueError(
                f"No segmentation coordinates found for any image in {plate_name}. Images: {image_names}"
            )

        return images, segmented_image_name

    def _get_abs_path(self, path: str):
        return os.path.abspath(os.path.expanduser(path))

    def __len__(self):
        return len(self.image_groups)

    """
    Dataset get item iterator
    """

    def __getitem__(self, index) -> Tuple[torch.Tensor, list, str]:
        group: pl.DataFrame = self.image_groups[index]
        image_names = list(group["storage.filename"])
        multi_channel_id = group["imaging.multi_channel_id"][0]
        plate_name = group["storage.zip"][0]

        images, segmented_image_name = self._read_images_get_masks(
            image_names, plate_name
        )
        images = torch.concat(images, dim=0)

        # Get coordinated using proper key
        # Make sure that the keys are correct
        st_key = self._get_st_key(segmented_image_name, plate_name)
        img_coords = self.coords[st_key]

        # y1, y2, x1, x2 coord order
        patches = get_segmentation_crops(images, img_coords)

        stacked = torch.stack(patches, dim=0)

        # Convert uint8 to float [0, 1] before any transforms
        patches_float = []
        for patch in patches:
            if patch.dtype == torch.uint8:
                patch = patch.float() / 255.0
            patches_float.append(patch)

        # Apply transforms to each patch individually
        if self.transform:
            transformed_patches = []
            for patch in patches_float:
                transformed_patch = self.transform(patch)  # Transform expects (C, H, W)
                transformed_patches.append(transformed_patch)
            stacked = torch.stack(transformed_patches, dim=0)

        # Maybe put some guardrails here

        return stacked, image_names, multi_channel_id, plate_name
