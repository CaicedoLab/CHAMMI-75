import os
import os.path
from typing import Any, Callable, List, Optional, Tuple, Union
import numpy as np
import polars as pl
import skimage

from torchvision.transforms import ToTensor
from torchvision.datasets import VisionDataset

t = ToTensor()


def make_dataset(
    root: str,
    extensions: Optional[Union[str, Tuple[str, ...]]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
    metadata_path: str = None,
):
    metadata = pl.read_csv(os.path.join(root, metadata_path))
    # metadata = metadata.with_columns(pl.col("Metadata_broad_sample").cast(pl.Categorical).to_physical())
    return metadata


def fold_channels(image: np.ndarray, channel_width: int, mode="ignore"):
    # Expected input image shape: (h, w * c)
    # Output image shape: (h, w, c)
    output = np.reshape(image, (image.shape[0], channel_width, -1), order="F")
    # training = True
    if mode == "ignore":
        # Keep all channels
        pass
    elif mode == "drop":
        # Drop mask channel (last)
        output = output[:, :, 0:-1]
    elif mode == "apply":
        # Use last channel as a binary mask
        mask = output["image"][:, :, -1:]
        output = output[:, :, 0:-1] * mask

    return t(output)


def scikit_loader(path: str) -> np.ndarray:
    image = skimage.io.imread(path)
    return fold_channels(image, channel_width=160)


class SingleCellDataset(VisionDataset):
    """
    JUMP data loader. Assumes that images are stored as combined .npy files.

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
        self,
        root: str = None,
        extensions: Optional[Tuple[str, ...]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        metadata_path: str = None,
    ) -> None:
        super().__init__(
            root=root, transform=transform, target_transform=target_transform
        )
        samples = self.make_dataset(self.root, extensions, is_valid_file, metadata_path)
        self.root = root
        self.loader = scikit_loader
        self.extensions = extensions
        self.samples = samples
        self.metadata_path = metadata_path

    @staticmethod
    def make_dataset(
        directory: str,
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        metadata_path=None,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).

        This can be overridden to e.g. read files from a compressed zip file instead of from the disk.

        Args:
            directory (str): root dataset directory, corresponding to ``self.root``.
            class_to_idx (Dict[str, int]): Dictionary mapping class name to class index.
            extensions (optional): A list of allowed extensions.
                Either extensions or is_valid_file should be passed. Defaults to None.
            is_valid_file (optional): A function that takes path of a file
                and checks if the file is a valid file
                (used to check of corrupt files) both extensions and
                is_valid_file should not be passed. Defaults to None.

        Raises:
            ValueError: In case ``class_to_idx`` is empty.
            ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.
            FileNotFoundError: In case no valid file was found for any class.

        Returns:
            List[Tuple[str, int]]: samples of a form (path_to_sample, class)
        """
        return make_dataset(
            directory,
            extensions=extensions,
            is_valid_file=is_valid_file,
            metadata_path=metadata_path,
        )

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples.item(row=index, column="Image_Name"), 0
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample

    def __len__(self) -> int:
        return len(self.samples)
