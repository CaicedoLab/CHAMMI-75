from .dataset_functions import FUNCTIONS
from dataclasses import dataclass, field
import typing
import os
from torch.utils.data import Dataset

@dataclass
class DatasetConfig:
    """Config dataclass outlining the structure of possible configuration variables
    Parameters:
    """
    data_path: str
    guided_crops_path: str = None
    guided_crops_size: tuple[int, int] = None
    split_fns: list[typing.Union[callable, str]] = field(default_factory=list)
    dataset_size: str = "small"  # "small" or "large". Used to determine the size of the dataset.
    num_procs: int = 1
    proc: int = 1
    seed: typing.Union[int, float, str, bytes] = None
    transform: callable = None
    img_type: str = ".png" # .png or .jpg or .jpeg. What torchvision.io.decode_image will support.
    test: bool = False # Set this to true, and the indexes of the files will be returned as the "classes"

    def __post_init__(self):
        self.data_path = os.path.abspath(os.path.expanduser(self.data_path))
        
        if len(self.split_fns) > 0 and any([isinstance(split_fn, str) for split_fn in self.split_fns]):
            self.split_fns = self.get_callables(self.split_fns)
            
        if self.guided_crops_path: # Assert only if we are given a path. Also checks if size exists.
            assert len(self.guided_crops_size) == 2, "You must include both the path and crop size to use guided cropping."
            assert isinstance(self.guided_crops_size[0], int), "First size entry in the crop size was not an int."
            assert self.guided_crops_size[0]%2==0, "First entry in the crop size was not divisible by 2."
            assert isinstance(self.guided_crops_size[1], int), "Second size entry in the crop size was not an int."
            assert self.guided_crops_size[1]%2==0, "Second entry in the crop size was not divisible by 2."
        elif self.guided_crops_size:
            assert self.guided_crops_path, "You must include both the path and crop size to use guided cropping."

    def get_callables(self, function_names: list[str]):
        # self.data_path = os.path.abspath(os.path.expanduser(self.data_path))
        # self.dataset = dataset
        return [FUNCTIONS[fn] for fn in function_names if isinstance(fn, str)]
    