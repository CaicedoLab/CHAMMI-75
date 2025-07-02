import csv
import multiprocessing.pool
import os
import io
import torch
import polars as pl
from tqdm import tqdm
from zipfile import ZipFile
from safetensors.torch import save
from functools import partial
import multiprocessing
import sys

def main():
    dataset_root_dir = os.path.abspath(os.path.expanduser('/scr/jpeters/datasets/v2/'))

    image_paths = []
    for dirpath, dirnames, filenames in os.walk(os.path.abspath(os.path.expanduser(dataset_root_dir))):
        image_paths.extend([os.path.join(dirpath, filename) for filename in filenames if 'csv' in filename])
        
    output_dir = '/scr/jpeters/content_filtering'
    output_zip_path = os.path.join(output_dir, 'test.zip')
    mem_buff = io.BytesIO()   
    cpus = multiprocessing.cpu_count() - 5
    print(f"Running with {cpus} cpus")
    with multiprocessing.Pool(cpus) as p:
        results = list(tqdm(p.imap(get_safetensors_from_row, image_paths), total=len(image_paths), desc="Generating safetensors bytes"))    
        
    zip_path = '/scr/data/CHAMMIv2m.zip'
    
    dataset = ZipFile(zip_path)
    dataset_files = frozenset([file.filename for file in dataset.filelist if '.png' in file.filename])
    combined_results = {}
    for result in results: combined_results.update(result)
    unique_keys = set(combined_results.keys())
    
    print(f"We have {len(dataset_files)} files in the zip and {len(combined_results)} safetensors files.")
    merged = unique_keys.intersection(dataset_files)            
    print(f"After filtering, we have {len(merged)} keys.")

    with ZipFile(mem_buff, 'w') as out_zip:
        for og_filename in tqdm(merged, desc="Writing images to memory zip"):
            im_zip_path, bytes_safetensor =  combined_results[og_filename]   
            out_zip.writestr(zinfo_or_arcname=im_zip_path, data=bytes_safetensor)
    print("Writing to on disk zip")
    with open(output_zip_path, 'wb') as f:
        f.write(mem_buff.getvalue())
            
def get_safetensors_from_row(image_path: str):
    path_tensors = {}
    content_filters = pl.read_csv(image_path)
    for idx, image in enumerate(content_filters.iter_rows(named=False)):
        im_zip_path = os.path.join('CHAMMIv2m', image[2], image[1].replace('.zip', ''), image[0].replace('.png', '.safetensors'))
        original_zip_path = os.path.join('CHAMMIv2m', image[2], image[1].replace('.zip', ''), image[0])
        
        path_as_list = [int(item) for item in [image[-3], image[-2], *image[-1].split(':')]]
        tensor = {"data":torch.asarray(path_as_list)}
        bytes_safetensor = save(tensor)
        path_tensors[original_zip_path] = (im_zip_path, bytes_safetensor)
        # path_tensors.append((im_zip_path, original_zip_path, bytes_safetensor))
    
    return path_tensors
        
if __name__ == "__main__":
    main()