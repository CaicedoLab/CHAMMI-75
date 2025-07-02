import torch
import os
import sys
import zipfile
import argparse
import time
from multiprocessing import Pool
from functools import partial
from torchvision.io import decode_image
from safetensors.torch import save_file
import logging
import json
import math
import csv

def can_tile_tensor(tensor: torch.Tensor, tile_width: int, tile_height: int):
    tensor_shape = tensor[0].shape
    assert (
        tensor_shape[1] % tile_width == 0
    ), "Tile width is not divisible by image width"
    assert (
        tensor_shape[0] % tile_height == 0
    ), "Tile height is not divisible by image width"


def get_all_tiles(tensor: torch.Tensor, tile_width: int, tile_height: int):
    """Returns a tensor in format batch, tile_number, tile_height, tile_width
    tensor: A torch tensor in with shape: batch, width, height
    """
    can_tile_tensor(tensor, tile_width, tile_height)

    tiles = []
    rows = tensor.split(tile_height, dim=1)
    for row in rows:
        tiles.extend(row.split(tile_width, dim=2))

    return torch.stack(tiles, dim=1)

def get_tile_image_dims(tile_width, tile_height, image_width, image_height):
    max_height = 45
    max_width = 45
    temp_tile_height = tile_height
    temp_tile_width = tile_width
    
    while True:
        temp_tile_height = tile_height
        temp_tile_width = tile_width
    
        if image_height % temp_tile_height != 0 or image_width % temp_tile_width != 0:
            while image_height % temp_tile_height != 0 and temp_tile_height < max_height:
                temp_tile_height += 1
            while image_width % temp_tile_width != 0 and temp_tile_width < max_width:
                temp_tile_width += 1
                
        if image_height % temp_tile_height != 0:
            image_height -= 1
        if image_width % temp_tile_width != 0:
            image_width -= 1
    
        if image_width % temp_tile_width == 0 and image_height % temp_tile_height == 0:
            break
        
    return temp_tile_width, temp_tile_height, image_width, image_height

def parse_zip_file(file_path:str, global_path:str, config: dict, tile_width:int, tile_height:int, output_dir:str, log:bool, save_csv:bool):
    zip_info = {
        "zip": file_path,
    }
    
    study = os.path.basename(output_dir)
    if config:
        config = config[study]
    
    ext = ".safetensors"
    if save_csv:
        ext = ".csv"
    new_name = zip_info["zip"].split(".")[0] + ext
    output_file = os.path.join(output_dir, new_name)
    
    if os.path.exists(output_file):
        return 0
        
    if os.path.isdir(global_path):
        archive = zipfile.ZipFile(os.path.join(global_path, file_path), "r")
    else:
        archive = zipfile.ZipFile(global_path, 'r')
        
    image_paths = [file for file in archive.infolist() if not file.is_dir()]

    images = {}
    
    plate_has_thresh = False    
    if config:
        plates = list(config.keys())
        
        for plate in plates:
            if plate in file_path:
                config = config[plate]
                plate_has_thresh = True
                break  
        if not plate_has_thresh:
            config = config['all']

    csv_file = [("filename", "plate", "study", "tile_height", "tile_width", "tiles")]
    starting_tile_width, starting_tile_height = (tile_width, tile_height)
    for idx, img_path in enumerate(image_paths):        
        tile_width, tile_height = (starting_tile_width, starting_tile_height)
        
        if img_path.filename.split(".")[-1] != "png":
            continue 

        img_bytes = bytearray(archive.read(img_path.filename))
        try:
            torch_buffer = torch.frombuffer(img_bytes, dtype=torch.uint8)
        except:
            if log:
                logging.log(logging.INFO, (file_path, img_path.filename, len(img_bytes), "Torch wouldn't buffer. Why? Empty Zip?"))
            continue
                         
        if plate_has_thresh:
            image_tensor = decode_image(torch_buffer)[0]
        else:
            image_tensor = decode_image(torch_buffer).to(torch.float32)[0]
        
        img_shape = image_tensor.shape
        tile_width, tile_height, img_width, img_height = get_tile_image_dims(
                                                                            tile_width, 
                                                                            tile_height, 
                                                                            image_tensor.shape[1], 
                                                                            image_tensor.shape[0]
                                                                            )
                
        if tuple(image_tensor.shape) != (img_height, img_width):
            image_tensor = image_tensor[:img_height, :img_width]
            img_shape = image_tensor.shape

        num_tiles_wide = img_shape[1] // tile_width
        num_tiles_tall = img_shape[0] // tile_height
        tiles: torch.Tensor = get_all_tiles(
            image_tensor.unsqueeze(0), tile_width, tile_height
        )[0]
        
        if config:
            filename = img_path.filename 
            image_config = None
            for key in config.keys():
                key_type = config[key]['keyword_type']
                base_filename = filename.split('.')[0]
                if key_type == "number":
                    try:
                        int(base_filename)
                        image_config = config[key]
                        break
                    except:
                        continue
                elif key_type == "replace":
                    if key in filename:
                        image_config = config[key]
                        break
                
            if not image_config:
                image_config = config['default']
                
            if image_config['threshold'] == 0:
                continue # Process next image. 
            
            if image_config['value_type'] == "pc":
                tile_maxes = tiles.amax(dim=(1,2))
                tile_mins = tiles.amin(dim=(1,2))
                tile_processed = tile_maxes - tile_mins
            else:
                tile_processed = tiles.to(torch.float32).mean(dim=(1, 2))
            
            torch_hist = torch.reshape(tile_processed, (num_tiles_tall, num_tiles_wide))        
            heatmap_data = torch_hist.tolist()
            tiled_width = len(heatmap_data[0])
            tiled_height = len(heatmap_data)
                
            values_with_coords = []
            tile_idx = 0
            for i, row in enumerate(heatmap_data):
                for j, tile_val in enumerate(row):
                    values_with_coords.append((tile_idx, tile_val))
                    tile_idx += 1
                    
            reverse = False
            if image_config["direction"] == "top":
                reverse = True
                
            values_with_coords.sort(key=lambda x: x[-1], reverse=reverse)
        
            top_k = image_config['threshold']
            if image_config["threshold_type"] == "%":
                top_k = math.ceil(tiled_height * tiled_width * (top_k / 100))

            top_coordinates = [str(tile) for (tile,val) in values_with_coords[:top_k]]
            csv_file.append((filename, file_path, study, tile_height, tile_width, ":".join(top_coordinates)))
        else:
            tile_means = tiles.mean(dim=(1, 2))
            tile_means_shaped = torch.reshape(tile_means, (num_tiles_tall, num_tiles_wide))
            images[os.path.join(file_path, img_path.filename)] = tile_means_shaped
        
    meta_data = {"idr": os.path.basename(output_dir)}
    meta_data.update(zip_info)

    if not save_csv:
        save_file(images, output_file, metadata=meta_data)
    else:
        with open(output_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(csv_file)            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process individual zip files or zip files in a directory")
    parser.add_argument(
        "-p", "--zip_path", type=str, help="Path to the directory containing zip files or an individual zip file"
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True, help="Output directory or file name"
    )
    parser.add_argument(
        "-c", "--config", type=str, help="Optional path to config file to use for generating good starting tiles"
    )  
    parser.add_argument(
       "-z", "--zips", type=int, default=4, help="Number of zips to processes in parallel (Multiprocessing pool size)"
    ) 
    parser.add_argument(
        "-t", "--torch_threads", type=int, default=8, help="Number of threads for torch.set_num_threads."
    )
    parser.add_argument(
        '-l', '--log', action="store_true", help="Log to the output directory"
    )
    parser.add_argument(
        '--csv', action="store_true", help="Generate csv files instead of heatmaps"
    )
    
    parser.add_argument("-x", "--tile_width", type=int, default=32, help="Tile width")
    parser.add_argument("-y", "--tile_height", type=int, default=32, help="Tile height")
    args = parser.parse_args()

    torch.set_num_threads(args.torch_threads)
    
    if args.log:
        logging.basicConfig(
            filename=f'{os.path.join(args.output, "logs.log")}', level=logging.INFO
        )
    
    if args.config:
        config_path = os.path.abspath(os.path.expanduser(args.config))
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = args.config

    zip_path = os.path.abspath(os.path.expanduser(args.zip_path))
    
    if os.path.isdir(zip_path):    
        files = os.listdir(zip_path)
        zip_files = [file for file in files if "zip" == file.split(".")[-1]]
        num_procs = args.zips
        global_output_dir = os.path.join(args.output, os.path.basename(zip_path))
    else:
        zip_files = [os.path.basename(args.zip_path)]
        num_procs = 1
        global_output_dir = os.path.join(args.output, zip_path.split('/')[-1].split('-')[0])
    
    
    os.makedirs(global_output_dir, exist_ok=True)
    with Pool(processes=num_procs) as P:
        start = time.time()
        ret = P.map(
            partial(
                parse_zip_file,
                global_path=zip_path,
                config=config,
                tile_width=args.tile_width,
                tile_height=args.tile_height,
                output_dir=global_output_dir,
                log=args.log,
                save_csv=args.csv
            ),
            zip_files,
        )
        end = time.time()
        logging.info(f"Finished with {args.zips}. Time took: {round((end-start)/60, 2)}")
