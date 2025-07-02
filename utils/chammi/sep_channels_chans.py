import os
import multiprocessing
import polars as pl
import torch
import imageio
from tqdm import tqdm
import imageio.v3 as iio

def split_files(row):
    chammi_dir = './CHAMMI/'
    
    img_numpy = iio.imread(os.path.join(chammi_dir, row['file_path']))
    chan_width = row['channel_width']
    split_tensors = torch.split(torch.from_numpy(img_numpy), chan_width, dim=1)
    ext = row['file_path'].split('.')[-1]
    output_basedir = os.path.join("./split_CHAMMI", os.path.dirname(row['file_path']))
    os.makedirs(output_basedir, exist_ok=True)
    for idx, channel in enumerate(split_tensors):
        channel_path = row['file_path'].replace(f".{ext}", f"_{idx}.png")
        output_path = os.path.join("./split_CHAMMI", channel_path)
        imageio.imwrite(output_path, channel.numpy())

meta_data = pl.read_csv('./CHAMMI/combined_metadata.csv')
rows = list(meta_data.iter_rows(named=True))

if __name__ == "__main__":
    with multiprocessing.Pool(40) as p:
        list(tqdm(p.imap(split_files, rows), total=len(rows)))