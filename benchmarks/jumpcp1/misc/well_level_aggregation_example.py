import torch
import numpy as np
import pandas as pd
import argparse
import os
from tqdm import tqdm

"""
Mean profile over single-cells in the well for DINO4Cells pth files
"""


def aggregate_profiles(plate_filename, profiles_folder, metadata_folder):
        feature_columns = ['emb_' + str(i) for i in range(384)]
        plate_metadata = pd.read_csv(os.path.join(metadata_folder, plate_filename + '.csv' ))
        plate_metadata.drop(columns = ['index', 'Nuclei_Location_Center_X', 'Nuclei_Location_Center_Y', 'Target', 'Metadata_broad_sample', 'val', 'Image_Name' ], inplace = True)
        plate_metadata['source'] = plate_filename.split('_')[0]
        plate_metadata['batch'] = '2020_11_04_CPJUMP1'
        plate_metadata[['plate', 'well', 'site']] = plate_metadata['Key'].str.split('/', n=3, expand=True)
        plate_metadata.drop(columns = ['Key'], inplace = True)
        features = pd.DataFrame(np.array(torch.load(os.path.join(profiles_folder, plate_filename + '.pth' ))[0]))
        assert len(features) == len(plate_metadata)
        profile_data = pd.concat((plate_metadata, features), axis = 1)
        profile_data.rename(columns = dict(zip([i for i in range(384)], feature_columns)), inplace = True, errors='raise')
        profile_data = profile_data.groupby(["source", "batch", "plate", "well"])[feature_columns].mean().reset_index()
        plate = plate_filename.split('_')[1]
        os.mkdir(f'./data/profiles_cpj1/2020_11_04_CPJUMP1/{plate}')
        profile_data.to_parquet(f'./data/profiles_cpj1/2020_11_04_CPJUMP1/{plate}/{plate}.parquet')


if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Aggregate DINOv1 features to well level profiles")
        parser.add_argument('--profiles', help = 'Path to folder with pth files of plate profiles')
        parser.add_argument('--metadata', help = 'Path to folder with csv metadata files of plate profiles')
        args = parser.parse_args()
        os.mkdir(f'./data/profiles_cpj1/2020_11_04_CPJUMP1/')
        plates = [i.split('.')[0] for i in os.listdir(args.profiles)]
        for plate in tqdm(plates):
                aggregate_profiles(plate, args.profiles, args.metadata)