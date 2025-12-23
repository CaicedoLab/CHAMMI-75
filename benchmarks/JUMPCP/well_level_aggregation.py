import numpy as np
import pandas as pd
import argparse
import os
import pycytominer
from tqdm import tqdm

"""
Mean profile over single-cells in the well for DINO4Cells pth files
"""

def aggregate_profiles(plate_filename, profiles_folder, metadata_folder, model, feature_size, feature_columns, output_folder):
    plate_metadata = pd.read_csv(os.path.join(metadata_folder, plate_filename + '.csv' ))
    plate_metadata.drop(columns = ['index', 'Nuclei_Location_Center_X', 'Nuclei_Location_Center_Y', 'Target', 'Metadata_broad_sample', 'val', 'Image_Name' ], inplace = True)
    plate_metadata['batch'] = '2020_11_04_CPJUMP1'
    plate_metadata[['plate', 'well', 'site']] = plate_metadata['Key'].str.split('/', n=3, expand=True)
    plate_metadata.drop(columns = ['Key'], inplace = True)

    if model in ['vit', 'dinov2', 'dinov3']:
        features = np.load(os.path.join(profiles_folder, f'{plate_filename}.npy' ))
        features = pd.DataFrame(np.load(os.path.join(profiles_folder, plate_filename + '.npy' )))
    if model in ['subcell', 'openphenom']:
        features = pd.DataFrame(np.load(os.path.join(profiles_folder, plate_filename + '.npz' ))['features'])

    assert len(features) == len(plate_metadata)
    profile_data = pd.concat((plate_metadata, features), axis = 1)
    profile_data.rename(columns = dict(zip([i for i in range(feature_size)], feature_columns)), inplace = True, errors='raise')
    profile_data = profile_data.groupby(["batch", "plate", "well"])[feature_columns].mean().reset_index()
    plate = plate_filename.split('.')[0]
    os.makedirs(f'{output_folder}/{model}/{plate}', exist_ok = True)
    profile_data.to_parquet(f'{output_folder}/{model}/{plate}/{plate}.parquet')


def normalize_features(plates, model, feature_columns, output_folder):
    group_df = pd.DataFrame()
    for plate in plates:
        cellprofiler_plate = pd.read_csv(f'./features/aggregated/cellprofiler/{plate}/{plate}_normalized_feature_select_negcon_batch.csv.gz')
        cellprofiler_plate = cellprofiler_plate[[i for i in cellprofiler_plate.columns if 'Metadata' in i]]
        dl_plate = pd.read_parquet(f'{output_folder}/{model}/{plate}/{plate}.parquet')
        dl_plate = pd.merge(cellprofiler_plate, dl_plate, how = 'left', left_on=['Metadata_Plate', 'Metadata_Well'], right_on = ['plate', 'well']).reset_index(drop=True)
        dl_plate.drop(columns = ['batch', 'plate', 'well'], inplace = True)
        if model == 'cp-cnn':
            dl_plate = pd.concat([dl_plate.drop(columns=['all_emb']), dl_plate['all_emb'].apply(pd.Series)], axis=1).reset_index(drop = True)
            dl_plate.rename(columns = dict(zip([i for i in range(672)], feature_columns)), inplace = True, errors='raise')
        
        dl_plate.to_csv(f'{output_folder}/{model}/{plate}/{plate}_raw.csv.gz', compression='gzip', index = False)
        group_df = pd.concat((group_df, dl_plate)).reset_index(drop = True)

    normalized_df = pycytominer.normalize(group_df, features = feature_columns, meta_features = 'infer', method = 'spherize', spherize_epsilon = 1e-3, samples = "Metadata_control_type == 'negcon'")
    for plate in normalized_df.Metadata_Plate.unique():
        to_save = normalized_df[normalized_df.Metadata_Plate == plate].reset_index(drop = True)
        to_save.to_csv(f'{output_folder}/{model}/{plate}/{plate}_group_spherized_0.001.csv.gz', compression='gzip', index = False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate features to well level profiles")
    parser.add_argument('--profiles', help = 'Path to folder with .npy files of plate profiles', required=False)
    parser.add_argument('--model', help = 'With which model features were obtained', required=True)
    parser.add_argument('--output_folder', help = 'Path to folder to save aggregated profiles', required=False, default='./features/aggregated/')
    parser.add_argument('--feature_size', help = 'Size of feature vector for the model', required=False, type=int, default=0)
    args = parser.parse_args()
    os.makedirs(f'./features/aggregated/{args.model}', exist_ok = True)
    feature_size = {'cp-cnn':672, 'vit':1920, 'dinov2':1920, 'dinov3':1920, 'openphenom':1920, 'subcell':4096}
    if args.feature_size == 0:
        args.feature_size = feature_size[args.model]
    feature_columns = ['emb_' + str(i) for i in range(args.feature_size)]
    plates = ['BR00117010', 'BR00117011', 'BR00117012', 'BR00117013', 'BR00117024', 'BR00117025', 'BR00117026'] # CP-JUMP1 compound plat
    
    if args.model != 'cp-cnn':
        for plate in tqdm(plates):
            aggregate_profiles(plate, args.profiles, './features/metadata', args.model, args.feature_size, feature_columns, args.output_folder)

    normalize_features(plates, args.model, feature_columns, args.output_folder)