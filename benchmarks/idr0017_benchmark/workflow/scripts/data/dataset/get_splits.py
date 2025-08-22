import pandas as pd
import numpy as np
import os
import random


def create_pivot(study: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a pivot table from a study's metadata DataFrame.

    This function processes a pandas DataFrame containing metadata for a study and generates a pivot table.
    The pivot table organizes the metadata such that each row represents a single image, and the final columns
    represent a different channel's filename.

    Parameters:
    study (pd.DataFrame): A DataFrame containing the study's metadata, single channel per row.

    Returns:
    pd.DataFrame: A pivot table DataFrame with metadata for images, including filenames for each individual channel.
    """
    groups = ["filename", "channel_bio", "study", "local_plate", "plate", "well", "fov", "z_slice", "timepoint", "control_type", "organism", "cell_line", "cell_type",
              "reagent_identifier", "split"]
    for group in groups:
        if group not in study.columns or pd.isna(pd.unique(study[group])).all():
            study[group] = 'unknown'
    
    aggregates = study.groupby(groups).count().reset_index()
    #print("Aggregated:",aggregates.shape, "Original", study.shape)
    pivot = aggregates.pivot_table(
        values="filename",
        index=["study", "local_plate", "plate", "well", "fov", "z_slice", "timepoint", "control_type", "organism", "cell_line", "cell_type", "reagent_identifier", "split",],
        columns="channel_bio", 
        aggfunc=lambda x:x #if type(x)==str else random.sample(x,1)
    ).reset_index()
    return pivot.apply(lambda x: x.explode() if x.name in study.channel.unique() else x)
    #pivot = pivot.apply(lambda x: x.explode() if x.name in study.channel.unique() else x)
    #print("Total single images",pivot.shape[0])


def determine_split(row, split_column, split_dict):
    if not row['split'] == '':
        return row['split']
    return split_dict[row[split_column]]

def assign_split(group):
    return np.random.choice(['train', 'val'], size=len(group), p=[0.5, 0.5])

def get_train_and_val(df: pd.DataFrame, split_column: str) -> pd.DataFrame:
    """
    Splits a DataFrame into training and validation sets based on a specified column.

    This function takes a pandas DataFrame containing study metadata and splits it into training and validation sets.
    The split is determined by the specified column. If the 'split' column already exists and contains 'train' and 'val' values,
    the function returns the DataFrame as is. Otherwise, it creates the 'split' column based on the unique values in the specified column.

    Parameters:
    df (pd.DataFrame): A DataFrame containing the study's metadata.
    split_column (str): The column name used to determine the split.

    Returns:
    pd.DataFrame: The DataFrame with an added 'split' column indicating 'train' or 'val' for each row.
    """
    if 'split' in df.columns and 'train' in df.split.unique() and 'val' in df.split.unique():
        return df

    if split_column == 'plate':
        unique_elements = list(df['plate'].unique())
        random.shuffle(unique_elements)
        df['split'] = ''
        split_dict = {element: 'train' if i < len(unique_elements) // 2 else 'val' for i, element in enumerate(unique_elements)}
        df['split'] = df.apply(determine_split, split_column='plate', split_dict=split_dict, axis=1)
        return df

    #assign half of each well to train/val (done individually for each plate)
    if split_column == 'well':
        df['split'] = ''
        grouped = df.groupby(['plate', 'well'])
        for name, group in grouped:
            indices = group.index.to_list()
            np.random.shuffle(indices)
            half = len(indices) // 2
            train_indices = indices[:half]
            val_indices = indices[half:]
            
            df.loc[train_indices, 'split'] = 'train'
            df.loc[val_indices, 'split'] = 'val'
        return df
        
    unique_elements = list(df[split_column].unique())
    random.shuffle(unique_elements)
    df['split'] = ''
    df.loc[df['control_type'] != 'unknown', 'split'] = df[df['control_type'] != 'unknown'].groupby('control_type').transform(assign_split)
    
    split_dict = {element: 'train' if i <= len(unique_elements) // 2 else 'val' for i, element in enumerate(unique_elements)}
    df['split'] = df.apply(determine_split, split_column=split_column, split_dict=split_dict, axis=1)

    return df


def main(study_meta: pd.DataFrame, split_col: str) -> pd.DataFrame:
    # ## Get channels for each image    
    #study = pd.read_csv(os.path.join(path, version, f"{study_id}_meta.csv"))
    pivot = create_pivot(study_meta)
    pivot = get_train_and_val(pivot, split_col)
    return pivot



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help = "Root path to metadata", default = "/mnt/cephfs/mir/jcaicedo/morphem/dataset/metadata/")
    parser.add_argument("--version", help = "Version of metadata to use", default = "v5")
    parser.add_argument("--study", help = "Study ID to read", required = True)
    parser.add_argument("--split-col", help = "Column of metadata to determine train/val split from", required = True)

    args = parser.parse_args()
    main(args.path, args.version, args.study, args.split_col)
