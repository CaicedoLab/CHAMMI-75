import pandas as pd
import os

def read_metadata():
    target1_metadata = pd.read_csv(
        "metadata/JUMP-Target-1_compound_metadata_additional_annotations.tsv",
        sep="\t",
        usecols=["broad_sample", "target_list"],
    ).rename(
        columns={
            "broad_sample": "Metadata_broad_sample",
            "target_list": "Metadata_target_list",
        }
    ).dropna().assign(col=lambda x: list(x['Metadata_target_list'].str.split("|")))
    return target1_metadata


def remove_empty_wells(df):
    """return dataframe of non-empty wells"""
    df = df.dropna(subset=["Metadata_broad_sample"]).reset_index(drop=True)
    return df


def load_aggregated_profiles(feat_dir, postfix, model):
    plates = ['BR00117010', 'BR00117011', 'BR00117012', 'BR00117013', 'BR00117024', 'BR00117025', 'BR00117026']
    profiles = pd.DataFrame()
    for plate in plates:
        profiles = pd.concat((profiles, pd.read_csv(os.path.join(feat_dir, model, plate, f"{plate}_{postfix}.csv.gz" ))))
    
    profiles = profiles.reset_index(drop = True)
    profiles.loc[profiles.Metadata_pert_iname == 'DMSO', 'Metadata_broad_sample'] = 'DMSO'
    return profiles
