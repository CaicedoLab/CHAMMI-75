# This code is reorganized from Chandrasekaran et al. 2024 Nature Methods paper
# Original repository: https://github.com/jump-cellpainting/2024_Chandrasekaran_NatureMethods/ 

import utils
import numpy as np
import pandas as pd
from typing import Optional

def load_data(timepoint_df: pd.DataFrame, 
              perturbation_type:str, 
              feature_extractor:str, 
              data_input:str):
    well_level_data = pd.DataFrame()
    for plate in timepoint_df.Assay_Plate_Barcode.unique():
    # Read all the plates
        if perturbation_type == 'compound':
            data_df = utils.load_data(
                feature_extractor, plate, f"{data_input}.csv.gz"
            ).assign(Metadata_modality=perturbation_type)
        elif perturbation_type in ['orf', 'crispr']:
            data_df = utils.load_data(feature_extractor, plate, f"{data_input}.csv.gz"
                ).assign(Metadata_modality=perturbation_type).assign(Metadata_matching_target=lambda x: x.Metadata_gene)
        else:
            raise ValueError('Incorrect perturbation type name')

        if data_input == 'spherized' and feature_extractor == 'cellprofiler':
            data_df = data_df[data_df.Metadata_Plate == plate].reset_index(drop=True)
        well_level_data = utils.concat_profiles(well_level_data, data_df)

    if perturbation_type == 'compound':
        well_level_data.loc[well_level_data.Metadata_pert_iname == 'DMSO', 'Metadata_broad_sample'] = 'DMSO'

    well_level_data = utils.remove_empty_wells(well_level_data)
    well_level_data["Metadata_negcon"] = np.where(
        well_level_data["Metadata_control_type"] == "negcon", 1, 0
    )
    #if brightfield_features_presence is False:
    #    brightfield_columns = [i for i in well_level_data.columns if 'Brightfield' in i ]
    #    zbf_columns = [i for i in well_level_data.columns if 'ZBF' in i ]
    #    well_level_data.drop(columns = brightfield_columns + zbf_columns, inplace = True)

    #if data_input == 'spherized':
    #    well_level_data.reset_index(inplace=True, drop=True)
    #    well_level_data = pd.merge(well_level_data, wells_nfsnb[plate], on = ['Metadata_Plate', 'Metadata_Well'], how = 'right')

    well_level_data.reset_index(inplace=True, drop=True)
    #if data_input == 'normalized_feature_select_negcon_batch':
    #    wells_nfsnb[plate] = pd.DataFrame(well_level_data[['Metadata_Plate', 'Metadata_Well']])

    return well_level_data#, wells_nfsnb


def load_data_deeplearning(timepoint_df: pd.DataFrame, 
                           perturbation_type:str, 
                           feature_extractor:str, 
                           data_input:str):
    
    well_level_data = pd.DataFrame()
    for plate in timepoint_df.Assay_Plate_Barcode.unique():
    # Read all the plates
        if perturbation_type == 'compound':
            data_df = utils.load_data(
                feature_extractor, plate, f"{data_input}.csv.gz"
            ).assign(Metadata_modality=perturbation_type)
        elif perturbation_type in ['orf', 'crispr']:
            data_df = utils.load_data(feature_extractor, plate, f"{data_input}.csv.gz"
                ).assign(Metadata_modality=perturbation_type).assign(Metadata_matching_target=lambda x: x.Metadata_gene)
        else:
            raise ValueError('Incorrect perturbation type name')
        
        well_level_data = utils.concat_profiles(well_level_data, data_df)
    
    if perturbation_type == 'compound':
        well_level_data.loc[well_level_data.Metadata_pert_iname == 'DMSO', 'Metadata_broad_sample'] = 'DMSO'

    well_level_data = utils.remove_empty_wells(well_level_data)
    well_level_data["Metadata_negcon"] = np.where(
        well_level_data["Metadata_control_type"] == "negcon", 1, 0
    )
    well_level_data.reset_index(inplace=True, drop=True)
    return well_level_data


def replicability_pipeline(replicability_map_df:pd.DataFrame,
                            replicability_fr_df:pd.DataFrame,
                            well_level_data: pd.DataFrame,
                            perturbation_type: str,
                            cell_type: str,
                            timepoint: int,
                            data_input: str):
    batch_size = 100000
    null_size = 100000

    pos_sameby = ["Metadata_broad_sample"]
    pos_diffby = []
    neg_sameby = ["Metadata_Plate"]
    neg_diffby = ["Metadata_negcon"]

    metadata_df = utils.get_metadata(well_level_data)
    feature_df = utils.get_featuredata(well_level_data)
    feature_values = feature_df.values

    result = utils.run_pipeline(
        metadata_df,
        feature_values,
        pos_sameby,
        pos_diffby,
        neg_sameby,
        neg_diffby,
        anti_match=False,
        batch_size=batch_size,
        null_size=null_size,
    )

    result = result.query("Metadata_negcon==0").reset_index(drop=True)

    replicability_map_df, replicability_fr_df = utils.create_replicability_df(
        replicability_map_df,
        replicability_fr_df,
        result,
        pos_sameby,
        0.05,
        perturbation_type,
        cell_type,
        timepoint,
        data_input,
        len(well_level_data)
    )

    return replicability_map_df, replicability_fr_df


def matching_pipeline_compound(well_level_profiles:pd.DataFrame,
                               matching_map_df:pd.DataFrame,
                               matching_fr_df:pd.DataFrame,
                               replicability_map_df: pd.DataFrame, 
                               perturbation_type: str,
                               cell_type: str,
                               timepoint: int,
                               replicate_feature:str,
                               data_input: str,
                               target1_metadata:pd.DataFrame):

    batch_size = 100000
    null_size = 100000
    # Remove DMSO wells
    well_level_profiles = utils.remove_negcon_and_empty_wells(well_level_profiles)

    # Create consensus profiles
    consensus_profiles = utils.consensus(well_level_profiles, replicate_feature)
    description = f"compound_{cell_type}_{utils.time_point('compound', timepoint)}"

    # Filter out non-replicable compounds
    replicable_compounds = list(
        replicability_map_df[
            (replicability_map_df.Description == description)
            & (replicability_map_df.above_q_threshold == True)
        ][replicate_feature]
    )
    consensus_profiles = consensus_profiles.query(
        "Metadata_broad_sample==@replicable_compounds"
    ).reset_index(drop=True)

    # Adding additional gene annotation metadata
    consensus_profiles = (
        consensus_profiles.merge(
            target1_metadata, on="Metadata_broad_sample", how="left"
        )
        .assign(
            Metadata_matching_target=lambda x: x.Metadata_target_list.str.split("|")
        )
        .drop(["Metadata_target_list"], axis=1)
    )

    # Calculate compound-compound matching
    pos_sameby = ["Metadata_matching_target"]
    pos_diffby = []
    neg_sameby = []
    neg_diffby = ["Metadata_matching_target"]

    metadata_df = utils.get_metadata(consensus_profiles)
    feature_df = utils.get_featuredata(consensus_profiles)
    feature_values = feature_df.values
    result = utils.run_pipeline(
        metadata_df,
        feature_values,
        pos_sameby,
        pos_diffby,
        neg_sameby,
        neg_diffby,
        anti_match=True,
        batch_size=batch_size,
        null_size=null_size,
        multilabel_col="Metadata_matching_target",
    )

    matching_map_df, matching_fr_df = utils.create_matching_df(
        matching_map_df,
        matching_fr_df,
        result,
        pos_sameby,
        0.05,
        perturbation_type,
        cell_type,
        timepoint,
        data_input
    )

    return matching_map_df, matching_fr_df, consensus_profiles


def matching_pipeline_gene_compound(well_level_profiles:pd.DataFrame,
                                    gene_compound_matching_map_df: pd.DataFrame, 
                                    gene_compound_matching_fr_df: pd.DataFrame, 
                                    matching_map_df:pd.DataFrame,
                                    matching_fr_df:pd.DataFrame,
                                    replicability_map_df: pd.DataFrame, 
                                    perturbation_type: str,
                                    cell_type: str,
                                    timepoint: int,
                                    replicate_feature:str,
                                    data_input: str,
                                    compound_consensus:dict):

    batch_size = 100000
    null_size = 100000
    # Remove DMSO wells
    well_level_profiles = utils.remove_negcon_and_empty_wells(well_level_profiles)
    description_gene = f"{perturbation_type}_{cell_type}_{utils.time_point(perturbation_type, timepoint)}"

    # Create consensus profiles
    consesus_profiles = utils.consensus(well_level_profiles, "Metadata_broad_sample")

    # Filter out non-replicable genes
    replicable_genes = list(
        replicability_map_df[
            (
                replicability_map_df.Description == description_gene
            )
            & (replicability_map_df.above_q_threshold == True)
        ][replicate_feature]
    )
    print(data_input, len(replicable_genes))
    consesus_profiles = consesus_profiles.query("Metadata_broad_sample==@replicable_genes").reset_index(drop=True)
    # Filter out reagents without a sister guide
    genes_without_sister = (
        consesus_profiles.Metadata_gene.value_counts()
        .reset_index()
        .query("Metadata_gene==1")["index"]
        .to_list()
    )
    consesus_profiles_for_matching = (
        consesus_profiles.query("Metadata_gene!=@genes_without_sister").reset_index(drop=True)
    )
    if perturbation_type == "crispr":
        if not matching_map_df.Description.str.contains(description_gene).any():

            pos_sameby = ["Metadata_matching_target"]
            pos_diffby = []
            neg_sameby = []
            neg_diffby = ["Metadata_matching_target"]

            metadata_df = utils.get_metadata(consesus_profiles_for_matching)
            feature_df = utils.get_featuredata(consesus_profiles_for_matching)
            feature_values = feature_df.values

            result = utils.run_pipeline(
                metadata_df,
                feature_values,
                pos_sameby,
                pos_diffby,
                neg_sameby,
                neg_diffby,
                anti_match=False,
                batch_size=batch_size,
                null_size=null_size,
            )

            matching_map_df, matching_fr_df = utils.create_matching_df(
                matching_map_df,
                matching_fr_df,
                result,
                pos_sameby,
                0.05,
                perturbation_type,
                cell_type,
                timepoint,
                data_input
            )

    # Filter out genes that are not perturbed by ORFs or CRISPRs
    perturbed_genes = list(
        set(consesus_profiles.Metadata_matching_target)
    )

    pos_sameby = ["Metadata_matching_target"]
    pos_diffby = ["Metadata_modality"]
    neg_sameby = []
    neg_diffby = ["Metadata_matching_target", "Metadata_modality"]

    for compound_timepoint in compound_consensus.keys():
        compound_filtered_genes = (
            compound_consensus[compound_timepoint][
                ["Metadata_broad_sample", "Metadata_matching_target"]
            ]
            .copy()
            .explode("Metadata_matching_target")
            .query("Metadata_matching_target==@perturbed_genes")
            .reset_index(drop=True)
            .groupby(["Metadata_broad_sample"])
            .Metadata_matching_target.apply(list)
            .reset_index()
        )

        compound_consensus_filtered = compound_consensus[compound_timepoint].drop(
            columns=["Metadata_matching_target"]
        ).merge(
            compound_filtered_genes,
            on="Metadata_broad_sample",
            how="inner",
        )

        # Calculate gene-compound matching mAP
        compound_genes = utils.concat_profiles(
            compound_consensus_filtered, consesus_profiles
        )

        metadata_df = utils.get_metadata(compound_genes)
        feature_df = utils.get_featuredata(compound_genes)
        feature_values = feature_df.values

        result = utils.run_pipeline(
            metadata_df,
            feature_values,
            pos_sameby,
            pos_diffby,
            neg_sameby,
            neg_diffby,
            anti_match=True,
            batch_size=batch_size,
            null_size=null_size,
            multilabel_col="Metadata_matching_target",
        )

        (
            gene_compound_matching_map_df,
            gene_compound_matching_fr_df,
        ) = utils.create_gene_compound_matching_df(
            gene_compound_matching_map_df,
            gene_compound_matching_fr_df,
            result,
            pos_sameby,
            0.05,
            'compound',
            perturbation_type,
            cell_type,
            compound_timepoint,
            timepoint,
            data_input
        )

    if perturbation_type == 'crispr':
        return matching_map_df, matching_fr_df, gene_compound_matching_map_df, gene_compound_matching_fr_df
    if perturbation_type == 'orf':
        return gene_compound_matching_map_df, gene_compound_matching_fr_df