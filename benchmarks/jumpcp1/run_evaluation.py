import pandas as pd
import utils
import numpy as np
import pipelines
from tqdm import tqdm


def get_downstream_result(feature_extractor):
    replicate_feature = "Metadata_broad_sample"
    batch = "2020_11_04_CPJUMP1"
    experiment_df = (
        pd.read_csv("metadata/experiment-metadata.tsv", sep="\t")
        .query("Batch==@batch")
        .query("Density==100")
        .query('Antibiotics=="absent"')
    )

    experiment_df.drop(
        experiment_df[
            (experiment_df.Perturbation == "compound") & (experiment_df.Cell_line == "Cas9")
        ].index,
        inplace=True,
    )

    target1_metadata = pd.read_csv(
        "metadata/JUMP-Target-1_compound_metadata_additional_annotations.tsv",
        sep="\t",
        usecols=["broad_sample", "target_list"],
    ).rename(
        columns={
            "broad_sample": "Metadata_broad_sample",
            "target_list": "Metadata_target_list",
        }
    )

    perturbation_types = ['compound', 'crispr', 'orf']
    replicability_map_df = pd.DataFrame()
    replicability_fr_df = pd.DataFrame()
    matching_map_df = pd.DataFrame()
    matching_fr_df = pd.DataFrame()
    gene_compound_matching_map_df = pd.DataFrame()
    gene_compound_matching_fr_df = pd.DataFrame()
    compound_consensus_profiles = {}

    if feature_extractor == 'cellprofiler':
        data_inputs = ['normalized_feature_select_negcon_batch','bygroupfilt_cellpaint_robustized_featsel', 'bygroupfilt_all_robustized_featsel', 'bybatch_cellpaint_robustized_featsel', 
                        'bybatch_all_robustized_featsel', 'byplate_cellpaint_robustized_featsel', 'byplate_all_robustized_featsel']
        for se in [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
            data_inputs.append(f'bygroupfilt_cellpaint_spherized_featsel_{str(se)}')
            data_inputs.append(f'bygroupfilt_all_spherized_featsel_{str(se)}')
            data_inputs.append(f'bygroupfilt_cellpaint_spherized_{str(se)}')
    else:
        data_inputs = []
        for se in [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
            data_inputs.append(f'bygroupfilt_spherized_{str(se)}')

    for data_input in data_inputs:
        for cell_type in ["U2OS", "A549"]:
            cell_df = experiment_df.query("Cell_type==@cell_type")
            compound_consensus_profiles = {}
            for perturbation_type in perturbation_types:
                perturbation_df = cell_df.query("Perturbation==@perturbation_type")
                for timepoint in perturbation_df.Time.unique():
                    timepoint_df = perturbation_df.query(
                        "Time==@timepoint"
                    )
                    well_level_data = pipelines.load_data(timepoint_df, perturbation_type, feature_extractor, data_input)
                    replicability_map_df, replicability_fr_df = pipelines.replicability_pipeline(replicability_map_df, replicability_fr_df, well_level_data, perturbation_type, cell_type, timepoint, data_input)
                    if perturbation_type == 'compound':
                        matching_map_df, matching_fr_df, consensus_profiles = pipelines.matching_pipeline_compound(well_level_data, matching_map_df, matching_fr_df, replicability_map_df, perturbation_type, cell_type, timepoint, replicate_feature, data_input, target1_metadata)
                        compound_consensus_profiles[timepoint] = consensus_profiles
                    elif perturbation_type == 'crispr':
                        matching_map_df, matching_fr_df, gene_compound_matching_map_df, gene_compound_matching_fr_df = pipelines.matching_pipeline_gene_compound(well_level_data, gene_compound_matching_map_df, gene_compound_matching_fr_df, matching_map_df, matching_fr_df, replicability_map_df, perturbation_type, cell_type, timepoint, replicate_feature, data_input, compound_consensus_profiles)
                    elif perturbation_type == 'orf':
                        gene_compound_matching_map_df, gene_compound_matching_fr_df = pipelines.matching_pipeline_gene_compound(well_level_data, gene_compound_matching_map_df, gene_compound_matching_fr_df, matching_map_df, matching_fr_df, replicability_map_df, perturbation_type, cell_type, timepoint, replicate_feature, data_input, compound_consensus_profiles)


    replicability_fr_df.to_csv(f'fr_replicability_{feature_extractor}_group_results.csv', index = False)
    replicability_map_df.to_csv(f'map_replicability_{feature_extractor}_group_results.csv', index = False)
    matching_fr_df.to_csv(f'fr_matching_{feature_extractor}_group_results.csv', index = False)
    matching_map_df.to_csv(f'map_matching_{feature_extractor}_group_results.csv', index = False)
    gene_compound_matching_fr_df.to_csv(f'fr_genecompoundmatching_{feature_extractor}_group_results.csv', index = False)
    gene_compound_matching_map_df.to_csv(f'map_genecompoundmatching_{feature_extractor}_group_results.csv', index = False)


if __name__ == "__main__":
    feature_extractors = ['cellprofiler', 'cpcnn', 'dino4cells','openphenom_comp_chmean', 'openphenom_comp_allch']
    for feature_extractor in feature_extractors:
        get_downstream_result(feature_extractor)