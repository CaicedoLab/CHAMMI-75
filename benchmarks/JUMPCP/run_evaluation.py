import numpy as np
import pandas as pd
import os
from copairs import map
from copairs.matching import assign_reference_index
import argparse
import utils
import itertools

def run_phenotypic_activity(profiles, model, null_size, batch_size, fdr, output_folder):
    reference_col = "Metadata_reference_index"
    profiles_activity = assign_reference_index(
        profiles,
        "Metadata_control_type == 'negcon'",  # condition to get reference profiles (neg controls)
        reference_col=reference_col,
        default_value=-1,
    )
    pos_sameby = ["Metadata_broad_sample"]
    pos_diffby = []
    neg_sameby = ["Metadata_Plate"]
    neg_diffby = [reference_col]
    metadata = profiles_activity.filter(regex="^Metadata")
    if model == 'cellprofiler':
        brightfield_features = [i for i in profiles_activity.columns if 'ZBF' in i or 'Brightfield' in i]
        profiles_activity = profiles_activity.drop(columns = brightfield_features)
        profiles_features_only = profiles_activity.filter(regex="^(?!Metadata)").values
    else:
        profiles_features_only = profiles_activity.filter(regex="^emb").values

    activity_ap = map.average_precision(
        meta = metadata, 
        feats = profiles_features_only,
        pos_sameby = pos_sameby, 
        pos_diffby = pos_diffby,
        neg_sameby = neg_sameby,
        neg_diffby = neg_diffby,
        batch_size = batch_size
    )
    activity_ap = activity_ap.query("Metadata_control_type != 'negcon'")  # remove DMSO
    activity_map = map.mean_average_precision(
        activity_ap, pos_sameby, null_size=null_size, threshold=fdr, seed=0
    )
    os.makedirs(f'{output_folder}/{model}/', exist_ok=True)
    activity_map["-log10(p-value)"] = -activity_map["corrected_p_value"].apply(np.log10)    
    activity_map.to_csv(f"{output_folder}/{model}/phenotypic_activity_all_map.csv")
    active_ratio = activity_map.below_corrected_p.mean()
    mean_map = activity_map[activity_map.below_corrected_p]['mean_average_precision'].sum() / len(activity_map)
    activity_map.to_csv(f'{output_folder}/{model}/phenotypic_activity_map.csv')
    pd.DataFrame(columns = ['Active fraction', 'mean mAP (all - non-active = 0)', 'mean mAP (all)', 'mean mAP (active only)'], 
                 data = [[active_ratio, mean_map, activity_map['mean_average_precision'].mean(), 
                         activity_map[activity_map.below_corrected_p]['mean_average_precision'].mean()]]).to_csv(
                     f'{output_folder}/{model}/phenotypic_activity_result.csv',
                index=False)
    print(f"Phenotypic activity: fraction: {active_ratio}, mean mAP {mean_map}")
    return activity_map


def run_phenotypic_consistency(profiles, activity_map, model, null_size, batch_size, fdr, output_folder):
    multi_label_col = "Metadata_matching_target"
    active_compounds = activity_map.query("below_corrected_p")["Metadata_broad_sample"]
    consensus_profiles = profiles.query("Metadata_broad_sample in @active_compounds")
    if model == 'cellprofiler':
        brightfield_features = [i for i in consensus_profiles.columns if 'ZBF' in i or 'Brightfield' in i]
        feature_columns = [i for i in consensus_profiles.columns if 'Metadata' not in i]
    else:
        feature_columns = [i for i in consensus_profiles.columns if 'emb_' in i]
    
    columns = ["Metadata_broad_sample"] + feature_columns
    consensus_profiles = consensus_profiles[columns]
    consensus_profiles = consensus_profiles.groupby(["Metadata_broad_sample"], as_index=False)[feature_columns].median()

    total_targets = (
        profiles.merge(
            utils.read_metadata(), on="Metadata_broad_sample", how="left"
        )
        .assign(
            Metadata_matching_target=lambda x: x.Metadata_target_list.str.split("|")
        )
        .drop(["Metadata_target_list"], axis=1)
    ).Metadata_matching_target
    total_targets = len(set(list(itertools.chain.from_iterable(total_targets[total_targets.notna()].to_numpy()))))

    consensus_profiles = (
        consensus_profiles.merge(
            utils.read_metadata(), on="Metadata_broad_sample", how="left"
        )
        .assign(
            Metadata_matching_target=lambda x: x.Metadata_target_list.str.split("|")
        )
        .drop(["Metadata_target_list", "col"], axis=1)
    )

    metadata_df = consensus_profiles.filter(regex="^(Metadata)")
    if model == 'cellprofiler':
        brightfield_features = [i for i in consensus_profiles.columns if 'ZBF' in i or 'Brightfield' in i]
        consensus_profiles = consensus_profiles.drop(columns = brightfield_features)
        feature_values = consensus_profiles.filter(regex="^(?!Metadata)").values
    else:
        feature_values = consensus_profiles.filter(regex="^(emb)").values

    pos_sameby = [multi_label_col]
    pos_diffby = []
    neg_sameby = []
    neg_diffby = [multi_label_col]

    result = map.multilabel.average_precision(
        meta = metadata_df,
        feats = feature_values,
        pos_sameby = pos_sameby,
        pos_diffby = pos_diffby,
        neg_sameby = neg_sameby,
        neg_diffby = neg_diffby,
        batch_size=batch_size,
        multilabel_col=multi_label_col,
    )

    agg_result = map.mean_average_precision(
        result, pos_sameby, null_size, threshold=fdr, seed=0
    )
    consistent_ratio = agg_result.below_corrected_p.mean()
    consistent_true_fraction = agg_result.below_corrected_p.sum() / total_targets
    consistent_map = agg_result[agg_result.below_corrected_p]['mean_average_precision'].mean()
    overall_map = agg_result[agg_result.below_corrected_p]['mean_average_precision'].sum() / total_targets
    print(f"Phenotypic consistency: fraction: {consistent_ratio}, mean mAP for consistent passed targets {agg_result.mean_average_precision.mean()}, Overall mAP where not-passed and non-consistent targets mAP = 0 {overall_map}")

    pd.DataFrame(columns = ['Consistent fraction', 'Consistent fraction true', 'mean mAP for consistent passed targets', 'mean mAP for passed targets', 'mean mAP all; targets non-consistent and non-present = 0 '], 
                data = [[consistent_ratio, consistent_true_fraction, consistent_map, agg_result['mean_average_precision'].mean(), overall_map]]).to_csv(
                    f'{output_folder}/{model}/phenotypic_consistency_result.csv',
                index=False)

    agg_result.to_csv(f'{output_folder}/{model}/phenotypic_consistency_map.csv')

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feat_dir", type=str, default="./features/aggregated/", help="The directory that contains the aggregated features", required=True)
    parser.add_argument("--model", type=str, help="The type of model that is being trained and evaluated (mae, openphenom, dinov2 or vit)", required=True)
    parser.add_argument("--output_folder", type=str, default="./features/aggregated/", help="Output folder for aggregated features", required=False)
    parser.add_argument("--postfix", type=str, default="group_spherized_0.001", help="Postfix for aggregated features name", required=False)
    parser.add_argument("--fdr", default = 0.05, type=float, help="P-value threshold", required=False)
    parser.add_argument("--batch_size", default = 100000, type=int, help = "Batch size for copairs", required=False)
    parser.add_argument("--null_size_pa", default = 100000, type=int, help = "Null distribution sample size", required=False)
    parser.add_argument("--null_size_pc", default=20000, type=int, required=False)
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    profiles = utils.load_aggregated_profiles(args.feat_dir, args.postfix, args.model)
    activity_map = run_phenotypic_activity(profiles, args.model, args.null_size_pa, args.batch_size, args.fdr, args.output_folder)
    run_phenotypic_consistency(profiles, activity_map, args.model, args.null_size_pc, args.batch_size, args.fdr, args.output_folder)