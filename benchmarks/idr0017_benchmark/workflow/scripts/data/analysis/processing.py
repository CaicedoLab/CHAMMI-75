from utils import get_feature_cols, read_seg_coord_csv, load_bool_csv
import data
from pathlib import Path
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import pdist, cdist
from scipy.stats import ttest_ind
import torch
import csv
import json


def build_channel_filter(cfg: dict, channel: str, quantile: float=0.7) -> None:
        if cfg['segment']['segment']:
            # total_coords = get_segmentation_coord_list(cfg)
            total_coords = None
        else:
            total_coords = None
        dataset = data.StudyDataset(cfg['split_df'], '.', cfg['classification_column'], segment_coord_list=total_coords, mask_dir=cfg['paths']['mask_dir'])
        dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'])
        chan_index = cfg['split_df'].columns[13:].get_loc(channel)
        mitotic_nuclei_intensities = []
        total_images_size = dataset[0][0].shape[1] * dataset[0][0].shape[2] # H x W of [first][image] in dataset

        for images, infos in tqdm(dataloader, total=len(dataloader), desc=f"Filtering by {channel} intensity"):
            images = images.to(cfg['device'])
            if total_coords is not None:
                clssifications, sizes = infos
            else:
                clssifications, sizes = infos, torch.tensor([total_images_size]*images.shape[0])
            sizes = sizes.to(cfg['device'])
            sizes = sizes + 1 #there were still nans for some reason with 0.00001
            means = images[:, chan_index, :, :].sum(dim=(1,2)) / sizes
            mitotic_nuclei_intensities.append(means)

        mitotic_nuclei_intensities = torch.cat(mitotic_nuclei_intensities)
        mitotic_nuclei_intensities = torch.nan_to_num(mitotic_nuclei_intensities, nan=0.0)
        threshold = torch.quantile(mitotic_nuclei_intensities, quantile)
        labels = mitotic_nuclei_intensities > threshold

        if Path(cfg['out']['out_folder'], f'dataset_filter.csv').exists():
            previous_labels = load_bool_csv(Path(cfg['out']['out_folder'], f'dataset_filter.csv')).to(cfg['device'])
            labels = labels & previous_labels

        labels = labels.cpu().numpy()
        with open(Path(cfg['out']['out_folder'], f'dataset_filter.csv'), mode='w', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(labels)
        
        with open(Path(cfg['out']['out_folder'], f'applied_filters.csv'), mode='a', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow([channel])


class ConcatenateChannelFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, split_df: pd.DataFrame):
        self.split_df = split_df

    def fit(self, X, y=None):
        return self

    def transform(self, base_df: pd.DataFrame) -> pd.DataFrame:
        base_df = base_df.set_index(['study', 'local_plate', 'filename', 'img_id', 'patch_id']).sort_index()
        concatenated_features = []

        for idx, row in tqdm(self.split_df.iterrows(), desc="concatenating channel features", total=self.split_df.shape[0]):
            study = row.study
            local_plate = row.local_plate
            filenames = row[13:]

            # Collect features for each patch unique to each filename
            feature_list = []
            for filename in filenames:
                try:
                    # each appended object is dataframe with shape: [num_patches, num_features]
                    feature_list.append(base_df.loc[(study, local_plate, filename)])
                except KeyError: # filename not found
                    continue
            if feature_list:
                combined_features = pd.concat(feature_list, axis=1).reset_index()
                combined_features.columns = ['img_id', 'patch_id'] + list(range(combined_features.shape[1]-2))
                row_metadata = pd.concat([row[:13]] * combined_features.shape[0], axis=1).T.reset_index(drop=True)
                concatenated_features.append(pd.concat([row_metadata, combined_features], axis=1))
        final_df = pd.concat(concatenated_features).reset_index(drop=True)
        return final_df

class NormalizeFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        first_feature_col, total_features = get_feature_cols(df.columns)
        feature_columns = df.columns[first_feature_col:first_feature_col+total_features].to_list()

        grouped = df[df['control_type'] == 'negative control'].groupby('plate')
        stats = grouped[feature_columns].agg(['mean', 'std'])
        means = stats.xs('mean', axis=1, level=1).fillna(0)
        stds = stats.xs('std', axis=1, level=1).fillna(1)
        for col in tqdm(feature_columns, desc="Normalizing Features"):
            col_mean = means[col]
            col_std = stds[col]
            df[col] = (df[col] - df['plate'].map(col_mean)) / df['plate'].map(col_std)
        return df

# class FilterChannelIntensity(BaseEstimator, TransformerMixin):
#     def __init__(self, cfg: dict, channel: str, quantile: float=0.7):
#         self.cfg = cfg
#         self.channel = channel
#         self.quantile = quantile

#     def fit(self, X, y=None):
#         return self

#     def transform(self, df: pd.DataFrame) -> pd.DataFrame:
#         total_coords = read_seg_coord_csv(Path(self.cfg['out']['out_folder'], self.cfg['out']['total_coords']))
#         dataset = data.StudyDataset(self.cfg['split_df'], '.', self.cfg['classification_column'], segment_coord_list=total_coords, mask_dir=self.cfg['paths']['mask_dir'])
#         dataloader = DataLoader(dataset, batch_size=1, num_workers=15)
#         chan_index = self.cfg['split_df'].columns[13:].get_loc(self.channel)
#         mitotic_nuclei_intensities = []
#         indices = []

#         for image_idx, item in enumerate(tqdm(dataloader, total=len(dataloader), desc=f"Filtering by {self.channel} intensity")):
#             cells, infos = item
#             clssifications, sizes = zip(*infos)
#             cells = torch.cat(cells, dim=0)
#             sizes = torch.cat(sizes, dim=0) + 0.001 #there were still nans for some reason with 0.00001
#             means = (cells[:, chan_index, :, :].sum(dim=(1,2)) / sizes).tolist()
#             mitotic_nuclei_intensities.extend(means)
#             indices.extend([[image_idx, cell_idx] for cell_idx in range(cells.size(0))])
#         mitotic_nuclei_intensities = np.nan_to_num(np.array(mitotic_nuclei_intensities), nan=0)
#         np_indices = np.array(indices, dtype=np.int64)
#         threshold = np.quantile(mitotic_nuclei_intensities, self.quantile)
#         labels = mitotic_nuclei_intensities > threshold
#         with open(Path(self.cfg['out']['out_folder'], f'dataset_filter.csv'), mode='w', newline='') as file:
#             csv_writer = csv.writer(file)
#             csv_writer.writerow(labels)
#         return df.loc[labels]
#         #return mitotic_nuclei_intensities, labels, np_indices

class FilterTreatmentIntraGroupDist(BaseEstimator, TransformerMixin):
    """Performs hypothesis test on the distince distributions between images per treatment compared to distance distribution of negative controls. 
    Does this on a plate-by-plate level.
    """
    def __init__(self, treatment_col: str, significance_thresh: float=0.05):
        """
        Args:
            treatment_col (str): column in dataframe containing treatments we are interestd in testing
            significance_thresh (float, optional): statistical significance threshold to signify hits. Defaults to 0.05.
        """
        self.treatment_col = treatment_col
        self.significance_thresh = significance_thresh

    def fit(self, X, y=None):
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        first_feature_col, total_features = get_feature_cols(df.columns)
        feature_columns = df.columns[first_feature_col:first_feature_col+total_features].to_list()

        # Calculate distances for negative controls per plate
        negative_controls = df[df['control_type'] == 'negative control']
        negative_distances = {}
        for plate, group in tqdm(negative_controls.groupby('plate'), desc="Getting negative control distances"):
            group = group.copy()
            group = group.groupby('img_id')[feature_columns].mean()
            distances = pdist(group[feature_columns], metric='cosine')
            negative_distances[plate] = distances
        
        # Calculate distances for each treatment per plate
        total_iterations = sum(len(plate_group.groupby(self.treatment_col)) for plate, plate_group in df.groupby('plate'))
        with tqdm(total=total_iterations, desc='Processing plates and treatments') as pbar:
            treatment_distances_per_plate = {}
            for plate, plate_group in df.groupby('plate'):
                treatment_distances_per_plate[plate] = {}
                for treatment, treatment_group in plate_group.groupby(self.treatment_col):
                    if not treatment_group.empty:
                        treatment_group = treatment_group.copy()
                        treatment_group = treatment_group.groupby('img_id')[feature_columns].mean()
                        distances = pdist(treatment_group[feature_columns], metric='cosine')
                        treatment_distances_per_plate[plate][treatment] = distances
                    pbar.update()
        
        # Perform hypothesis tests comparing treatment distances to negative control distances
        hypothesis_test_results = {}
        for treatment in tqdm(df[self.treatment_col].unique(), desc="Calculating statistics"):
            treatment_t_stats = []
            treatment_p_vals = []
            for plate in df['plate'].unique():
                if treatment in treatment_distances_per_plate[plate]:
                    t_statistic, p_value = ttest_ind(treatment_distances_per_plate[plate][treatment], negative_distances[plate])

                    #Tried this and got the same number of hits with equal_var=True; more hits with equal_var=False
                    # t_statistic, p_value = ttest_ind_from_stats(mean1=np.mean(treatment_distances_per_plate[plate][treatment]), 
                    #                                             std1=np.std(treatment_distances_per_plate[plate][treatment]), 
                    #                                             nobs1=len(treatment_distances_per_plate[plate][treatment]), 
                    #                                             mean2=np.mean(negative_distances[plate]), 
                    #                                             std2=np.std(negative_distances[plate]), 
                    #                                             nobs2=len(negative_distances[plate]), 
                    #                                             equal_var=False)

                    treatment_t_stats.append(t_statistic)
                    treatment_p_vals.append(p_value)
            hypothesis_test_results[treatment] = np.average(np.nan_to_num(treatment_p_vals, nan=1))
        significance_dict = {k: v < self.significance_thresh for k, v in hypothesis_test_results.items()}
        filter_col = df[self.treatment_col].map(significance_dict).rename('hit')
        return pd.concat([df, filter_col], axis=1)
    

class FilterTreatmentInterGroupDist(BaseEstimator, TransformerMixin):
    """Performs hypothesis test on the distince distributions between images per treatment compared to distance distribution of negative controls. 
    Does this on a plate-by-plate level.
    """
    def __init__(self, treatment_col: str, significance_thresh: float=0.05):
        """
        Args:
            treatment_col (str): column in dataframe containing treatments we are interestd in testing
            significance_thresh (float, optional): statistical significance threshold to signify hits. Defaults to 0.05.
        """
        self.treatment_col = treatment_col
        self.significance_thresh = significance_thresh

    def fit(self, X, y=None):
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        first_feature_col, total_features = get_feature_cols(df.columns)
        feature_columns = df.columns[first_feature_col:first_feature_col+total_features].to_list()

        # Calculate distances for negative controls per plate
        negative_controls = df[df['control_type'] == 'negative control']
        negative_distances = {}
        for plate, group in tqdm(negative_controls.groupby('plate'), desc="Getting negative control distances"):
            group = group.copy()
            group = group.groupby('img_id')[feature_columns].mean()
            distances = pdist(group[feature_columns], metric='cosine')
            negative_distances[plate] = distances
        
        # Calculate distances for each treatment compared to each negative control per plate
        total_iterations = sum(len(plate_group.groupby(self.treatment_col)) for plate, plate_group in df.groupby('plate'))
        with tqdm(total=total_iterations, desc='Getting treatment distances') as pbar:
            treatment_distances_per_plate = {}
            for plate, plate_group in df.groupby('plate'):
                treatment_distances_per_plate[plate] = {}
                for treatment, treatment_group in plate_group.groupby(self.treatment_col):
                    if not treatment_group.empty:
                        plate_negative_controls = negative_controls[negative_controls['plate']==plate].copy()
                        plate_negative_controls = plate_negative_controls.groupby('img_id')[feature_columns].mean()

                        treatment_group = treatment_group.copy()
                        treatment_group = treatment_group.groupby('img_id')[feature_columns].mean()

                        distances = cdist(plate_negative_controls[feature_columns], treatment_group[feature_columns], metric='cosine').flatten()

                        treatment_distances_per_plate[plate][treatment] = distances
                    pbar.update()
        
        # Perform hypothesis tests comparing treatment distances to negative control distances
        hypothesis_test_results = {}
        for treatment in tqdm(df[self.treatment_col].unique(), desc="Calculating statistics"):
            treatment_t_stats = []
            treatment_p_vals = []
            for plate in df['plate'].unique():
                if treatment in treatment_distances_per_plate[plate]:
                    t_statistic, p_value = ttest_ind(treatment_distances_per_plate[plate][treatment], negative_distances[plate])

                    #Tried this and got the same number of hits with equal_var=True; more hits with equal_var=False
                    # t_statistic, p_value = ttest_ind_from_stats(mean1=np.mean(treatment_distances_per_plate[plate][treatment]), 
                    #                                             std1=np.std(treatment_distances_per_plate[plate][treatment]), 
                    #                                             nobs1=len(treatment_distances_per_plate[plate][treatment]), 
                    #                                             mean2=np.mean(negative_distances[plate]), 
                    #                                             std2=np.std(negative_distances[plate]), 
                    #                                             nobs2=len(negative_distances[plate]), 
                    #                                             equal_var=False)

                    treatment_t_stats.append(t_statistic)
                    treatment_p_vals.append(p_value)
            hypothesis_test_results[treatment] = np.average(np.nan_to_num(treatment_p_vals, nan=1))
        significance_dict = {k: bool(v < self.significance_thresh) for k, v in hypothesis_test_results.items()}
        filter_col = df[self.treatment_col].map(significance_dict).rename('hit')
        return pd.concat([df, filter_col], axis=1)


def build_processing_pipeline():
    #todo
    pass
