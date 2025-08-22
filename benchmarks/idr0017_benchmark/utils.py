import os
from safetensors.torch import safe_open
import torch
import scipy
import numpy as np
import polars as pl
from tqdm import tqdm
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import scipy
from umap import UMAP
import seaborn as sns
from scipy.stats import wasserstein_distance
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import csv



def load_single_plate_features(path):
    safe_tensor = safe_open(path)
    return safe_tensor

def fetch_embeddings_from_metadata(embedding_path: str, metadata: pl.DataFrame, model_name, study_name = "idr0017"):

    embedding_dict = {}

    if model_name == "dinov2":

        for row in metadata.iter_rows(named=True):
            plate_id = row['storage.zip']
            image_id = row['imaging.multi_channel_id']
            image_name_1 = row['storage.filename']
            image_name_2 = image_name_1.replace("DAPI", "Cy3")
            plate_name = f"{study_name}-{plate_id}-converted_features.safetensors"
            plate_emb_path = os.path.join(embedding_path, plate_name)
            plate_emb_dict = safe_open(plate_emb_path, framework = "pt")
            channel_1 = plate_emb_dict.get_tensor(image_name_1).mean(dim = 0)
            channel_2 = plate_emb_dict.get_tensor(image_name_2).mean(dim = 0)
            image_emb = torch.cat([channel_1, channel_2])
            embedding_dict[image_id] = image_emb
        



    else:
        for idx, row in enumerate(metadata.iter_rows(named=True)):
            plate_id = row['storage.zip']
            image_id = row['imaging.multi_channel_id']
            plate_name = f"{study_name}-{plate_id}-converted_features.safetensors"
            plate_emb_path = os.path.join(embedding_path, plate_name)
            plate_emb_dict = safe_open(plate_emb_path, framework = "pt")
            image_emb = plate_emb_dict.get_tensor(image_id)
            embedding_dict[image_id] = image_emb

    return embedding_dict

def fetch_dinov2_embeddings_from_metadata(embedding_path: str, metadata: pl.DataFrame, study_name = "idr0017"):

    embedding_dict = {}

    for row in tqdm(metadata.iter_rows(named=True)):
        plate_id = row['storage.zip']
        image_id = row['imaging.multi_channel_id']
        image_name_1 = row['storage.filename']
        image_name_2 = image_name_1.replace("DAPI", "Cy3")
        plate_name = f"{study_name}-{plate_id}-converted_features.safetensors"
        plate_emb_path = os.path.join(embedding_path, plate_name)
        plate_emb_dict = safe_open(plate_emb_path, framework = "pt")
        
        channel_1 = plate_emb_dict.get_tensor(image_name_1).mean(dim = 0)
        channel_2 = plate_emb_dict.get_tensor(image_name_2).mean(dim = 0)

        image_emb = torch.cat([channel_1, channel_2])
        embedding_dict[image_id] = image_emb

    return embedding_dict



class WhiteningNormalizer(object):
    def __init__(self, controls, reg_param=1e-6):
        # Whitening transform on population level data
        self.mu = controls.mean(axis = 0)
        self.whitening_transform(controls - self.mu, reg_param, rotate=True)

        
    def whitening_transform(self, X, lambda_, rotate=True):
        C = (1/X.shape[0]) * np.dot(X.T, X)
        s, V = scipy.linalg.eigh(C)
        D = np.diag( 1. / np.sqrt(s + lambda_) )
        W = np.dot(V, D)
        if rotate:
            W = np.dot(W, V.T)
        self.W = W

    def normalize(self, X):
        return np.dot(X - self.mu, self.W)
    

class OpenPhenom_Normalization(object):

    def __init__(self, controls, reg_param=1e-6):
        self.mu = controls.mean(axis = 0)
        self.whitening_transform(controls - self.mu, reg_param, rotate=True)

class Standard_Normalizer(object):
    def __init__(self, controls):
        self.mu = controls.mean(axis = 0)
        self.std = controls.std(axis = 0)
        #print("shape of mu: ", self.mu.shape)
        #print("shape of std: ", self.std.shape)

    def normalize(self, X):
        return (X - self.mu) / self.std
    

from io import BytesIO

# Calculate Distribution of Embeddings
def get_self_distribution(embeddings, nbins, distance_metric='euclidean'):
    pairwise_distances = scipy.spatial.distance.pdist(embeddings, metric=distance_metric)
    return pairwise_distances

def get_cross_distribution(embeddings1, embeddings2, nbins, distance_metric='euclidean'):
    pairwise_distances = scipy.spatial.distance.cdist(embeddings1, embeddings2, metric=distance_metric).flatten()
    return pairwise_distances

def calculate_distance(distribution1, distribution2):
    distance = wasserstein_distance(distribution1, distribution2)
    return distance
    

def calculate_effect_size(control, treated, bins, normalizer,plot_save_dir=None, distance_matrix = "euclidean"):

    # Normalize
    control_features = normalizer.normalize(control)
    treated_features = normalizer.normalize(treated)

    # Get distributions
    control_distance_matrix = get_self_distribution(control_features, nbins=bins, distance_metric=distance_matrix)
    treated_distance_matrix = get_cross_distribution(treated_features, control_features, nbins=bins, distance_metric=distance_matrix)

    # Calculate distance 
    distance = wasserstein_distance(control_distance_matrix, treated_distance_matrix)


    if plot_save_dir is not None:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(control_distance_matrix, color='blue', label='Control-Control', fill=True, alpha=0.5)
        sns.kdeplot(treated_distance_matrix, color='red', label='Treated-Control', fill=True, alpha=0.5)
        plt.title("Distribution of Distance Matrices")
        plt.xlabel("Distance")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    else:
        img = None
    return distance, img

class ComputeEffectSize:

    def __init__(self, control_features, bins = 100, distance_metric='eucleidean'):
        self.control_features = control_features
        self.bins = bins
        self.distance_metric = distance_metric
        self.normalizer = WhiteningNormalizer(control_features)
        self.normalized_control_features = self.normalizer.normalize(control_features)
        
    def compute_effect_size(self, treated_features, plot_save_dir= None):

        ''' Distance between control and treated features'''
        pass

    def self_distribution(self, features):
        ''' Calculate the self distribution of the features '''
        return get_self_distribution(features, nbins=self.bins, distance_metric=self.distance_metric)
    
    def cross_distribution(self, features1, features2):
        ''' Calculate the cross distribution of the features '''
        return get_cross_distribution(features1, features2, nbins=self.bins, distance_metric=self.distance_metric)



# -----------------------------------------------

# HIT LIST FOR CELL LINE
def get_hit_list_for_cell_line(gt_csv, cell_line_name):

    # Read csv file
    gt_df = pl.read_csv(gt_csv)

    # Filter the dataframe for the cell line
    cell_line_gt_df = gt_df.filter(pl.col(cell_line_name) == True)

    # Get the list of reagents that are hits for this cell line
    hit_list = cell_line_gt_df["matched_drug_name"].unique().to_list()

    return hit_list 

#  CREATE GT FOR CELL
def create_gt_for_cell_line(cell_line_distance_csv, cell_line_hit_list):

    # Read CSV
    distance_df = pl.read_csv(cell_line_distance_csv)

    # Create the GT column
    gt = [1 if reagent in cell_line_hit_list else 0 for reagent in distance_df["Reagent"]]


    # Add the ground truth labels to the dataframe
    distance_df = distance_df.with_columns(pl.Series("ground_truth", gt))

    # return the distance column and gt column as lists along with the dataframe
    return distance_df["Distance"].to_list(), distance_df["ground_truth"].to_list(), distance_df


def merge_replicate_distance(replicate_1_csv, replicate_2_csv):
    
    # Read both replicate CSVs
    df1 = pl.read_csv(replicate_1_csv)
    df2 = pl.read_csv(replicate_2_csv)

    print("Shape of df1 before grouping: ", df1.shape)
    print("Shape of df2 before grouping: ", df2.shape)

    # Reduce the repetative reagents to single entry using the mean distance
    df1 = df1.group_by("Reagent").agg(pl.col("Distance").mean())
    df2 = df2.group_by("Reagent").agg(pl.col("Distance").mean())



    # Merge on 'Reagent' column, keeping only reagents present in both
    merged = df1.join(df2, on="Reagent", how="inner", suffix="_rep2")

    # Average the 'Distance' columns from both replicates
    merged = merged.with_columns(
        ((pl.col("Distance") + pl.col("Distance_rep2")) / 2).alias("Distance_mean")
    )

    # Rename the column Distance
    merged = merged.rename({"Distance": "Distance_rep1"})

    # Rename the 'Distance_mean' column to 'Distance'
    merged = merged.rename({"Distance_mean": "Distance"})

    return merged



def plot_umaps(features_dir, meta_data, column_name, control_metadata=None):

    # Unique names in column_name
    unique_column_values = meta_data[column_name].unique().to_list()

    # Fetch the featurs of the parent cell line
    features_array = np.array(list(fetch_embeddings_from_metadata(features_dir, meta_data).values()))
    print("Shape of Features: ", features_array.shape)

    # If control metadata is provided, normalize the features
    if control_metadata is not None:
        # Fetch control features
        control_features = np.array(list(fetch_embeddings_from_metadata(features_dir, control_metadata).values()))
        print("Shape of Control Features: ", control_features.shape)

        # Initialize the normalizer
        normalizer = WhiteningNormalizer(control_features)

        # Normalize the features
        control_features = normalizer.normalize(control_features)
        features_array = normalizer.normalize(features_array)
        print("Shape of Normalized Features: ", features_array.shape)

    # Fit UMAP model
    umap_model = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', random_state=42)
    umap_model.fit(features_array)

    # Createa a figure for plotting
    plt.figure(figsize=(10, 8))



    # Iterate over unique values in the specified column and plot the UMAP embedding
    for value in unique_column_values:

        # Filter metadata for the current value
        filtered_metadata = meta_data.filter(pl.col(column_name) == value)

        # Fetch features for the filtered metadata
        plate_features = np.array(list(fetch_embeddings_from_metadata(features_dir, filtered_metadata).values()))

        # Normalize the features if control metadata is provided
        if control_metadata is not None:
            plate_features = normalizer.normalize(plate_features)
        
        # Transform the features using the UMAP model
        umap_points = umap_model.transform(plate_features)

        # Plot the UMAP points for this value
        plt.scatter(umap_points[:, 0], umap_points[:, 1], label=value, alpha=0.6, s=20)


    # If control metadata is provided, plot the control points
    if control_metadata is not None:
        control_umap_points = umap_model.transform(control_features)
        plt.scatter(control_umap_points[:, 0], control_umap_points[:, 1], label='Control', alpha=0.6, s=20, color='gray')
    
    plt.title(f'UMAP Embedding')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend()
    plt.tight_layout()
    plt.show()

