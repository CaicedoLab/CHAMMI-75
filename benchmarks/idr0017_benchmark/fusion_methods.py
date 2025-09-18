import os
import polars as pl
import numpy as np
from utils import WhiteningNormalizer, fetch_embeddings_from_metadata, calculate_effect_size
from tqdm import tqdm
import csv


def early_fusion(mutation, fusion_config):
    
    print("Computing Effect Size for Mutation:", mutation)

    if not os.path.exists(os.path.join(fusion_config["save_dir"], "early_fusion", mutation + ".csv")):

        # Create a dictionary to hold the effect sizes for each reagent
        effect_size_dict = {reagent: {"Early Merge":[]} for reagent in fusion_config['reagents_list']}

        # Mutation Metadata
        print("Read Metadata")
        metadata_df = fusion_config["metadata_df"]
        
        # Filter Metadata
        print("Filter metdata")
        mutation_metadata = metadata_df.filter((pl.col("biology.cell_line") == mutation)
                                             & (pl.col("imaging.channel_type") == "nucleus"))
        
        # Mutation Control Metadata
        mutation_control_metadata = mutation_metadata.filter(pl.col("experiment.control") == "negative control")
        print("Fetching control features")
        mutation_control_features = np.array(list(fetch_embeddings_from_metadata(fusion_config["features_dir"], mutation_control_metadata, model_name = fusion_config["model_name"]).values()))

        # Normalization for the mutation control
        print("Normalzing control features")
        mutation_control_normalizer = WhiteningNormalizer(mutation_control_features)

        # Iterate through each reagent
        for reagent in tqdm(fusion_config["reagents_list"]):

            # Get the metadata for the reagent
            reagent_metadata = mutation_metadata.filter(pl.col("experiment.reagent") == reagent) 
        
            # Get the features for the reagent
            reagent_features = np.array(list(fetch_embeddings_from_metadata(fusion_config["features_dir"], reagent_metadata, model_name = fusion_config["model_name"]).values()))

            # Calculate Effect Size
            effect_size, img = calculate_effect_size(control=mutation_control_features, 
                                                    treated=reagent_features, 
                                                    bins=100, 
                                                    normalizer=mutation_control_normalizer,
                                                    plot_save_dir=None, 
                                                    distance_matrix=fusion_config["distance_metric"])

            # Save Effect Size to the Dictionary
            effect_size_dict[reagent]["Early Merge"].append(effect_size)




        # Load the ground truth data
        gt_df = pl.read_csv("ground_truth.csv")

        # Create a CSV file to save the effect sizes
        csv_filename = os.path.join(fusion_config["save_dir"], "early_fusion", mutation + ".csv")

        with open(csv_filename, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write the header row
            writer.writerow(["Reagent", "Effect Size", "Ground Truth"])

            # Iterate through each reagent in the effect size dictionary
            for reagent in fusion_config["reagents_list"]:

                # Find the row where 'matched_drug_name' matches the reagent and get the value in the column for the current mutation
                if reagent in gt_df["matched_drug_name"].to_list():
                    gt_value = int(gt_df.filter(pl.col("matched_drug_name") == reagent).select(mutation).to_numpy().flatten()[0])
                else:gt_value = 0

                # Write the row with the effect sizes and ground truth value
                row = [reagent] + [np.mean(effect_size_dict[reagent]["Early Merge"])] + [gt_value]
                writer.writerow(row)
    
    return None   