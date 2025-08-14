import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import polars as pl
import logging
import csv
from utils import fetch_embeddings_from_metadata, calculate_effect_size, get_hit_list_for_cell_line, create_gt_for_cell_line, Standard_Normalizer, WhiteningNormalizer
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from scipy.stats import pearsonr

class GeneCompoundInteraction:

    """Class to compute the effect size between mutations and reagents and 
        carry out different analysis on the effect size
        
        Attributes:
            config (dict): Configuration dictionary containing paths and parameters."""

    def __init__(self, config):
        
        # Initialize configuration of class from config file
        self.config = config
        self.features_dir = config['features_dir']
        self.save_dir = os.path.join(config['save_dir'], config['model_name'])
        self.metadata_path = config['metadata_path']
        self.model_name = config['model_name']
        self.mutations_list = config.get('mutations', None)
        self.fusion_type = config['fusion_type']
        self.replicate_list = ["Replicate_1", "Replicate_2"]
        self.metadata_df = pl.read_csv(self.metadata_path)
        self.distance_metric = config['distance_metric']
        
        # Get Unique list of Mutations to analyze
        if config['mutations'] is not None: self.mutations_list = config['mutations']
        else: self.mutations_list = self.metadata_df["biology.cell_line"].unique().to_list()

        # Get Unique List of Active Reagents to Analyze
        self.reagents_list = self.metadata_df.filter(pl.col("experiment.control") != "negative control")["experiment.reagent"].unique().to_list()

        # Make directory for the output
        os.makedirs(os.path.join(self.save_dir, self.fusion_type), exist_ok=True)

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # save the configuration to a file
        config_save_path = os.path.join(self.save_dir, self.fusion_type, "config.yaml")
        with open(config_save_path, 'w') as f:
            for key, value in config.items():
                f.write(f"{key}: {value}\n")


    

    # --------------------- COMPUTE EFFECT SIZE --------------------------

    def compute_study_effect_size(self):

        '''Compute the effect size for each mutation and reagent in the study.'''
        
        # Iterate through each mutation
        for mutation in self.mutations_list:
            
            # Platewise normalization
            if self.fusion_type == "early_fusion":
                os.makedirs(os.path.join(self.save_dir, "early_fusion"), exist_ok=True)
                self.early_fusion(mutation)

            # Cell line-wise normalization
            elif self.fusion_type == "late_fusion":
                os.makedirs(os.path.join(self.save_dir, "late_fusion"), exist_ok=True)
                self.late_fusion(mutation)

            # Throw Error for unknown fusion type
            else:
                raise ValueError(f"Unknown fusion type: {self.fusion_type['fusion_type']}")
    
    


    # --------------------- COMPUTE ROC CURVE --------------------------

    def compute_study_roc(self):

        '''Compute the ROC curve for each mutation in the study.'''

        # Create a csv file to save the ROC scores
        csv_file_path = os.path.join(self.save_dir, self.fusion_type, "roc_scores.csv")
        with open(csv_file_path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)

        # Iterate through each mutation
        for mutation in self.mutations_list:

            # Row to track ROC-AUC scores
            roc_score_list = [mutation]

            # Create a figure
            plt.figure(figsize=(10, 6))
            plot_save_path = os.path.join(self.save_dir, self.fusion_type, f"roc_curve_{mutation}.png")

            # Read effect size from csv files
            effect_size_csv = os.path.join(self.save_dir, self.fusion_type, f"{mutation}.csv")
            effect_size_df = pl.read_csv(effect_size_csv)

            # Get Ground truth for the mutation
            groud_truth_array = effect_size_df["Ground Truth"].to_numpy()

            # Get distance column names
            distance_columns = [col for col in effect_size_df.columns if col not in ["Reagent", "Ground Truth"]]

            # Iterate through each reagent
            for column in distance_columns:
                
                # Get the effect size for the replicate
                distance_array = effect_size_df[column].to_numpy()
                
                # normalize the distance array
                distance_array = (distance_array - np.min(distance_array)) / (np.max(distance_array) - np.min(distance_array))

                # Calculate AUC-ROC score
                auc_score = roc_auc_score(groud_truth_array, distance_array)

                # Calculate Recall and Precison AUC score
                precision, recall, thresholds = precision_recall_curve(groud_truth_array, distance_array)
                pr_auc_score = auc(recall, precision)

                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(groud_truth_array, distance_array, drop_intermediate=False)

                # Plot ROC curve
                plt.plot(fpr, tpr, label=f"{column} (AUC = {auc_score:.2f})")

                # Add to the list
                roc_score_list.append(auc_score)

            # Plot Settings
            plt.plot([0, 1], [0, 1], '--', label='Chance')
            plt.title(f"ROC Curve for {mutation}")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend()
            plt.grid()
            plt.savefig(plot_save_path)
            plt.close()
        
            # Write the ROC scores to the csv file
            with open(csv_file_path, mode='a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(roc_score_list)

        # Read the csv file again to add the distance columns
        with open(csv_file_path, mode='r') as csvfile:
            reader = csv.reader(csvfile)
            result_rows = list(reader)

        # Add the distance columns to the first row
        with open(csv_file_path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Mutation"] + distance_columns)
            writer.writerows(result_rows)


    # ------------------------- CALCULATE CORELATION ----------------------

    def compute_replicate_correlation(self):

        ''' Compute the correlation between the replicates for each mutation.'''

        if self.fusion_type != "late_fusion":
            raise ValueError("Correlation can only be computed for late fusion type.")
        
        # Create a directory to save the correlation results
        correlation_csv_path = os.path.join(self.save_dir, self.fusion_type, "replicate_correlation.csv")
        # Create the CSV file and write the header
        with open(correlation_csv_path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Mutation", "Spearman Correlation", "Pearson Correlation"])


        for mutation in self.mutations_list:

            # Read the effect size from the csv file
            effect_size_csv = os.path.join(self.save_dir, self.fusion_type, f"{mutation}.csv")
            effect_size_df = pl.read_csv(effect_size_csv)

            # Get the distance columns
            replicate_1_effect_size = effect_size_df[self.replicate_list[0]].to_numpy()
            replicate_2_effect_size = effect_size_df[self.replicate_list[1]].to_numpy()

            # Calculate the correlation between the replicates
            spearman_correlation, _ = spearmanr(replicate_1_effect_size, replicate_2_effect_size)
            pearson_correlation, _ = pearsonr(replicate_1_effect_size, replicate_2_effect_size)

            # Save the correlation to a csv file
            with open(correlation_csv_path, mode='a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([mutation, spearman_correlation, pearson_correlation])



            

            

  
            




        


            


    # ------------------------- EARLY FUSION ----------------------


    def late_fusion(self, mutation):

        if not os.path.exists(os.path.join(self.save_dir, "late_fusion", mutation + ".csv")):

            # Create a dictionary to hold the effect sizes for each reagent
            effect_sizes_dict = {reagent: {self.replicate_list[0]:[], self.replicate_list[1]:[]} for reagent in self.reagents_list}

            # Iterate through replicate
            for replicate in self.replicate_list:

                # Get the features for the mutation and replicate
                replicate_metadata = self.metadata_df.filter((pl.col("biology.cell_line") == mutation) 
                                                            & (pl.col("imaging.channel_type") == "nucleus")
                                                            & (pl.col("experiment.plate").str.contains(replicate)))
                
                # Get Unique Plates
                unique_plates = replicate_metadata["storage.zip"].unique().to_list()

                # Iterate through each plate
                for plate in unique_plates:

                    # Get the feature for the plate
                    plate_metadata = replicate_metadata.filter(pl.col("storage.zip") == plate)

                    # Get Control for the plate
                    plate_control_metadata = plate_metadata.filter((pl.col("experiment.control") == "negative control"))
                    plate_control_features = np.array(list(fetch_embeddings_from_metadata(self.features_dir, plate_control_metadata, model_name = self.model_name).values()))

                    # Get Treated metadata
                    plate_treated_metadata = plate_metadata.filter((pl.col("experiment.control") == "no") | (pl.col("experiment.control") == "positive control"))


                    # Get unique reagents in the plate
                    unique_reagents = plate_treated_metadata["experiment.reagent"].unique().to_list()

                    # Create Plate Normalizer to normalize the reagents in the plate
                    plate_normalizer = Standard_Normalizer(plate_control_features)

                    # Iterate through each reagent
                    for reagent in unique_reagents:

                        # Reagents Metadata
                        reagent_metadata = plate_treated_metadata.filter(pl.col("experiment.reagent") == reagent)   
                        reagent_features = np.array(list(fetch_embeddings_from_metadata(self.features_dir, reagent_metadata,model_name = self.model_name).values()))

                        # Calculate Effect Size
                        effect_size, img = calculate_effect_size(control = plate_control_features, 
                                                                treated = reagent_features, 
                                                                bins = 100, 
                                                                normalizer = plate_normalizer,
                                                                plot_save_dir=None, 
                                                                distance_matrix = self.distance_metric)
                        
                        # Save Effect Size to the Dictionary
                        effect_sizes_dict[reagent][replicate].append(effect_size)


            # Save Effect Sizes to CSV
            csv_filename = os.path.join(self.save_dir, "late_fusion", mutation + ".csv")
            with open(csv_filename, mode='w', newline='') as csvfile:
                writer = csv.writer(csvfile)

                writer.writerow(["Reagent"] + self.replicate_list + ["Mean Distance"] + ["Ground Truth"])

                # Read ground truth csv file
                gt_df = pl.read_csv("ground_truth.csv")

                # Write the effect sizes for each reagent
                for reagent in effect_sizes_dict:
                    
                    # Find the row where 'matched_drug_name' matches the reagent and get the value in the column for the current mutation
                    if reagent in gt_df["matched_drug_name"].to_list():
                        gt_value = int(gt_df.filter(pl.col("matched_drug_name") == reagent).select(mutation).to_numpy().flatten()[0])
                    else:gt_value = 0

                    # Write the row with the effect sizes and ground truth value
                    mean_effect_size = (np.mean(effect_sizes_dict[reagent][self.replicate_list[0]]) + np.mean(effect_sizes_dict[reagent][self.replicate_list[1]]))/2
                    row = [reagent] + [np.mean(effect_sizes_dict[reagent][self.replicate_list[0]])] + [np.mean(effect_sizes_dict[reagent][self.replicate_list[1]])] + [mean_effect_size] + [gt_value]
                    writer.writerow(row)




    # ------------------------- EARLY FUSION ----------------------

    def early_fusion(self, mutation):

        if not os.path.exists(os.path.join(self.save_dir, "early_fusion", mutation + ".csv")):

            # Create a dictionary to hold the effect sizes for each reagent
            effect_size_dict = {reagent: {"Early Merge":[]} for reagent in self.reagents_list}

            # Mutation Metadata
            mutation_metadata = self.metadata_df.filter((pl.col("biology.cell_line") == mutation)
                                                        & (pl.col("imaging.channel_type") == "nucleus"))
            
            # Mutation Control Metadata
            mutation_control_metadata = mutation_metadata.filter(pl.col("experiment.control") == "negative control")
            mutation_control_features = np.array(list(fetch_embeddings_from_metadata(self.features_dir, mutation_control_metadata, model_name = self.model_name).values()))

            # Normalization for the mutation control
            mutation_control_normalizer = WhiteningNormalizer(mutation_control_features)

            # Iterate through each reagent
            for reagent in tqdm(self.reagents_list):

                # Get the metadata for the reagent
                reagent_metadata = mutation_metadata.filter(pl.col("experiment.reagent") == reagent) 
            
                # Get the features for the reagent
                reagent_features = np.array(list(fetch_embeddings_from_metadata(self.features_dir, reagent_metadata, model_name = self.model_name).values()))

                # Calculate Effect Size
                effect_size, img = calculate_effect_size(control=mutation_control_features, 
                                                        treated=reagent_features, 
                                                        bins=100, 
                                                        normalizer=mutation_control_normalizer,
                                                        plot_save_dir=None, 
                                                        distance_matrix=self.distance_metric)

                # Save Effect Size to the Dictionary
                effect_size_dict[reagent]["Early Merge"].append(effect_size)




            # Load the ground truth data
            gt_df = pl.read_csv("ground_truth.csv")

            # Create a CSV file to save the effect sizes
            csv_filename = os.path.join(self.save_dir, "early_fusion", mutation + ".csv")

            with open(csv_filename, mode='w', newline='') as csvfile:
                writer = csv.writer(csvfile)

                # Write the header row
                writer.writerow(["Reagent", "Effect Size", "Ground Truth"])

                # Iterate through each reagent in the effect size dictionary
                for reagent in self.reagents_list:

                    # Find the row where 'matched_drug_name' matches the reagent and get the value in the column for the current mutation
                    if reagent in gt_df["matched_drug_name"].to_list():
                        gt_value = int(gt_df.filter(pl.col("matched_drug_name") == reagent).select(mutation).to_numpy().flatten()[0])
                    else:gt_value = 0

                    # Write the row with the effect sizes and ground truth value
                    row = [reagent] + [np.mean(effect_size_dict[reagent]["Early Merge"])] + [gt_value]
                    writer.writerow(row)

            

            

        








        
        
   
            


                    
                    

                    
        

                    
                    
                    
                    

                

            
            
            

        




            





        

    


    



        

        



    

