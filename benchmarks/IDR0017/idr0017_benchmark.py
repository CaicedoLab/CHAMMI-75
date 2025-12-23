import yaml
import pandas as pd
import glob
import os
import argparse


def merge_csv_files(csv_files, common_column, include_columns, how="inner"):
    """
    Merges multiple CSV files on a common column, including only specific columns from each file.

    Parameters:
        csv_files (list): List of CSV file paths to merge.
        common_column (str): Column name to merge on.
        include_columns (list of str): Column names to keep (excluding the common column).
        how (str): Type of join - 'outer', 'inner', 'left', 'right'.

    Returns:
        pd.DataFrame: Merged DataFrame with selected columns.
    """

    merged_df = None

    for i, file in enumerate(csv_files):
        model_name, fusion_type = file.split("/")[-3:-1]

        df = pd.read_csv(file)

        # Ensure common column is present
        if common_column not in df.columns:
            raise ValueError(
                f"Common column '{common_column}' not found in file: {file}"
            )

        # Keep only the common column and specified include_columns
        selected_cols = [common_column]
        if include_columns:
            selected_cols += [col for col in include_columns if col in df.columns]
        df = df[selected_cols]

        # Rename the include_columns to include model_name and fusion_type
        rename_map = {}
        for col in include_columns:
            if col in df.columns:
                rename_map[col] = f"{model_name}_{fusion_type}"
        df = df.rename(columns=rename_map)

        # Merge with suffixes if needed
        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(
                merged_df,
                df,
                on=common_column,
                how=how,
            )

    merged_df = merged_df.sort_values(by=common_column).reset_index(drop=True)
    return merged_df


def run_benchmark(config):

    # Import the test class
    from tests.gene_compound_roc.gene_compound_roc import GeneCompoundInteraction

    gene_compound_inter = GeneCompoundInteraction(config)
    gene_compound_inter.compute_study_effect_size()

    save_dir = gene_compound_inter.save_dir
    fusion_type = gene_compound_inter.fusion_type

    # Always calculate AUC-ROC
    config["benchmark_test"] = "auc_roc"
    gene_compound_inter.compute_study_auc()

    # Always calculate AUC-PR
    config["benchmark_test"] = "auc_pr"
    gene_compound_inter.compute_study_auc()

    # Always calculate Recall@50
    config["benchmark_test"] = "recall_50"
    gene_compound_inter.compute_study_auc()

    # Always calculate Recall@100
    config["benchmark_test"] = "recall_100"
    gene_compound_inter.compute_study_auc()

    # Merge all CSV files for final comparison
    merge_benchmark_results(save_dir, fusion_type)


def merge_benchmark_results(save_dir, fusion_type):
    """Merge all benchmark results into a single comprehensive CSV."""

    benchmark_types = ["auc_roc_scores.csv", "auc_pr_scores.csv", "recall_50_scores.csv", "recall_100_scores.csv"]

    for benchmark_file in benchmark_types:
        # Find all CSV files in subdirectories
        input_files = glob.glob(
            os.path.join(save_dir, "**", benchmark_file), recursive=True
        )
        output_file = os.path.join(save_dir, fusion_type, "merged_" + benchmark_file)

        if len(input_files) == 0:
            print(f"No {benchmark_file} files found to merge.")
            continue

        print(f"Found {len(input_files)} {benchmark_file} files to merge.")

        merged_df = merge_csv_files(
            csv_files=input_files,
            common_column="Mutation",
            include_columns=[col for col in pd.read_csv(input_files[0]).columns if col != "Mutation"],
            how="inner",
        )

        # Remove rows that contains certain cell lines
        merged_df = merged_df[
            ~merged_df["Mutation"].str.contains("PAR", case=False, na=False)
        ]

        # Calculate averages of numeric columns
        numeric_cols = merged_df.select_dtypes(include="float64").columns
        averages = merged_df[numeric_cols].mean()
        print(f"\nAverages for {benchmark_file}:")
        print(averages)

        # Add averages as the last row
        avg_row = {
            col: averages[col] if col in averages else "" for col in merged_df.columns
        }
        avg_row[merged_df.columns[0]] = "Average"
        merged_df = pd.concat([merged_df, pd.DataFrame([avg_row])], ignore_index=True)

        # Save the merged DataFrame to a new CSV
        merged_df.to_csv(output_file, index=False)
        print(f"Merged DataFrame saved to {output_file}\n")


if __name__ == "__main__":
    # Read config.yaml file
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    argparser = argparse.ArgumentParser(description="Run IDR0017 Benchmark")
    argparser.add_argument(
        "--features_dir",
        type=str,
        default=config["features_dir"],
        help="Path to the feature embeddings directory",
    )
    argparser.add_argument(
        "--metadata_path",
        type=str,
        default=config["metadata_path"],
        help="Path to the metadata CSV file",
    )
    argparser.add_argument(
        "--save_dir",
        type=str,
        default=config["save_dir"],
        help="Directory to save analysis results",
    )
    args = argparser.parse_args()
    
    config["features_dir"] = args.features_dir
    config["metadata_path"] = args.metadata_path
    config["save_dir"] = args.save_dir

    # Run the benchmark test
    run_benchmark(config)