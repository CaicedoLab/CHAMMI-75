import pandas as pd
import glob
import os
import argparse


def merge_csv_files(csv_files, common_column, include_columns, how="inner"):
    """
    Merges multiple CSV files on a common column, including only specific columns from each file.

    Parameters:
        folder_path (str): Path to folder containing CSV files.
        common_column (str): Column name to merge on.
        include_columns (list of str): Column names to keep (excluding the common column).
        how (str): Type of join - 'outer', 'inner', 'left', 'right'.
        suffix_base (str): Base string used for suffixes in case of overlapping column names.

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge CSV files in subdirectories.")
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/scr/jpeters/idr17_scores",
        help="Root directory containing CSV files",
    )
    parser.add_argument(
        "--csv_file_name",
        type=str,
        default="auc_roc_scores.csv",
        help="CSV file name to merge",
    )
    args = parser.parse_args()
    root_dir = args.root_dir
    csv_file_name = args.csv_file_name

    # Find all CSV files in subdirectories
    input_files = glob.glob(os.path.join(root_dir, "**", csv_file_name), recursive=True)
    output_file = os.path.join(root_dir, "merged_" + csv_file_name)

    # Merge CSV Files
    print(f"Found {len(input_files)} files to merge.")
    if len(input_files) == 0:
        raise FileNotFoundError("No csvs found to merge.")

    merged_df = merge_csv_files(
        csv_files=input_files,
        common_column="Mutation",
        include_columns=["Effect Size"],
        how="inner",
    )

    # Remove rows that contains certain cell lines
    # Remove rows where the 'Mutation' column contains any word related to 'PARENT' (case-insensitive)
    merged_df = merged_df[
        ~merged_df["Mutation"].str.contains("PAR", case=False, na=False)
    ]

    # Calculate averages of numeric columns
    numeric_cols = merged_df.select_dtypes(include="float64").columns
    averages = merged_df[numeric_cols].mean()
    print("Averages of numeric columns:")
    print(averages)

    # Add averages as the last row
    avg_row = {
        col: averages[col] if col in averages else "" for col in merged_df.columns
    }
    avg_row[merged_df.columns[0]] = "Average"
    merged_df = pd.concat([merged_df, pd.DataFrame([avg_row])], ignore_index=True)

    # Save the merged DataFrame to a new CSV Files
    merged_df.to_csv(output_file, index=False)
    print(f"Merged DataFrame saved to {output_file}")
