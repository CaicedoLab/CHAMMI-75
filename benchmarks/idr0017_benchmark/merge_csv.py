import pandas as pd
import glob
import os

def merge_csv_files(
    csv_files,
    common_column,
    include_columns,
    how='inner'
):
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

        model_name, fusion_type = file.split('/')[-3:-1]


        df = pd.read_csv(file)

        # Ensure common column is present
        if common_column not in df.columns:
            raise ValueError(f"Common column '{common_column}' not found in file: {file}")

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
    root_dir = '/home/MORGRIDGE/akazi/foundation_models/idr0017_benchmarking'  # Replace with your actual root directory
    input_files = glob.glob(os.path.join(root_dir, '**', 'roc_scores.csv'), recursive=True)
    output_file = 'merged_output_late.csv'

    merged_df = merge_csv_files(
    csv_files = input_files,
    common_column = "Mutation",
    include_columns=["Mean Distance"],
    how='inner'
)

    merged_df.to_csv(output_file, index=False)
    print(f"Merged DataFrame saved to {output_file}")




