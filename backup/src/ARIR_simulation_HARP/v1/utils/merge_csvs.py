import pandas as pd
import glob
import os
import re

def extract_task_id(filename):
    """
    Extract numeric task id from filename like:
    Generated_HOA_SRIR_data_task_12.csv
    """
    match = re.search(r'task_(\d+)', filename)
    return int(match.group(1)) if match else -1


def merge_csvs_sorted(input_dir, output_file="Generated_HOA_SRIR_data.csv"):

    # Find csv files
    csv_files = glob.glob(os.path.join(input_dir, "Generated_HOA_SRIR_data_task_*.csv"))

    if not csv_files:
        print("No CSV files found")
        return

    # Sort by task number
    csv_files = sorted(csv_files, key=extract_task_id)

    print("Merging files in order:")
    for f in csv_files:
        print(os.path.basename(f))

    # Read and merge
    df_list = [pd.read_csv(f) for f in csv_files]

    merged_df = pd.concat(df_list, ignore_index=True)
    # Round selected coordinate columns to 2 decimals
    cols_to_round = [
        "x_source", "y_source", "z_source",
        "x_mic", "y_mic", "z_mic"
    ]

    for col in cols_to_round:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].round(2)

    output_path = os.path.join(input_dir, output_file)
    merged_df.to_csv(output_path, index=False)

    print(f"\nMerged {len(csv_files)} files")
    print(f"Total rows: {len(merged_df)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    merge_csvs_sorted("../../../data/harp/R4000_P8_S3_Order3_24kHz_shoeBox_rtProfilesMIT_ISMOrder30/") # change if needed