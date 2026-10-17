import os
import glob
import pandas as pd
import torch
import re

# =========================
# CONFIG
# =========================
filepath = "/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/harp/R2000_P10_S5_Order4_24kHz_shoeBox_rtProfilesMIT/harp_dataset_05-8kHz_highpass_10-SCM_randRot_10-repeats_task_0.csv"

base_dir = os.path.dirname(filepath)
filename = os.path.basename(filepath)
match = re.match(r"(.+)_task_\d+\.(csv|pt)$", filename)
if not match:
    raise ValueError(f"Unexpected filename format: {filename}")
prefix = match.group(1)
print("base_dir =", base_dir)
print("prefix   =", prefix)

csv_pattern = os.path.join(base_dir, f"{prefix}_task_*.csv") 
pt_pattern = os.path.join(base_dir, f"{prefix}_task_*.pt")

# =========================
# AUTOMATIC OUTPUT NAME
# =========================
def extract_task_number(path):
    match = re.search(r"_task_(\d+)", os.path.basename(path))
    return int(match.group(1)) if match else -1


csv_files = sorted(glob.glob(csv_pattern), key=extract_task_number)
pt_files = sorted(glob.glob(pt_pattern), key=extract_task_number)

if csv_files:
    first_csv = os.path.basename(csv_files[0])
    prefix = first_csv.split("_task_")[0]

    merged_csv_path = os.path.join(base_dir, f"{prefix}.csv")
    merged_pt_path = os.path.join(base_dir, f"{prefix}.pt")
else:
    raise FileNotFoundError("No matching *_task_* files found!")

# =========================
# MERGE CSV FILES
# =========================
print(f"Found {len(csv_files)} CSV files")

df_list = []

for f in csv_files:
    print(f"Reading {f}")
    df = pd.read_csv(f, skip_blank_lines=False)
    df_list.append(df)

merged_df = pd.concat(df_list, ignore_index=True)

# ---------------------------------
# Write CSV manually
# Preserve blank lines
# ---------------------------------
with open(merged_csv_path, "w", encoding="utf-8", newline="") as out:

    # header
    out.write(",".join(merged_df.columns) + "\n")

    for _, row in merged_df.iterrows():

        # row originated from blank line
        if row.isna().all():
            out.write("\n")
            continue

        values = []

        for v in row:
            if pd.isna(v):
                values.append("")
            else:
                values.append(str(v))

        out.write(",".join(values) + "\n")

print(f"Saved merged CSV → {merged_csv_path}")

# =========================
# MERGE PT FILES
# =========================
print(f"Found {len(pt_files)} PT files")

X_list = []
Y_list = []

for f in pt_files:
    print(f"Loading {f}")

    data = torch.load(f, map_location="cpu")

    X_list.append(data["X"])
    Y_list.append(data["Y"])

X_merged = torch.cat(X_list, dim=0)
Y_merged = torch.cat(Y_list, dim=0)

torch.save(
    {
        "X": X_merged,
        "Y": Y_merged,
    },
    merged_pt_path,
)

print(f"Saved merged PT → {merged_pt_path}")