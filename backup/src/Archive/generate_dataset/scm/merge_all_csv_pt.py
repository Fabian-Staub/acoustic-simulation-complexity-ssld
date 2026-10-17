import os
import glob
import pandas as pd
import torch
import re
import gc
from collections import defaultdict

# =========================
# CONFIG
# =========================
base_dir = "/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/harp/R2000_P10_S5_Order4_24kHz_shoeBox_rtProfilesMIT_ISMOrder2+Late"

print("base_dir =", base_dir)

if not os.path.isdir(base_dir):
    raise NotADirectoryError(
        f"base_dir does not exist or is not a directory: {base_dir}"
    )

# =========================
# HELPERS
# =========================
def extract_prefix_and_task(path):
    """
    Extract prefix and task number from files like:

        something_task_0.csv
        something_task_12.pt

    Example:
        R2000_P10_S5_Order4_24kHz_shoeBox_rtProfilesMIT_task_0.csv

    returns:
        prefix = R2000_P10_S5_Order4_24kHz_shoeBox_rtProfilesMIT
        task_number = 0
    """
    filename = os.path.basename(path)

    match = re.match(r"(.+)_task_(\d+)\.(csv|pt)$", filename)
    if not match:
        return None, None

    prefix = match.group(1)
    task_number = int(match.group(2))

    return prefix, task_number


def sort_by_task_number(path):
    _, task_number = extract_prefix_and_task(path)
    return task_number if task_number is not None else -1


def delete_files_safely(files):
    """
    Delete files and print warnings if deletion fails.
    """
    if not files:
        print("No task files deleted")
        return

    print(f"Deleting {len(files)} task files...")

    for f in files:
        try:
            os.remove(f)
            print(f"Deleted: {f}")
        except Exception as e:
            print(f"WARNING: Could not delete {f}: {e}")


# =========================
# FIND ALL TASK FILES
# =========================
all_csv_files = glob.glob(os.path.join(base_dir, "*_task_*.csv"))
all_pt_files = glob.glob(os.path.join(base_dir, "*_task_*.pt"))

print(f"Raw CSV matches: {len(all_csv_files)}")
print(f"Raw PT matches: {len(all_pt_files)}")

csv_groups = defaultdict(list)
pt_groups = defaultdict(list)

for path in all_csv_files:
    prefix, task_number = extract_prefix_and_task(path)
    if prefix is not None:
        csv_groups[prefix].append(path)

for path in all_pt_files:
    prefix, task_number = extract_prefix_and_task(path)
    if prefix is not None:
        pt_groups[prefix].append(path)

all_prefixes = sorted(set(csv_groups.keys()) | set(pt_groups.keys()))

if not all_prefixes:
    raise FileNotFoundError(
        f"No matching *_task_* CSV or PT files found in {base_dir}"
    )

print(f"Found {len(all_prefixes)} prefix group(s):")
for prefix in all_prefixes:
    print("  ", prefix)

# =========================
# MERGE EACH PREFIX
# =========================
for prefix in all_prefixes:
    print("\n" + "=" * 80)
    print(f"Processing prefix: {prefix}")

    csv_files = sorted(csv_groups.get(prefix, []), key=sort_by_task_number)
    pt_files = sorted(pt_groups.get(prefix, []), key=sort_by_task_number)

    merged_csv_path = os.path.join(base_dir, f"{prefix}.csv")
    merged_pt_path = os.path.join(base_dir, f"{prefix}.pt")

    csv_merge_success = False
    pt_merge_success = False

    # =========================
    # MERGE CSV FILES
    # =========================
    if csv_files:
        print(f"Found {len(csv_files)} CSV files")

        df_list = []

        try:
            for f in csv_files:
                print(f"Reading CSV: {f}")
                df = pd.read_csv(f, skip_blank_lines=False)
                df_list.append(df)

                del df
                gc.collect()

            merged_df = pd.concat(df_list, ignore_index=True)

            with open(merged_csv_path, "w", encoding="utf-8", newline="") as out:
                out.write(",".join(merged_df.columns) + "\n")

                for _, row in merged_df.iterrows():
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

            if os.path.isfile(merged_csv_path):
                csv_merge_success = True
                print(f"Saved merged CSV → {merged_csv_path}")
            else:
                print(f"WARNING: merged CSV was not found after writing: {merged_csv_path}")

            del df_list
            del merged_df
            gc.collect()

        except Exception as e:
            print(f"ERROR while merging CSV files for prefix {prefix}: {e}")
            csv_merge_success = False

            try:
                del df_list
            except NameError:
                pass

            try:
                del merged_df
            except NameError:
                pass

            gc.collect()

    else:
        print("No CSV files for this prefix")

    # =========================
    # MERGE PT FILES
    # =========================
    if pt_files:
        print(f"Found {len(pt_files)} PT files")

        X_list = []
        Y_list = []

        try:
            for f in pt_files:
                print(f"Loading PT: {f}")
                data = torch.load(f, map_location="cpu")

                X_list.append(data["X"])
                Y_list.append(data["Y"])

                del data
                gc.collect()

            X_merged = torch.cat(X_list, dim=0)
            Y_merged = torch.cat(Y_list, dim=0)

            torch.save(
                {
                    "X": X_merged,
                    "Y": Y_merged,
                },
                merged_pt_path,
            )

            if os.path.isfile(merged_pt_path):
                pt_merge_success = True
                print(f"Saved merged PT → {merged_pt_path}")
            else:
                print(f"WARNING: merged PT was not found after writing: {merged_pt_path}")

            del X_list
            del Y_list
            del X_merged
            del Y_merged
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"ERROR while merging PT files for prefix {prefix}: {e}")
            pt_merge_success = False

            try:
                del X_list
            except NameError:
                pass

            try:
                del Y_list
            except NameError:
                pass

            try:
                del X_merged
            except NameError:
                pass

            try:
                del Y_merged
            except NameError:
                pass

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    else:
        print("No PT files for this prefix")

    # =========================
    # DELETE TASK FILES AFTER SUCCESSFUL MERGING
    # =========================
    files_to_delete = []

    if csv_merge_success:
        files_to_delete.extend(csv_files)
    else:
        if csv_files:
            print("CSV merge was not successful; CSV task files will NOT be deleted.")

    if pt_merge_success:
        files_to_delete.extend(pt_files)
    else:
        if pt_files:
            print("PT merge was not successful; PT task files will NOT be deleted.")

    delete_files_safely(files_to_delete)

    # Delete per-prefix variables.
    del csv_files
    del pt_files
    del files_to_delete
    gc.collect()

print("\nDone merging all prefix groups.")

# =========================
# FINAL CLEANUP
# =========================
del all_csv_files
del all_pt_files
del csv_groups
del pt_groups
del all_prefixes
gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()