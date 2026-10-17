import os
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.stats import linregress
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from collections import Counter
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from utils import estimate_rt60

# -----------------------------
# Paths
# -----------------------------
rir_folder = "/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/harp/R2000_P10_S5_Order4_24kHz_shoeBox_source-constraints_smallGrid_adj-Room-Distr/RIRs"

csv_path = "/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/harp/R2000_P10_S5_Order4_24kHz_shoeBox_source-constraints_smallGrid_adj-Room-Distr/Generated_HOA_SRIR_data.csv"

output_csv = csv_path.replace(".csv", "_with_rt60.csv")



# -----------------------------
# Worker function
# -----------------------------
def process_file(args):
    name, rir_folder = args
    rir_path = os.path.join(rir_folder, name)

    if not os.path.exists(rir_path):
        raise FileNotFoundError(f"Missing RIR file: {name}")

    try:
        fs, rir = wavfile.read(rir_path)

        rir = rir.T.astype(np.float64)
        rir = rir / (np.max(np.abs(rir)) + 1e-12)

        rir_mono = rir[0] if rir.ndim > 1 else rir

        return estimate_rt60(rir_mono, fs)

    except Exception as e:
        print(f"[ERROR] {name}: {e}")
        return np.nan


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":

    df = pd.read_csv(csv_path)
    csv_names = df["Name"].astype(str).tolist()

    # -----------------------------
    # Validate against folder
    # -----------------------------
    all_files = set(os.listdir(rir_folder))

    missing_files = [n for n in csv_names if n not in all_files]
    valid_mask = [n in all_files for n in csv_names]

    print(f"Total CSV entries: {len(csv_names)}")
    print(f"Missing files: {len(missing_files)}")

    if missing_files:
        print("\nExample missing files (first 20):")
        print(missing_files[:20])

    # -----------------------------
    # Check duplicates
    # -----------------------------
    dupes = [k for k, v in Counter(csv_names).items() if v > 1]
    if dupes:
        print(f"\nDuplicate CSV entries: {len(dupes)}")
        print(dupes[:20])

    # -----------------------------
    # Prepare indexed inputs (preserve CSV order)
    # -----------------------------
    indexed_inputs = [
        (i, name)
        for i, name in enumerate(csv_names)
        if name in all_files
    ]

    rt60_values = [np.nan] * len(csv_names)

    # -----------------------------
    # Parallel processing
    # -----------------------------
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:

        futures = {
            executor.submit(process_file, (name, rir_folder)): i
            for i, name in indexed_inputs
        }

        for future in tqdm(as_completed(futures),
                           total=len(futures),
                           desc="Computing RT60 (parallel)"):

            i = futures[future]
            try:
                rt60_values[i] = future.result()
            except Exception as e:
                print(f"[FUTURE ERROR] index {i}: {e}")
                rt60_values[i] = np.nan

    # -----------------------------
    # Save output
    # -----------------------------
    df["rt60"] = rt60_values
    df.to_csv(output_csv, index=False)

    print(f"\nSaved: {output_csv}")