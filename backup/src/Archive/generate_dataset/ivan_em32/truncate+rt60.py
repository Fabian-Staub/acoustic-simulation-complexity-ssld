import os
import numpy as np
import pandas as pd
from scipy.io import wavfile

import pyfar as pf
import pyrato as ra


# -----------------------------
# Settings
# -----------------------------
input_folder = "/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/arni_em32/sh_wavs/RIRs"

output_folder = "/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/arni_em32/sh_wavs/RIRs_truncated"

csv_path = "/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/arni_em32/sh_wavs/arni_directions.csv"

output_csv_path = "/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/arni_em32/sh_wavs/arni_directions_rt60.csv"

fade_len = 10
tail_offset = 300

# -----------------------------
# NEW: pre-peak trimming
# -----------------------------
pre_peak_cut = 200   # samples BEFORE peak (changeable)

os.makedirs(output_folder, exist_ok=True)


# -----------------------------
# Helpers
# -----------------------------
def to_scalar(x):
    return float(np.asarray(x).squeeze())


def apply_fade(rir, fade_in_len, fade_out_start, fade_out_len):
    n = rir.shape[0]

    fade_in = np.ones(n)
    fade_in[:fade_in_len] = np.linspace(0, 1, fade_in_len)

    fade_out = np.ones(n)
    start = max(fade_out_start, 0)
    end = min(fade_out_start + fade_out_len, n)

    if start < end:
        fade_out[start:end] = np.linspace(1, 0, end - start)
        fade_out[end:] = 0.0

    window = fade_in * fade_out
    return rir * window[:, None], window


# -----------------------------
# Load CSV
# -----------------------------
df = pd.read_csv(csv_path)
df["Name"] = df["Name"].astype(str)

rt60_values = []

# -----------------------------
# Cache for fallback RT60
# -----------------------------
prev_inter_time = np.nan
prev_rt60 = np.nan


# -----------------------------
# Processing loop
# -----------------------------
for _, row in df.iterrows():

    fname = row["Name"]
    in_path = os.path.join(input_folder, fname)
    out_path = os.path.join(output_folder, fname)

    if not os.path.exists(in_path):
        print(f"[WARN] Missing file: {fname}")
        rt60_values.append(np.nan)
        continue

    fs, rir_full = wavfile.read(in_path)
    rir_full = rir_full.astype(np.float32)

    if rir_full.ndim == 1:
        rir_full = rir_full[:, None]

    # ---------------------------------------------------------
    # NEW: PEAK DETECTION + PRE-PEAK CUT
    # ---------------------------------------------------------
    peak_idx_full = int(np.argmax(np.abs(rir_full[:, 0])))

    cut_start = max(0, peak_idx_full - pre_peak_cut)

    rir = rir_full[cut_start:, :].copy()

    # shift peak index into new coordinate system
    peak_idx = peak_idx_full - cut_start

    # ---------------------------------------------------------
    # RT60 estimation (robust)
    # ---------------------------------------------------------
    rir_sig = pf.Signal(rir[:, 0], fs)

    try:
        inter_time, rt60, _ = ra.edc.intersection_time_lundeby(rir_sig)

        inter_time = to_scalar(inter_time)
        rt60 = to_scalar(rt60)

        prev_inter_time = inter_time
        prev_rt60 = rt60

    except Exception as e:
        print(f"[WARN] RT60 estimation failed for {fname}: {e}")
        print("       -> using previous values")

        inter_time = prev_inter_time
        rt60 = prev_rt60

    rt60_values.append(rt60)

    # ---------------------------------------------------------
    # Convert to sample index (relative to CUT signal)
    # ---------------------------------------------------------
    inter_sample = int(inter_time * fs) if not np.isnan(inter_time) else 0
    inter_sample = np.clip(inter_sample, 0, len(rir) - 1)

    # recompute peak index safety (optional robustness)
    peak_idx = int(np.argmax(np.abs(rir[:, 0])))

    # ---------------------------------------------------------
    # Define truncation point (existing logic unchanged)
    # ---------------------------------------------------------
    fade_start = max(inter_sample + tail_offset,
                     peak_idx + tail_offset)

    fade_end = min(fade_start + fade_len, len(rir))
    cut_end = fade_end

    # ---------------------------------------------------------
    # Apply fade (NOW includes fade-in at start)
    # ---------------------------------------------------------
    rir_faded, window = apply_fade(
        rir,
        fade_in_len=fade_len,
        fade_out_start=fade_start,
        fade_out_len=fade_len
    )

    # ---------------------------------------------------------
    # HARD TRUNCATION
    # ---------------------------------------------------------
    rir_out = rir_faded[:cut_end, :]

    print("\n==============================")
    print("File:", fname)
    print("Original length:", len(rir_full))
    print("Cut start (pre-peak):", cut_start)
    print("Peak index (original):", peak_idx_full)
    print("Peak index (shifted):", peak_idx)
    print("Cut end:", cut_end)
    print("Final length:", len(rir_out))
    print("RT60:", rt60)

    # ---------------------------------------------------------
    # Save WAV
    # ---------------------------------------------------------
    wavfile.write(out_path, fs, rir_out.astype(np.int16))


# -----------------------------
# Save CSV
# -----------------------------
df["rt60"] = rt60_values
df.to_csv(output_csv_path, index=False)

print("\nSaved CSV:", output_csv_path)