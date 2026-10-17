import os
import re

import numpy as np
import pandas as pd
from scipy.io import wavfile

import pyfar as pf
import pyrato as ra


# =============================================================
# Settings
# =============================================================
input_folder = (
    "/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/"
    "staubf1/Projects/Master_Thesis/data/motus/RIRs"
)

output_folder = (
    "/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/"
    "staubf1/Projects/Master_Thesis/data/motus/RIRs_truncated"
)

output_csv_path = (
    "/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/"
    "staubf1/Projects/Master_Thesis/data/motus/"
    "motus_rt60.csv"
)

# Position number used in the renamed files:
# RXXXX_P1_SX.wav
position_id = 1

# Only these source indices are processed
allowed_sources = {1, 2, 3}

fade_len = 10
tail_offset = 300

# Number of samples kept before the direct-sound peak
pre_peak_cut = 200

os.makedirs(output_folder, exist_ok=True)


# =============================================================
# Filename pattern
# Example: 0001_2_sh_rirs.wav
# =============================================================
filename_pattern = re.compile(
    r"^(?P<rir_id>\d+)_(?P<source_id>\d+)_sh_rirs\.wav$",
    flags=re.IGNORECASE,
)


# =============================================================
# Helper functions
# =============================================================
def to_scalar(value):
    """Convert a scalar-like NumPy value to a Python float."""
    return float(np.asarray(value).squeeze())


def apply_fade(rir, fade_in_len, fade_out_start, fade_out_len):
    """
    Apply a linear fade-in and fade-out.

    Parameters
    ----------
    rir : ndarray, shape (samples, channels)
        Input room impulse response.
    fade_in_len : int
        Length of fade-in in samples.
    fade_out_start : int
        Sample at which fade-out begins.
    fade_out_len : int
        Length of fade-out in samples.

    Returns
    -------
    rir_faded : ndarray
        Faded RIR.
    window : ndarray
        Applied one-dimensional window.
    """
    n_samples = rir.shape[0]

    fade_in_len = min(max(int(fade_in_len), 0), n_samples)
    fade_out_start = int(np.clip(fade_out_start, 0, n_samples))
    fade_out_len = max(int(fade_out_len), 0)

    window = np.ones(n_samples, dtype=np.float64)

    # Fade-in
    if fade_in_len > 0:
        window[:fade_in_len] = np.linspace(
            0.0,
            1.0,
            fade_in_len,
            endpoint=True,
        )

    # Fade-out
    fade_out_end = min(fade_out_start + fade_out_len, n_samples)

    if fade_out_start < fade_out_end:
        window[fade_out_start:fade_out_end] *= np.linspace(
            1.0,
            0.0,
            fade_out_end - fade_out_start,
            endpoint=True,
        )

        window[fade_out_end:] = 0.0

    return rir * window[:, None], window


def prepare_audio_for_processing(audio):
    """
    Convert WAV data to floating point while retaining the original scale.

    Returning the original dtype allows the output to be converted back
    safely before writing.
    """
    original_dtype = audio.dtype
    return audio.astype(np.float64), original_dtype


def restore_audio_dtype(audio, original_dtype):
    """Convert processed floating-point audio back to its original WAV dtype."""
    if np.issubdtype(original_dtype, np.integer):
        limits = np.iinfo(original_dtype)
        audio = np.clip(np.rint(audio), limits.min, limits.max)
        return audio.astype(original_dtype)

    return audio.astype(original_dtype)


# =============================================================
# Find and parse input WAV files
# =============================================================
input_files = []

for filename in os.listdir(input_folder):
    match = filename_pattern.match(filename)

    if match is None:
        continue

    rir_id_text = match.group("rir_id")
    source_id = int(match.group("source_id"))

    # Skip source 4 and all other unwanted sources
    if source_id not in allowed_sources:
        print(f"[SKIP] Source {source_id}: {filename}")
        continue

    input_files.append(
        {
            "input_filename": filename,
            "rir_id": int(rir_id_text),
            "rir_id_width": len(rir_id_text),
            "source_id": source_id,
        }
    )


# Sort numerically, not alphabetically
input_files.sort(
    key=lambda item: (
        item["rir_id"],
        item["source_id"],
    )
)

print(f"Found {len(input_files)} files from sources 1–3.")


# =============================================================
# Output information
# =============================================================
results = []

# Used only as a fallback if Lundeby estimation fails
previous_intersection_time = np.nan
previous_rt60 = np.nan


# =============================================================
# Processing loop
# =============================================================
for file_info in input_files:
    input_filename = file_info["input_filename"]
    rir_id = file_info["rir_id"]
    rir_id_width = file_info["rir_id_width"]
    source_id = file_info["source_id"]

    input_path = os.path.join(input_folder, input_filename)

    # Preserve at least four digits in RXXXX
    output_id_width = max(4, rir_id_width)

    output_filename = (
        f"R{rir_id:0{output_id_width}d}"
        f"_P{position_id}"
        f"_S{source_id}.wav"
    )

    output_path = os.path.join(output_folder, output_filename)

    try:
        sampling_rate, rir_original = wavfile.read(input_path)
    except Exception as error:
        print(f"[ERROR] Could not read {input_filename}: {error}")

        results.append(
            {
                "Name": output_filename,
                "input_name": input_filename,
                "rt60": np.nan,
            }
        )
        continue

    rir_full, original_dtype = prepare_audio_for_processing(rir_original)

    # Convert mono input to shape (samples, channels)
    if rir_full.ndim == 1:
        rir_full = rir_full[:, None]

    if rir_full.shape[0] == 0:
        print(f"[ERROR] Empty WAV file: {input_filename}")

        results.append(
            {
                "Name": output_filename,
                "input_name": input_filename,
                "rt60": np.nan,
            }
        )
        continue

    # ---------------------------------------------------------
    # Peak detection and pre-peak trimming
    # ---------------------------------------------------------
    peak_idx_full = int(np.argmax(np.abs(rir_full[:, 0])))

    cut_start = max(0, peak_idx_full - pre_peak_cut)

    rir = rir_full[cut_start:, :].copy()

    # Peak position relative to the trimmed signal
    peak_idx = peak_idx_full - cut_start

    # ---------------------------------------------------------
    # RT60 estimation
    # ---------------------------------------------------------
    rt60_status = "estimated"

    try:
        rir_signal = pf.Signal(rir[:, 0], sampling_rate)

        intersection_time, rt60, _ = (
            ra.edc.intersection_time_lundeby(rir_signal)
        )

        intersection_time = to_scalar(intersection_time)
        rt60 = to_scalar(rt60)

        if not np.isfinite(intersection_time):
            raise ValueError(
                f"Invalid intersection time: {intersection_time}"
            )

        if not np.isfinite(rt60):
            raise ValueError(f"Invalid RT60: {rt60}")

        previous_intersection_time = intersection_time
        previous_rt60 = rt60

    except Exception as error:
        print(
            f"[WARN] RT60 estimation failed for "
            f"{input_filename}: {error}"
        )

        if (
            np.isfinite(previous_intersection_time)
            and np.isfinite(previous_rt60)
        ):
            print("       Using values from the previous valid RIR.")

            intersection_time = previous_intersection_time
            rt60 = previous_rt60
            rt60_status = "previous_value"

        else:
            print("       No previous valid RT60 value is available.")

            intersection_time = np.nan
            rt60 = np.nan
            rt60_status = "failed"

    # ---------------------------------------------------------
    # Convert intersection time to sample index
    # ---------------------------------------------------------
    if np.isfinite(intersection_time):
        intersection_sample = int(
            np.round(intersection_time * sampling_rate)
        )
        intersection_sample = int(
            np.clip(intersection_sample, 0, len(rir) - 1)
        )
    else:
        # The truncation is then based on the peak only
        intersection_sample = 0

    # Recalculate for additional safety
    peak_idx = int(np.argmax(np.abs(rir[:, 0])))

    # ---------------------------------------------------------
    # Define truncation and fade-out point
    # ---------------------------------------------------------
    fade_start = max(
        intersection_sample + tail_offset,
        peak_idx + tail_offset,
    )

    # Do not place the fade beyond the available signal
    fade_start = min(fade_start, len(rir))

    fade_end = min(fade_start + fade_len, len(rir))
    cut_end = fade_end

    # ---------------------------------------------------------
    # Apply fade-in and fade-out
    # ---------------------------------------------------------
    rir_faded, _ = apply_fade(
        rir=rir,
        fade_in_len=fade_len,
        fade_out_start=fade_start,
        fade_out_len=fade_len,
    )

    # ---------------------------------------------------------
    # Hard truncation
    # ---------------------------------------------------------
    rir_out = rir_faded[:cut_end, :]

    # Return mono files to one-dimensional form
    if rir_original.ndim == 1:
        rir_out = rir_out[:, 0]

    rir_out = restore_audio_dtype(rir_out, original_dtype)

    # ---------------------------------------------------------
    # Save WAV
    # ---------------------------------------------------------
    try:
        wavfile.write(output_path, sampling_rate, rir_out)
    except Exception as error:
        print(f"[ERROR] Could not save {output_filename}: {error}")

        results.append(
            {
                "Name": output_filename,
                "input_name": input_filename,
                "rt60": rt60,
                "rt60_status": rt60_status,
                "saved": False,
            }
        )
        continue

    # ---------------------------------------------------------
    # Store output CSV row
    # ---------------------------------------------------------
    results.append(
        {
            "Name": output_filename,
            "input_name": input_filename,
            "rir": rir_id,
            "position": position_id,
            "source": source_id,
            "sampling_rate": sampling_rate,
            "rt60": rt60,
            "rt60_status": rt60_status,
            "original_length_samples": len(rir_full),
            "cut_start_samples": cut_start,
            "final_length_samples": len(rir_out),
            "saved": True,
        }
    )

    print("\n========================================")
    print("Input file:              ", input_filename)
    print("Output file:             ", output_filename)
    print("Source:                  ", source_id)
    print("Original length:         ", len(rir_full))
    print("Pre-peak cut start:      ", cut_start)
    print("Peak index, original:    ", peak_idx_full)
    print("Peak index, shifted:     ", peak_idx)
    print("Intersection sample:     ", intersection_sample)
    print("Fade start:              ", fade_start)
    print("Final length:            ", len(rir_out))
    print("RT60:                    ", rt60)
    print("RT60 status:             ", rt60_status)


# =============================================================
# Save RT60 CSV
# =============================================================
output_df = pd.DataFrame(results)

# Ensure a predictable order in the output CSV
if not output_df.empty:
    output_df = output_df.sort_values(
        by=["rir", "source"],
        na_position="last",
    )

output_df.to_csv(output_csv_path, index=False)

print("\n========================================")
print(f"Processed files: {len(results)}")
print(f"Truncated RIR folder: {output_folder}")
print(f"RT60 CSV: {output_csv_path}")