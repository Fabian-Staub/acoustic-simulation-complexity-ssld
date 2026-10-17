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
    "/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/motus/sh_rirs"
)

output_csv_path = "motus_rt60_bands.csv"

# The filter range (100, 20 kHz) produces these nominal octave bands.
octave_bands = (125, 250, 500, 1000, 2000, 4000, 8000, 16000)

# A value must satisfy both limits to be treated as an outlier. Combining a
# robust z-score with a ratio limit avoids replacing normal spectral variation.
outlier_z_threshold = 3.5
outlier_ratio_threshold = 2.0


# =============================================================
# Helper functions
# =============================================================
def to_scalar(value):
    """Convert a scalar-like NumPy value to a Python float."""
    return float(np.asarray(value).squeeze())


def estimate_rt60(signal, frequency="broadband"):
    """Estimate RT60 with the Lundeby method for one pyfar Signal."""
    _, rt60, _ = ra.edc.intersection_time_lundeby(
        signal,
        freq=frequency,
    )

    rt60 = to_scalar(rt60)
    if not np.isfinite(rt60):
        raise ValueError(f"Invalid RT60: {rt60}")

    return rt60


def estimate_octave_band_rt60s(signal):
    """Filter the RIR once and estimate RT60 for every octave band."""
    rt60s = {
        f"rt60_{frequency}Hz": np.nan for frequency in octave_bands
    }
    # The returned Signal has one channel per octave band, ordered from
    # low to high center frequency. Bands above Nyquist are omitted by pyfar.
    band_signals = pf.dsp.filter.fractional_octave_bands(
        signal=signal,
        num_fractions=1,
        frequency_range=(100, 20e3),
        order=14,
    )

    num_available_bands = band_signals.cshape[0]

    for band_index, frequency in enumerate(
        octave_bands[:num_available_bands]
    ):
        column = f"rt60_{frequency}Hz"

        try:
            # Select one filter-bank channel while retaining a channel axis.
            band_signal = pf.Signal(
                band_signals.time[band_index],
                band_signals.sampling_rate,
            )
            rt60s[column] = estimate_rt60(
                band_signal,
                frequency=frequency,
            )
        except Exception as error:
            print(f"[WARN] {frequency} Hz RT60 estimation failed: {error}")

    for frequency in octave_bands[num_available_bands:]:
        print(
            f"[WARN] {frequency} Hz band is unavailable at "
            f"{signal.sampling_rate} Hz sampling rate"
        )

    return rt60s


def room_name(filename):
    """Return the room part of '<room>_P<number>_S<number>.wav'."""
    return re.sub(r"_P\d+_S\d+\.wav$", "", filename, flags=re.IGNORECASE)


def robust_outlier_mask(values):
    """Find strong outliers using the median, MAD, and a ratio guard."""
    values = np.asarray(values, dtype=float)
    outliers = np.zeros(values.shape, dtype=bool)
    finite = np.isfinite(values) & (values > 0)

    if np.count_nonzero(finite) < 3:
        return outliers

    finite_values = values[finite]
    median = np.median(finite_values)
    deviations = np.abs(finite_values - median)
    mad = np.median(deviations)

    ratios = np.maximum(
        finite_values / median,
        median / finite_values,
    )

    if mad > np.finfo(float).eps:
        robust_z = deviations / (1.4826 * mad)
    else:
        robust_z = np.where(
            deviations > np.finfo(float).eps,
            np.inf,
            0.0,
        )

    outliers[finite] = (
        (robust_z > outlier_z_threshold)
        & (ratios > outlier_ratio_threshold)
    )
    return outliers


def is_outlier_against(value, reference_values):
    """Check one value against a set of valid reference values."""
    references = np.asarray(reference_values, dtype=float)
    references = references[np.isfinite(references) & (references > 0)]

    if not np.isfinite(value) or value <= 0 or references.size < 3:
        return False

    median = np.median(references)
    mad = np.median(np.abs(references - median))
    ratio = max(value / median, median / value)

    if mad > np.finfo(float).eps:
        robust_z = abs(value - median) / (1.4826 * mad)
    else:
        robust_z = np.inf if value != median else 0.0

    return (
        robust_z > outlier_z_threshold
        and ratio > outlier_ratio_threshold
    )


def isolated_band_outlier_mask(values):
    """Find isolated interior-band spikes without flattening spectral trends."""
    values = np.asarray(values, dtype=float)
    global_outliers = robust_outlier_mask(values)
    isolated_outliers = np.zeros(values.shape, dtype=bool)

    # End bands are deliberately excluded: a real room can have a systematic
    # low- or high-frequency RT60 slope. Repeated same-room measurements handle
    # extreme values at 125 Hz and 16 kHz more reliably.
    for index in range(1, values.size - 1):
        local_values = values[index - 1:index + 2]
        if not global_outliers[index] or not np.all(
            np.isfinite(local_values) & (local_values > 0)
        ):
            continue

        neighbor_mean = np.mean([values[index - 1], values[index + 1]])
        local_ratio = max(
            values[index] / neighbor_mean,
            neighbor_mean / values[index],
        )
        isolated_outliers[index] = local_ratio > outlier_ratio_threshold

    return isolated_outliers


def interpolate_bad_bands(values, bad_mask):
    """Interpolate marked bands without using other replacements as inputs."""
    values = np.asarray(values, dtype=float)
    bad_mask = np.asarray(bad_mask, dtype=bool)
    repaired = values.copy()
    valid_indices = [
        index
        for index in range(values.size)
        if not bad_mask[index] and np.isfinite(values[index]) and values[index] > 0
    ]

    for band_index in np.flatnonzero(bad_mask):
        lower = [index for index in valid_indices if index < band_index]
        upper = [index for index in valid_indices if index > band_index]

        if lower and upper:
            source_indices = [lower[-1], upper[0]]
        elif lower:
            source_indices = lower[-2:]
        else:
            source_indices = upper[:2]

        if source_indices:
            repaired[band_index] = float(np.mean(values[source_indices]))

    return repaired


def repair_rt60_dataframe(dataframe, rt60_columns):
    """Repair failed and extreme RT60 values using room and band context."""
    dataframe = dataframe.copy()
    band_columns = [
        f"rt60_{frequency}Hz" for frequency in octave_bands
    ]
    room_keys = dataframe["filename"].map(room_name)

    # First use repeated measurements from the same room and same RT60 column.
    # This is preferable to cross-frequency interpolation when peers exist.
    for room, indices in dataframe.groupby(room_keys).groups.items():
        for column in rt60_columns:
            values = dataframe.loc[indices, column].to_numpy(dtype=float)
            outliers = robust_outlier_mask(values)
            invalid = ~np.isfinite(values) | (values <= 0)
            replace = outliers | invalid
            sources = ~replace & np.isfinite(values) & (values > 0)

            if not np.any(replace) or not np.any(sources):
                continue

            replacement = float(np.median(values[sources]))
            for position in np.flatnonzero(replace):
                row_index = indices[position]
                old_value = values[position]
                reason = "outlier" if outliers[position] else "failed"
                dataframe.at[row_index, column] = replacement
                print(
                    f"[REPAIR] {dataframe.at[row_index, 'filename']} "
                    f"{column}: {reason} {old_value!r} -> "
                    f"{replacement:.4g} s (same-room median)"
                )

    # Then repair remaining failed or extreme octave-band values within a RIR.
    for row_index in dataframe.index:
        values = dataframe.loc[row_index, band_columns].to_numpy(dtype=float)
        outliers = isolated_band_outlier_mask(values)
        invalid = ~np.isfinite(values) | (values <= 0)

        # Do not invent bands that pyfar could not create above Nyquist.
        sampling_rate = dataframe.at[row_index, "sampling_rate"]
        octave_edge_ratio = 10 ** (3 / 20)
        available = np.array(
            [
                frequency / octave_edge_ratio < sampling_rate / 2
                for frequency in octave_bands
            ]
        )
        repair = (outliers | invalid) & available
        repaired = interpolate_bad_bands(values, repair)

        for position in np.flatnonzero(repair & np.isfinite(repaired)):
            column = band_columns[position]
            old_value = values[position]
            new_value = repaired[position]
            reason = "outlier" if outliers[position] else "failed"
            dataframe.at[row_index, column] = new_value
            print(
                f"[REPAIR] {dataframe.at[row_index, 'filename']} "
                f"{column}: {reason} {old_value!r} -> "
                f"{new_value:.4g} s (neighbor-band interpolation)"
            )

        # Broadband has no position on the frequency axis. If same-room repair
        # was impossible, or it remains a strong outlier, use the cleaned band
        # median for this RIR.
        cleaned_bands = dataframe.loc[
            row_index, band_columns
        ].to_numpy(dtype=float)
        valid_bands = cleaned_bands[
            np.isfinite(cleaned_bands) & (cleaned_bands > 0)
        ]
        broadband = dataframe.at[row_index, "rt60"]
        broadband_failed = not np.isfinite(broadband) or broadband <= 0
        broadband_outlier = is_outlier_against(broadband, valid_bands)

        if (broadband_failed or broadband_outlier) and valid_bands.size:
            replacement = float(np.median(valid_bands))
            reason = "outlier" if broadband_outlier else "failed"
            dataframe.at[row_index, "rt60"] = replacement
            print(
                f"[REPAIR] {dataframe.at[row_index, 'filename']} rt60: "
                f"{reason} {broadband!r} -> {replacement:.4g} s "
                "(band median)"
            )

    return dataframe


# =============================================================
# Find all WAV files
# =============================================================
input_files = sorted(
    filename
    for filename in os.listdir(input_folder)
    if (
        filename.lower().endswith(".wav")
        and os.path.isfile(os.path.join(input_folder, filename))
    )
)

print(f"Found {len(input_files)} WAV files.")


# =============================================================
# Estimate broadband and octave-band RT60 values
# =============================================================
rt60_columns = ["rt60"] + [
    f"rt60_{frequency}Hz" for frequency in octave_bands
]

result_columns = [
    "filename",
    "sampling_rate",
] + rt60_columns

results = []

for input_filename in input_files:
    input_path = os.path.join(input_folder, input_filename)

    row = {
        "filename": input_filename,
        "sampling_rate": np.nan,
        **{column: np.nan for column in rt60_columns},
    }

    try:
        sampling_rate, rir = wavfile.read(input_path)
        row["sampling_rate"] = sampling_rate

        if rir.size == 0:
            raise ValueError("empty WAV file")

        # Analyze the first channel of a multichannel WAV.
        if rir.ndim > 1:
            rir = rir[:, 0]

        rir_signal = pf.Signal(
            rir.astype(np.float64, copy=False),
            sampling_rate,
        )

        try:
            row["rt60"] = estimate_rt60(rir_signal)
        except Exception as error:
            print(
                f"[WARN] Broadband RT60 estimation failed for "
                f"{input_filename}: {error}"
            )

        try:
            row.update(estimate_octave_band_rt60s(rir_signal))
        except Exception as error:
            print(
                f"[WARN] Octave filtering failed for "
                f"{input_filename}: {error}"
            )

    except Exception as error:
        print(f"[ERROR] Could not process {input_filename}: {error}")

    results.append(row)


# =============================================================
# Save the RT60 CSV
# =============================================================
output_df = pd.DataFrame(results, columns=result_columns)

if not output_df.empty:
    output_df = repair_rt60_dataframe(output_df, rt60_columns)
    output_df = output_df.sort_values(
        by="filename",
        key=lambda column: column.str.lower(),
    )

    for _, output_row in output_df.iterrows():
        values = ", ".join(
            (
                f"{column}={output_row[column]:.4g}"
                if np.isfinite(output_row[column])
                else f"{column}=NaN"
            )
            for column in rt60_columns
        )
        print(f"[DONE] {output_row['filename']}: {values}")

output_parent = os.path.dirname(output_csv_path)
if output_parent:
    os.makedirs(output_parent, exist_ok=True)

output_df.to_csv(output_csv_path, index=False)

print("\n========================================")
print(f"Processed files: {len(results)}")
print(f"RT60 CSV: {output_csv_path}")
