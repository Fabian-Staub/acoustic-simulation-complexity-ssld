from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.io import wavfile


# ============================================================
# Settings
# ============================================================

INPUT_FOLDER = Path("/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/motus/RIRs_truncated")
OUTPUT_FOLDER = Path("/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/motus/RIRs_truncated_n3d")

# Process subfolders too:
RECURSIVE = False

# Numerical tolerance used when checking the resulting peak.
PEAK_TOLERANCE = 1e-6


# ============================================================
# Helper functions
# ============================================================

def pcm_to_float64(audio: np.ndarray) -> np.ndarray:
    """
    Convert WAV samples to float64.

    Integer PCM is normalized approximately to [-1, 1].
    Floating-point input is retained at its existing scale.
    """
    if np.issubdtype(audio.dtype, np.floating):
        return audio.astype(np.float64)

    if audio.dtype == np.uint8:
        # 8-bit PCM is unsigned, with silence at 128.
        return (audio.astype(np.float64) - 128.0) / 128.0

    if np.issubdtype(audio.dtype, np.signedinteger):
        info = np.iinfo(audio.dtype)

        # Use the magnitude of the negative full-scale value.
        scale = float(abs(info.min))
        return audio.astype(np.float64) / scale

    raise TypeError(f"Unsupported WAV dtype: {audio.dtype}")


def determine_ambisonic_order(number_of_channels: int) -> int:
    """
    Infer the full-sphere Ambisonic order from the channel count.

    For full-sphere Ambisonics:
        channels = (order + 1)^2
    """
    order = int(np.sqrt(number_of_channels) - 1)

    if (order + 1) ** 2 != number_of_channels:
        raise ValueError(
            f"{number_of_channels} channels is not a valid full-sphere "
            "Ambisonic channel count. Expected 1, 4, 9, 16, 25, ..."
        )

    return order


def make_sn3d_to_n3d_factors(order: int) -> np.ndarray:
    """
    Make one SN3D-to-N3D factor for every ACN channel.

    Every channel of spherical-harmonic order n is multiplied by:

        sqrt(2*n + 1)

    Under ACN ordering, the channels of order n occupy:
        n^2 ... (n + 1)^2 - 1
    """
    number_of_channels = (order + 1) ** 2
    factors = np.empty(number_of_channels, dtype=np.float64)

    for n in range(order + 1):
        first_channel = n**2
        last_channel_exclusive = (n + 1) ** 2

        factors[first_channel:last_channel_exclusive] = np.sqrt(
            2 * n + 1
        )

    return factors


def convert_file_sn3d_to_n3d(
    input_path: Path,
    output_path: Path,
) -> dict:
    """
    Convert one ACN/SN3D WAV file to ACN/N3D.

    One common gain is applied after the normalization conversion so that
    the global peak across all samples and channels remains unchanged.

    The result is written as a 32-bit floating-point WAV.
    """
    sampling_rate, audio_original = wavfile.read(input_path)

    if audio_original.ndim != 2:
        raise ValueError(
            f"Expected a multichannel WAV with shape "
            f"(samples, channels), but got {audio_original.shape}."
        )

    number_of_samples, number_of_channels = audio_original.shape
    order = determine_ambisonic_order(number_of_channels)

    audio = pcm_to_float64(audio_original)

    if not np.all(np.isfinite(audio)):
        raise ValueError("Input contains NaN or infinite samples.")

    input_peak = float(np.max(np.abs(audio)))

    factors = make_sn3d_to_n3d_factors(order)

    # Apply the order-dependent N3D factors.
    n3d = audio * factors[np.newaxis, :]

    unscaled_n3d_peak = float(np.max(np.abs(n3d)))

    # Preserve the file's original global peak.
    if input_peak == 0.0:
        peak_preservation_gain = 1.0
    elif unscaled_n3d_peak == 0.0:
        raise ValueError(
            "Unexpected zero output peak for a non-zero input file."
        )
    else:
        peak_preservation_gain = input_peak / unscaled_n3d_peak

    n3d *= peak_preservation_gain

    output_peak = float(np.max(np.abs(n3d)))

    # Convert to IEEE 32-bit floating-point WAV.
    output_audio = n3d.astype(np.float32)

    # Recalculate after float32 conversion.
    stored_output_peak = float(
        np.max(np.abs(output_audio.astype(np.float64)))
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(output_path, sampling_rate, output_audio)

    relative_peak_error = (
        0.0
        if input_peak == 0.0
        else abs(stored_output_peak - input_peak) / input_peak
    )

    if relative_peak_error > PEAK_TOLERANCE:
        print(
            f"[WARNING] Peak mismatch for {input_path.name}: "
            f"input={input_peak:.9f}, "
            f"output={stored_output_peak:.9f}, "
            f"relative error={relative_peak_error:.3e}"
        )

    return {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "sampling_rate": sampling_rate,
        "samples": number_of_samples,
        "channels": number_of_channels,
        "ambisonic_order": order,
        "input_dtype": str(audio_original.dtype),
        "output_dtype": str(output_audio.dtype),
        "input_peak": input_peak,
        "unscaled_n3d_peak": unscaled_n3d_peak,
        "peak_preservation_gain": peak_preservation_gain,
        "output_peak": stored_output_peak,
        "relative_peak_error": relative_peak_error,
    }


def convert_folder(
    input_folder: Path,
    output_folder: Path,
    recursive: bool = False,
) -> list[dict]:
    """
    Convert every WAV file in a folder.

    When recursive=True, the input subfolder structure is preserved.
    """
    if not input_folder.is_dir():
        raise NotADirectoryError(
            f"Input folder does not exist: {input_folder}"
        )

    if input_folder.resolve() == output_folder.resolve():
        raise ValueError(
            "Input and output folders must be different."
        )

    pattern = "**/*.wav" if recursive else "*.wav"

    wav_files = sorted(
        path
        for path in input_folder.glob(pattern)
        if path.is_file()
    )

    if not wav_files:
        print(f"No WAV files found in {input_folder}")
        return []

    output_folder.mkdir(parents=True, exist_ok=True)

    results = []

    for input_path in wav_files:
        relative_path = input_path.relative_to(input_folder)
        output_path = output_folder / relative_path

        try:
            result = convert_file_sn3d_to_n3d(
                input_path=input_path,
                output_path=output_path,
            )
            results.append(result)

            print(
                f"[OK] {relative_path} | "
                f"order={result['ambisonic_order']} | "
                f"channels={result['channels']} | "
                f"input peak={result['input_peak']:.7f} | "
                f"output peak={result['output_peak']:.7f} | "
                f"global gain={result['peak_preservation_gain']:.7f}"
            )

        except Exception as error:
            print(f"[ERROR] {relative_path}: {error}")

    return results


def save_report(results: list[dict], report_path: Path) -> None:
    """Save conversion information as a CSV report."""
    if not results:
        return

    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)


# ============================================================
# Run conversion
# ============================================================

if __name__ == "__main__":
    conversion_results = convert_folder(
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER,
        recursive=RECURSIVE,
    )

    save_report(
        conversion_results,
        OUTPUT_FOLDER / "sn3d_to_n3d_report.csv",
    )

    print()
    print(f"Successfully converted: {len(conversion_results)} files")
    print(f"Output folder: {OUTPUT_FOLDER}")