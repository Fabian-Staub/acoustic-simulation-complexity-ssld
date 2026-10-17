import csv
import glob
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import spaudiopy as spa
from scipy.io import wavfile
from scipy.signal import find_peaks
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from utils import compute_scm_avg_log

EPS = 1e-12


def natural_key(path):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", os.path.basename(path))
    ]


def load_direct_sound_from_rir(audio_path, peak_window=20, threshold_ratio=0.5):
    """Load an RIR, retain a short window around its first direct arrival,
    and apply a Hann window.
    """
    fs, rir = wavfile.read(audio_path)

    # Convert integer PCM samples to floating-point full-scale values.
    # Example for int16: -32768...32767 becomes approximately -1...1.
    if np.issubdtype(rir.dtype, np.integer):
        scale = max(abs(np.iinfo(rir.dtype).min), np.iinfo(rir.dtype).max)
        rir = rir.astype(np.float64) / scale
    else:
        rir = rir.astype(np.float64)

    rir = np.asarray(rir).T

    if rir.ndim == 1:
        rir = rir[None, :]

    ch0_abs = np.abs(rir[0].astype(np.float64, copy=False))
    peak = ch0_abs.max(initial=0.0)

    if peak <= EPS:
        onset = 0
    else:
        peaks, _ = find_peaks(ch0_abs, height=threshold_ratio * peak)

        if peaks.size:
            onset = int(peaks[0])
        else:
            crossings = np.flatnonzero(ch0_abs >= threshold_ratio * peak)
            onset = int(crossings[0]) if crossings.size else 0

    half_win = peak_window // 2
    start = max(0, onset - half_win)
    end = min(rir.shape[1], start + peak_window)
    start = max(0, end - peak_window)

    direct_rir = rir[:, start:end].astype(np.float64, copy=False)

    # Match the Hann window to the actual extracted length, which may be
    # shorter than peak_window for very short input signals.
    hann_window = np.hanning(direct_rir.shape[1])
    direct_rir = direct_rir * hann_window[None, :]

    return fs, np.ascontiguousarray(direct_rir)

'''
def load_direct_sound_from_rir(audio_path, peak_window=20):
    """
    Load a multichannel RIR, locate the maximum absolute peak in channel 0,
    extract a fixed-length window centered on that peak, and apply a Hann
    window to every channel.

    Returns
    -------
    fs : int
        Sampling rate.
    direct_rir : np.ndarray
        Windowed RIR with shape (channels, peak_window).
    """
    fs, rir = wavfile.read(audio_path)
    rir = np.asarray(rir).T

    if rir.ndim == 1:
        rir = rir[None, :]

    num_samples = rir.shape[1]

    if num_samples == 0:
        return fs, np.empty((rir.shape[0], 0), dtype=np.float64)

    # Locate the strongest absolute peak using channel 0.
    ch0_abs = np.abs(rir[0].astype(np.float64, copy=False))
    peak_index = int(np.argmax(ch0_abs))

    # Define a window centered on the peak.
    half_window = peak_window // 2
    start = peak_index - half_window
    end = start + peak_window

    # Pad when the centered window extends beyond the RIR boundaries.
    pad_left = max(0, -start)
    pad_right = max(0, end - num_samples)

    start = max(0, start)
    end = min(num_samples, end)

    direct_rir = rir[:, start:end].astype(np.float64, copy=False)

    if pad_left or pad_right:
        direct_rir = np.pad(
            direct_rir,
            pad_width=((0, 0), (pad_left, pad_right)),
            mode="constant",
        )

    # Apply the same Hann window to every channel.
    hann_window = np.hanning(peak_window)
    direct_rir = direct_rir * hann_window[None, :]

    return fs, np.ascontiguousarray(direct_rir)
'''
def precompute_music_grid(order, az_grid, zen_grid):
    """
    Precompute steering vectors for paired n-design directions.

    Important: az_grid[i] and zen_grid[i] describe one direction. A meshgrid
    would incorrectly turn N directions into N**2 combinations.
    """
    az = np.asarray(az_grid, dtype=np.float64).ravel()
    zen = np.asarray(zen_grid, dtype=np.float64).ravel()

    if az.shape != zen.shape:
        raise ValueError("az_grid and zen_grid must contain paired directions")

    steering = spa.sph.sh_matrix(
        order,
        az,
        zen,
        sh_type="real",
    ).T

    # Precompute ||y||^2, used by the rank-1 single-source MUSIC formula.
    steering = np.ascontiguousarray(steering)
    steering_energy = np.einsum(
        "cg,cg->g", steering.conj(), steering, optimize=True
    ).real

    az_deg = np.rad2deg(az)
    el_deg = 90.0 - np.rad2deg(zen)
    return az, zen, az_deg, el_deg, steering, steering_energy


def music_spectrum_single_source(scm_bands, steering, steering_energy):
    """
    Compute the averaged single-source MUSIC spectrum.

    For one source, P_noise = I - u uᴴ. Therefore
        yᴴ P_noise y = ||y||² - |uᴴ y|²,
    avoiding construction and multiplication of a full projector per band.
    """
    scm_bands = np.asarray(scm_bands)
    if scm_bands.ndim != 3:
        raise ValueError("Expected scm_bands with shape (bands, channels, channels)")

    # np.linalg.eigh supports stacked matrices. Eigenvalues are ascending, so
    # the final eigenvector is the principal (single-source) eigenvector.
    _, eigenvectors = np.linalg.eigh(scm_bands)
    principal = eigenvectors[:, :, -1]  # (bands, channels)

    projections = principal.conj() @ steering  # (bands, directions)
    denominator = steering_energy[None, :] - np.abs(projections) ** 2
    denominator = np.maximum(denominator.real, EPS)

    return np.mean(1.0 / denominator, axis=0)


def estimate_doa_top1(az_deg, el_deg, spectrum):
    """TOP-1 needs only argmax, not a complete O(N log N) sort."""
    if spectrum.size == 0 or not np.any(np.isfinite(spectrum)):
        return np.nan, np.nan

    idx = int(np.nanargmax(spectrum))
    return float(az_deg[idx]), float(el_deg[idx])


def azel_to_cartesian(az_deg, el_deg):
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)
    cos_el = np.cos(el)
    return np.array(
        [cos_el * np.cos(az), cos_el * np.sin(az), np.sin(el)],
        dtype=np.float64,
    )


def plot_signal(signal, fs, save_path, title="Direct Sound Component of the Multichannel RIR", dpi=150):
    t = np.arange(signal.shape[1], dtype=np.float64) / fs
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, signal.T)
    ax.set(xlabel="Time (s)", ylabel="Amplitude", title=title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_music(az_deg, el_deg, spectrum, az_est, el_est, save_path, dpi=150):
    peak = np.nanmax(spectrum)
    spectrum_db = 10.0 * np.log10(spectrum / max(peak, EPS) + EPS)

    fig, ax = plt.subplots(figsize=(8, 4))
    contour = ax.tricontourf(az_deg, el_deg, spectrum_db, levels=50)
    fig.colorbar(contour, ax=ax, label="dB")
    ax.scatter(az_est, el_est, c="red", s=100, marker="x", label="Estimated DOA")
    ax.set(
        xlabel="Azimuth (deg)",
        ylabel="Elevation (deg)",
        title="MUSIC Pseudospectrum",
    )
    ax.invert_xaxis()
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)


def process_sample(
    audio_path,
    steering,
    steering_energy,
    az_deg,
    el_deg,
    output_dir=None,
    save_plots=False,
    plot_dpi=150,
):
    fs, signal = load_direct_sound_from_rir(
        audio_path,
        peak_window=100,
        threshold_ratio=0.3,
    )

    scm_bands = compute_scm_avg_log(
        signal,
        fs,
        freq_range=(2000, 8000),
        num_scms=10,
    )
    scm_bands = np.asarray(scm_bands) * 2.0 - 1.0

    spectrum = music_spectrum_single_source(
        scm_bands,
        steering,
        steering_energy,
    )
    az_est, el_est = estimate_doa_top1(az_deg, el_deg, spectrum)
    xyz = azel_to_cartesian(az_est, el_est)

    if save_plots:
        if output_dir is None:
            raise ValueError("output_dir is required when save_plots=True")
        basename = Path(audio_path).stem
        plot_signal(
            signal,
            fs,
            os.path.join(output_dir, f"{basename}_direct_sound.pdf"),
            dpi=plot_dpi,
        )
        plot_music(
            az_deg,
            el_deg,
            spectrum,
            az_est,
            el_est,
            os.path.join(output_dir, f"{basename}_music.pdf"),
            dpi=plot_dpi,
        )

    return az_est, el_est, xyz


def evaluate_folder(
    audio_folder,
    steering,
    steering_energy,
    az_deg,
    el_deg,
    output_dir,
    save_plots=False,
    plot_dpi=150,
):
    audio_files = []
    for extension in ("*.wav", "*.flac", "*.mp3"):
        audio_files.extend(glob.glob(os.path.join(audio_folder, extension)))
    audio_files.sort(key=natural_key)

    if not audio_files:
        raise ValueError(f"No audio files found in {audio_folder}")

    results = []
    total = len(audio_files)

    for index, audio_path in enumerate(audio_files, start=1):
        az, el, xyz = process_sample(
            audio_path,
            steering,
            steering_energy,
            az_deg,
            el_deg,
            output_dir=output_dir,
            save_plots=save_plots,
            plot_dpi=plot_dpi,
        )

        print(
            f"[{index}/{total}] {os.path.basename(audio_path)}: "
            f"az={az:.2f}°, el={el:.2f}°"
        )
        results.append(
            [
                os.path.basename(audio_path),
                round(float(xyz[0]), 6),
                round(float(xyz[1]), 6),
                round(float(xyz[2]), 6),
                round(az, 2),
                round(el, 2),
            ]
        )

    return results


def main():
    audio_folder = "/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/ivan_em32/processed_data/sh_wavs//RIRs_truncated"
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)

    order = 4

    # Plot generation is commonly slower than the numerical DOA calculation.
    # Leave this False for maximum throughput; enable it only when needed.
    save_plots = True
    plot_dpi = 300

    vecs = spa.grids.load_n_design(degree=124)
    az_grid = np.arctan2(vecs[:, 1], vecs[:, 0])
    zen_grid = np.arccos(np.clip(vecs[:, 2], -1.0, 1.0))

    _, _, az_deg, el_deg, steering, steering_energy = precompute_music_grid(
        order,
        az_grid,
        zen_grid,
    )

    results = evaluate_folder(
        audio_folder,
        steering,
        steering_energy,
        az_deg,
        el_deg,
        output_dir,
        save_plots=save_plots,
        plot_dpi=plot_dpi,
    )

    csv_path = os.path.join(output_dir, "motus_directions_music.csv")
    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "dx", "dy", "dz", "az", "el"])
        writer.writerows(results)

    print(f"\nCSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
