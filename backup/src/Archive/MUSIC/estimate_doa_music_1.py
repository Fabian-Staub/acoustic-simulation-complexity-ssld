import torch
import numpy as np
import spaudiopy as spa
import matplotlib.pyplot as plt
import os
import sys
import time
import resource
import glob
from scipy.io import wavfile
from scipy.signal import find_peaks
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from utils import detect_active_regions_relative, random_segment_with_activity, compute_scm, compute_scm_avg_log, convolve_rirs, apply_order_weighting
import re
import csv

def natural_key(path):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', os.path.basename(path))]
# =========================================================
# DATA LOADING
# =========================================================
def load_dataset(pt_path):
    data = torch.load(pt_path)
    return data["X"].numpy(), data["Y"].numpy()


def load_direct_sound_from_rir(audio_path, peak_window=64, threshold_ratio=0.1):
    """
    Extract direct sound from a multichannel RIR using channel 0 for detection,
    and apply the same time window to all channels.

    Parameters
    ----------
    audio_path : str
    peak_window : int
        Window size around direct arrival (samples)
    threshold_ratio : float
        Peak detection threshold (relative to max)

    Returns
    -------
    fs : int
    direct_rir : np.ndarray (channels, samples)
    onset : int
    """

    fs, rir = wavfile.read(audio_path)
    rir = rir.T

    # ensure shape = (channels, samples)
    if rir.ndim == 1:
        rir = rir[np.newaxis, :]

    # --------------------------------------------------
    # 1. detect onset using FIRST CHANNEL ONLY
    # --------------------------------------------------
    ch0 = rir[0]
    ch0 = ch0 / (np.max(np.abs(ch0)) + 1e-12)

    peaks, _ = find_peaks(np.abs(ch0), height=threshold_ratio)

    if len(peaks) > 0:
        onset = peaks[0]
    else:
        onset = np.argmax(np.abs(ch0) > threshold_ratio)

    # fallback safety
    if onset is None:
        onset = 0

    # --------------------------------------------------
    # 2. define window around onset
    # --------------------------------------------------
    half_win = peak_window // 2
    start = max(0, onset - half_win)
    end = min(rir.shape[1], onset + half_win)

    # --------------------------------------------------
    # 3. apply SAME window to ALL channels
    # --------------------------------------------------
    direct_rir = rir[:, start:end]

    return fs, direct_rir

def plot_signal(signal, fs, save_path="direct_sound.png", title="Direct Sound RIR"):
    """
    Plot multichannel signal and save figure.
    signal: (channels, samples)
    """

    channels, n = signal.shape
    t = np.arange(n) / fs

    plt.figure(figsize=(10, 4))

    for ch in range(channels):
        plt.plot(t, signal[ch], label=f"Ch {ch}")

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    # ensure folder exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# ANGULAR DISTANCE
# =========================================================
def angular_distance(a, b):
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    dot = np.clip(np.dot(a, b), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


# =========================================================
# EIG / SUBSPACE
# =========================================================
def eig_decompose(scm):
    evals, evecs = np.linalg.eigh(scm)
    idx = np.argsort(evals)[::-1]
    return evals[idx].real, evecs[:, idx]


def estimate_subspaces(evecs, num_sources):
    return evecs[:, num_sources:]



# =========================================================
# DOA HELPERS
# =========================================================
def estimate_doa_topk(AZ, ZEN, P, K, min_sep_deg=15):
    flat = P.ravel()
    idx_sorted = np.argsort(flat)[::-1]

    az_all = np.rad2deg(AZ.ravel())
    el_all = 90 - np.rad2deg(ZEN.ravel())

    selected = []

    def dist(a1, e1, a2, e2):
        return np.sqrt((a1 - a2) ** 2 + (e1 - e2) ** 2)

    for idx in idx_sorted:
        cand = (az_all[idx], el_all[idx])

        if all(dist(cand[0], cand[1], s[0], s[1]) >= min_sep_deg for s in selected):
            selected.append(cand)

        if len(selected) >= K:
            break

    if len(selected) == 0:
        return np.array([]), np.array([])

    az = np.array([s[0] for s in selected])
    el = np.array([s[1] for s in selected])

    return az, el


def azel_to_cartesian(az_deg, el_deg):
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)

    x = np.cos(el) * np.cos(az)
    y = np.cos(el) * np.sin(az)
    z = np.sin(el)

    return np.stack([x, y, z], axis=-1)


# =========================================================
# PRECOMPUTE GRID
# =========================================================
def precompute_music_grid(order, az_grid, zen_grid):
    AZ, ZEN = np.meshgrid(az_grid, zen_grid)

    Y = spa.sph.sh_matrix(
        order,
        AZ.ravel(),
        ZEN.ravel(),
        sh_type="real"
    ).T

    return AZ, ZEN, Y, AZ.shape


def plot_music(AZ, ZEN, P, az_est=None, el_est=None, save_path="music_map.png"):
    P_db = 10 * np.log10(P / np.max(P) + 1e-12)

    az_deg = np.rad2deg(AZ)
    el_deg = 90 - np.rad2deg(ZEN)  # convert colatitude → elevation

    plt.figure(figsize=(8, 4))
    plt.tricontourf(
        np.rad2deg(AZ.ravel()),
        90 - np.rad2deg(ZEN.ravel()),
        P_db.ravel(),
        levels=50
    )
    plt.colorbar(label="dB")
    
    if az_est is not None and el_est is not None:
        plt.scatter(
            az_est,
            el_est,
            c='red',
            s=100,
            marker='x',
            label='Estimated DOA'
        )
    
    plt.legend()
    plt.xlabel("Azimuth (deg)")
    plt.ylabel("Elevation (deg)")
    plt.title("MUSIC Spectrum")

    plt.gca().invert_xaxis()  # invert azimuth axis

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def process_sample(audio_path, Y, AZ, ZEN, grid_shape, output_dir):
    """
    Single-source MUSIC processing for one audio file.
    """

    # -----------------------------------------------------
    # 1. Load audio + compute SCM per band
    # -----------------------------------------------------
    fs, signal = load_direct_sound_from_rir(audio_path, peak_window=64, threshold_ratio=0.1)
    scm_bands = compute_scm_avg_log(signal, fs, freq_range=(2000, 8000), num_scms=10)
    scm_bands = scm_bands * 2.0 - 1.0

    spectra = np.zeros(grid_shape, dtype=np.float64)
    Yc = Y.conj()

    # -----------------------------------------------------
    # 2. MUSIC over frequency bands (SINGLE SOURCE)
    # -----------------------------------------------------
    for b in range(scm_bands.shape[0]):
        scm = scm_bands[b]

        eigvals, eigvecs = eig_decompose(scm)

        # single source => noise subspace = all but largest eigenvector
        idx = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, idx]

        signal_subspace = eigvecs[:, :1]
        noise_subspace = eigvecs[:, 1:]

        Qn = noise_subspace
        Pn = Qn @ Qn.conj().T

        denom = np.sum((Yc * (Pn @ Y)), axis=0).real
        spectra += (1.0 / (denom + 1e-12)).reshape(grid_shape)

    P_avg = spectra / scm_bands.shape[0]

    # -----------------------------------------------------
    # 3. DOA estimation (TOP-1 only)
    # -----------------------------------------------------
    az_est, el_est = estimate_doa_topk(AZ, ZEN, P_avg, K=1)
    pred_xyz = azel_to_cartesian(az_est, el_est)

    if len(pred_xyz) == 0:
        pred_xyz = np.zeros((0, 3))

    # -----------------------------------------------------
    # 4. Print result
    # -----------------------------------------------------
    print("\n====================================")
    print(f"Audio: {os.path.basename(audio_path)}")
    print("Estimated DOA:")

    if az_est is None or len(az_est) == 0:
        print("  None")
    else:
        az = float(az_est[0])
        el = float(el_est[0])

        print(f"  Az: {az:.2f}°  El: {el:.2f}°")

        if pred_xyz.shape[0] > 0:
            print(f"  Cartesian: {pred_xyz[0]}")
        else:
            print("  Cartesian: None")

    # -----------------------------------------------------
    # 5. Plot MUSIC map
    # -----------------------------------------------------
    basename = os.path.splitext(os.path.basename(audio_path))[0]
    out_path = os.path.join(output_dir, f"{basename}_music.png")
    direct_out_path = os.path.join(output_dir, f"{basename}_direct_sound.png")

    plot_signal(signal, fs, save_path=direct_out_path)

    plot_music(
        AZ,
        ZEN,
        P_avg,
        az_est=az_est,
        el_est=el_est,
        save_path=out_path
    )

    return az_est, el_est, pred_xyz


# =========================================================
# =============== FOLDER EVALUATION LOOP ==================
# =========================================================
def evaluate_folder(audio_folder, Y, AZ, ZEN, grid_shape, output_dir):
    """
    Iterate over all audio files in folder and run MUSIC.
    """

    audio_files = sorted(
        glob.glob(os.path.join(audio_folder, "*.wav")) +
        glob.glob(os.path.join(audio_folder, "*.flac")) +
        glob.glob(os.path.join(audio_folder, "*.mp3"))
    )
    audio_files = sorted(audio_files, key=natural_key)

    if len(audio_files) == 0:
        raise ValueError(f"No audio files found in {audio_folder}")

    all_results = []

    for i, audio_path in enumerate(audio_files):

        az_est, el_est, pred_xyz = process_sample(
            audio_path,
            Y,
            AZ,
            ZEN,
            grid_shape,
            output_dir
        )

        # handle failed estimation
        dx, dy, dz = [round(float(v), 6) for v in pred_xyz[0]]

        az = round(float(az_est[0]), 2) if len(az_est) > 0 else np.nan
        el = round(float(el_est[0]), 2) if len(el_est) > 0 else np.nan

        all_results.append([
            os.path.basename(audio_path),
            dx,
            dy,
            dz,
            az,
            el
        ])

    return all_results


# =========================================================
# =========================== MAIN =========================
# =========================================================
if __name__ == "__main__":

    audio_folder = "/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/arni_em32/sh_wavs/RIRs_truncated"
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)

    order = 4

    vecs = spa.grids.load_t_design(degree=21)
    az_grid = np.arctan2(vecs[:, 1], vecs[:, 0])
    zen_grid = np.arccos(vecs[:, 2])

    AZ, ZEN, Y, grid_shape = precompute_music_grid(
        order,
        az_grid,
        zen_grid
    )

    results = evaluate_folder(
        audio_folder,
        Y,
        AZ,
        ZEN,
        grid_shape,
        output_dir
    )

    csv_path = os.path.join(output_dir, "arni_directions_music.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "Name",
            "dx",
            "dy",
            "dz",
            "az",
            "el"
        ])

        writer.writerows(results)

    print(f"\nCSV saved to: {csv_path}")