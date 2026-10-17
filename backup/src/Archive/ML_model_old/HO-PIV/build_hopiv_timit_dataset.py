import os
import csv
import sys
import json
import math
import random
import time
import warnings
import resource
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
import librosa
from scipy.io import wavfile
from scipy.signal import fftconvolve, resample_poly
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Configuration helpers
# -----------------------------------------------------------------------------

def get_slurm_task_info(n_items: int):
    """
    Return task_id, task_count, start, end for SLURM array execution.

    If no SLURM array is active, this returns task_id=0, task_count=1,
    start=0, end=n_items. This function splits the TOTAL work across
    SLURM array tasks.
    """
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    task_count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

    if task_count < 1:
        raise ValueError(f"Invalid SLURM_ARRAY_TASK_COUNT={task_count}")

    if task_id < 0 or task_id >= task_count:
        raise ValueError(
            f"Invalid SLURM_ARRAY_TASK_ID={task_id} for "
            f"SLURM_ARRAY_TASK_COUNT={task_count}"
        )

    if n_items < 0:
        raise ValueError(f"n_items must be non-negative, got {n_items}")

    items_per_task = n_items // task_count
    remainder = n_items % task_count

    start = task_id * items_per_task + min(task_id, remainder)
    end = start + items_per_task + (1 if task_id < remainder else 0)

    return task_id, task_count, start, end


def find_wavs_recursive(folder: str):
    folder = Path(folder)
    wavs = sorted(str(p) for p in folder.rglob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"No .wav files found recursively in {folder}")
    return wavs


def pcm_to_float32(x: np.ndarray) -> np.ndarray:
    """Convert wavfile output to mono/multichannel float32 in [-1, 1] where possible."""
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        # Use max(abs(min), max) to handle int16 asymmetry robustly.
        denom = max(abs(info.min), info.max)
        x = x.astype(np.float32) / float(denom)
    else:
        x = x.astype(np.float32)
    return x


def resample_if_needed(x: np.ndarray, orig_fs: int, target_fs: int, axis: int = -1) -> np.ndarray:
    if orig_fs == target_fs:
        return x.astype(np.float32, copy=False)
    g = math.gcd(orig_fs, target_fs)
    up = target_fs // g
    down = orig_fs // g
    return resample_poly(x, up, down, axis=axis).astype(np.float32)


# -----------------------------------------------------------------------------
# Audio/RIR loading
# -----------------------------------------------------------------------------

def load_speech_mono(path: str, target_fs: int) -> np.ndarray:
    fs, x = wavfile.read(path)
    x = pcm_to_float32(x)
    if x.ndim == 2:
        x = np.mean(x, axis=1)
    x = resample_if_needed(x, fs, target_fs, axis=0)
    # Avoid all-zero and preserve dynamic range reasonably.
    peak = np.max(np.abs(x)) + 1e-8
    x = 0.9 * x / peak
    return x.astype(np.float32)


def repeat_speech_with_random_pauses(x: np.ndarray, fs: int, rng: random.Random, max_pause_s: float = 0.5):
    """
    Reference-like source construction:
    silence_start + speech + silence_middle + same speech + silence_end.
    Pauses are exact zeros with random lengths in [0, max_pause_s].
    """
    pauses_s = [rng.uniform(0.0, max_pause_s) for _ in range(3)]
    pauses_n = [int(round(p * fs)) for p in pauses_s]
    z0 = np.zeros(pauses_n[0], dtype=np.float32)
    z1 = np.zeros(pauses_n[1], dtype=np.float32)
    z2 = np.zeros(pauses_n[2], dtype=np.float32)
    y = np.concatenate([z0, x.astype(np.float32), z1, x.astype(np.float32), z2])

    # Exact activity from the constructed dry source. Use a tiny threshold to
    # avoid zero crossings being interpreted as inactive samplewise.
    activity_sample = (np.abs(y) > 1e-5).astype(np.float32)
    return y.astype(np.float32), activity_sample, pauses_s


def candidate_rir_paths(rir_folder: str, name: str):
    base = Path(rir_folder)
    stem = Path(str(name)).stem
    raw = Path(str(name))
    candidates = []
    if raw.suffix:
        candidates.append(base / raw.name)
    else:
        for ext in (".wav", ".npy", ".npz"):
            candidates.append(base / f"{stem}{ext}")
    # Some datasets store nested files; fall back to recursive name match.
    for c in candidates:
        if c.exists():
            return c
    matches = list(base.rglob(f"{stem}.*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find RIR file for Name={name!r} in {rir_folder}")


def load_hoa_rir(rir_folder: str, name: str, target_fs: int, expected_channels: int) -> np.ndarray:
    path = candidate_rir_paths(rir_folder, name)
    if path.suffix.lower() == ".wav":
        fs, h = wavfile.read(path)
        h = pcm_to_float32(h)
        # wavfile gives (samples, channels). Convert to (channels, samples).
        if h.ndim == 1:
            h = h[None, :]
        else:
            h = h.T
        h = resample_if_needed(h, fs, target_fs, axis=1)
    elif path.suffix.lower() == ".npy":
        h = np.load(path).astype(np.float32)
    elif path.suffix.lower() == ".npz":
        data = np.load(path)
        key = "rir" if "rir" in data else list(data.keys())[0]
        h = data[key].astype(np.float32)
    else:
        raise ValueError(f"Unsupported RIR file type: {path}")

    # Ensure shape (channels, samples).
    if h.ndim != 2:
        raise ValueError(f"RIR {path} must be 2D after loading, got shape {h.shape}")
    if h.shape[0] != expected_channels and h.shape[1] == expected_channels:
        h = h.T
    if h.shape[0] < expected_channels:
        raise ValueError(f"RIR {path} has {h.shape[0]} channels, expected at least {expected_channels}")
    if h.shape[0] > expected_channels:
        h = h[:expected_channels]
    return h.astype(np.float32)


def convolve_hoa_track(dry: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Convolve mono dry speech with multichannel HOA RIR. Output (Q, samples)."""
    return fftconvolve(rir, dry[None, :], mode="full", axes=1).astype(np.float32)


# -----------------------------------------------------------------------------
# Reference-like HO-PIV feature extraction
# -----------------------------------------------------------------------------

def stft_hoa(x_hoa: np.ndarray, fs: int, n_fft: int = 1024, win_length: int = 640, hop_length: int = 320):
    """
    x_hoa: (Q, samples)
    Return complex STFT as (Q, F, T). For a 1 s / 16 kHz signal, librosa with
    center=False and these parameters yields T=47, so we pad to exactly 50.
    The paper reports T=50, F=513 for 1 s, 640 window, 320 hop, 1024 FFT.
    """
    specs = []
    for ch in range(x_hoa.shape[0]):
        spec = librosa.stft(
            x_hoa[ch],
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window="hann",
            center=True,
            pad_mode="constant",
        )
        specs.append(spec)
    A = np.stack(specs, axis=0).astype(np.complex64)  # (Q, F, T)

    # Force reference shape T=50 for 1 s segments.
    target_T = int(round(x_hoa.shape[1] / hop_length))
    if A.shape[-1] > target_T:
        A = A[..., :target_T]
    elif A.shape[-1] < target_T:
        A = np.pad(A, ((0, 0), (0, 0), (0, target_T - A.shape[-1])), mode="constant")
    return A


def order_energy_normalization(A: np.ndarray, order: int, eps: float = 1e-8) -> np.ndarray:
    """
    EN(k,l) = sum_n 1/(2n+1) sum_m |a_nm(k,l)|^2.
    Works for order-grouped SH channel layouts, e.g. ACN/N3D or SID-like grouping.
    A has shape (Q, F, T). Return (F, T).
    """
    E = np.zeros(A.shape[1:], dtype=np.float32)
    for n in range(order + 1):
        start = n * n
        end = (n + 1) * (n + 1)
        E += (np.sum(np.abs(A[start:end]) ** 2, axis=0) / float(2 * n + 1)).astype(np.float32)
    return np.maximum(E, eps)


def fill_empty_mel_filter_rows(mel: np.ndarray, empty_filter_mask: np.ndarray) -> np.ndarray:
    """
    Replace rows produced by empty mel filters with the nearest non-empty
    neighbouring mel row. This keeps n_mels fixed, e.g. 513, while avoiding
    constant floor-valued channels caused by empty filterbank rows.

    mel: (n_mels, T) power-domain mel spectrogram, before log compression.
    empty_filter_mask: (n_mels,) True for rows whose mel filter contains no FFT bins.
    """
    if not np.any(empty_filter_mask):
        return mel

    valid = np.flatnonzero(~empty_filter_mask)
    if len(valid) == 0:
        # Extremely defensive fallback; should not happen with normal settings.
        return mel

    filled = mel.copy()
    for i in np.flatnonzero(empty_filter_mask):
        nearest_valid = valid[np.argmin(np.abs(valid - i))]
        filled[i, :] = mel[nearest_valid, :]
    return filled


def logmel_per_sh_channel(x_hoa: np.ndarray, fs: int, n_fft: int = 1024, win_length: int = 640,
                          hop_length: int = 320, n_mels: int = 513):
    """
    Return log-mel features as (T, n_mels, Q).

    With n_mels=513, sr=16 kHz, and n_fft=1024, some librosa mel filters can
    be empty. Those empty mel rows are copied from the nearest non-empty
    neighbouring mel row before the logarithm is applied.
    """
    feats = []
    target_T = int(round(x_hoa.shape[1] / hop_length))

    mel_basis = librosa.filters.mel(
        sr=fs,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=0.0,
        fmax=fs / 2,
    )
    empty_filter_mask = np.sum(mel_basis, axis=1) == 0.0

    for ch in range(x_hoa.shape[0]):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Empty filters detected in mel frequency basis.*",
            )
            mel = librosa.feature.melspectrogram(
                y=x_hoa[ch],
                sr=fs,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window="hann",
                center=True,
                pad_mode="constant",
                n_mels=n_mels,
                power=2.0,
                fmin=0.0,
                fmax=fs / 2,
            )

        # Fill rows corresponding to empty mel filters before log compression.
        mel = fill_empty_mel_filter_rows(mel, empty_filter_mask)

        mel = np.log(np.maximum(mel, 1e-10)).astype(np.float32)  # (M, T)
        if mel.shape[-1] > target_T:
            mel = mel[..., :target_T]
        elif mel.shape[-1] < target_T:
            mel = np.pad(mel, ((0, 0), (0, target_T - mel.shape[-1])), mode="constant")
        feats.append(mel.T)
    return np.stack(feats, axis=-1).astype(np.float32)  # (T, M, Q)


def compute_hopiv_features(x_hoa: np.ndarray, fs: int, order: int) -> np.ndarray:
    """
    Reference HO-PIV input:
      - active and reactive HO-PIV: real/imag(-a0 * conj(a1..aQ-1)) / EN
      - concatenated with order-matched log-mel spectrograms.
    Output shape for order 4: (50, 513, 73) = 48 HO-PIV + 25 log-mel.
    """
    Q = (order + 1) ** 2
    if x_hoa.shape[0] != Q:
        raise ValueError(f"Expected {Q} HOA channels for order {order}, got {x_hoa.shape[0]}")

    A = stft_hoa(x_hoa, fs)  # (Q, 513, 50)
    a0 = A[0:1]
    hopiv_complex = -a0 * np.conj(A[1:])  # (Q-1, F, T)
    E = order_energy_normalization(A, order)[None, :, :]  # (1, F, T)

    hopiv_real = (np.real(hopiv_complex) / E).astype(np.float32)
    hopiv_imag = (np.imag(hopiv_complex) / E).astype(np.float32)
    hopiv = np.concatenate([hopiv_real, hopiv_imag], axis=0)  # (2(Q-1), F, T)
    hopiv = np.transpose(hopiv, (2, 1, 0))  # (T, F, C_hopiv)

    logmel = logmel_per_sh_channel(x_hoa, fs, n_mels=A.shape[1])  # (T, F, Q)
    X = np.concatenate([hopiv, logmel], axis=-1).astype(np.float32)
    return X


def minmax_normalize_features_0_1(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Apply 0-1 normalization per feature channel over the full time-frequency
    grid of the segment.

    X has shape (T, F, C), e.g. (50, 513, 73). For each feature channel c,
    min/max are computed over all T=50 frames and all F bins together:

        X_norm[:, :, c] = (X[:, :, c] - min(X[:, :, c]))
                          / (max(X[:, :, c]) - min(X[:, :, c]))

    This intentionally does NOT normalize each time frame independently.
    Constant channels are mapped to 0.
    """
    x_min = np.min(X, axis=(0, 1), keepdims=True)
    x_max = np.max(X, axis=(0, 1), keepdims=True)
    denom = x_max - x_min
    X_norm = np.where(denom > eps, (X - x_min) / denom, 0.0)
    return X_norm.astype(np.float32)


# -----------------------------------------------------------------------------
# Label generation
# -----------------------------------------------------------------------------

def frame_activity_from_sample_mask(mask: np.ndarray, start: int, fs: int, segment_len_samples: int,
                                    hop_length: int = 320, win_length: int = 640, n_frames: int = 50):
    """
    Frame-level activity from the dry constructed speech mask aligned to the
    convolved timeline. Reverberation tail alone is not counted as speaker active.
    """
    seg_mask = np.zeros(segment_len_samples, dtype=np.float32)
    src_start = max(start, 0)
    src_end = min(start + segment_len_samples, len(mask))
    if src_end > src_start:
        dst_start = src_start - start
        dst_end = dst_start + (src_end - src_start)
        seg_mask[dst_start:dst_end] = mask[src_start:src_end]

    active = np.zeros(n_frames, dtype=np.float32)
    for t in range(n_frames):
        center = t * hop_length
        lo = max(0, center - win_length // 2)
        hi = min(segment_len_samples, center + win_length // 2)
        if hi > lo and np.max(seg_mask[lo:hi]) > 0:
            active[t] = 1.0
    return active


def unit_direction_from_rir_row(rir_row) -> np.ndarray:
    dx = float(rir_row["x_source"]) - float(rir_row["x_mic"])
    dy = float(rir_row["y_source"]) - float(rir_row["y_mic"])
    dz = float(rir_row["z_source"]) - float(rir_row["z_mic"])
    v = np.array([dx, dy, dz], dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def build_accdoa_targets(activity_tracks, directions, max_output_sources: int, n_frames: int = 50):
    """Return (T, max_output_sources, 3); unused source slots are zero."""
    Y = np.zeros((n_frames, max_output_sources, 3), dtype=np.float32)
    for i, (act, direction) in enumerate(zip(activity_tracks, directions)):
        if i >= max_output_sources:
            break
        Y[:, i, :] = act[:, None] * direction[None, :]
    return Y


# -----------------------------------------------------------------------------
# Per-room/receiver processing
# -----------------------------------------------------------------------------

def process_one(args):
    (sample_id, room, receiver, group_df, rir_folder, speech_files, output_dir,
     fs, order, num_voice, max_output_sources, segment_len_s, seed) = args

    rng = random.Random(seed)

    Q = (order + 1) ** 2
    segment_len_samples = int(round(segment_len_s * fs))

    if len(group_df) < num_voice:
        raise ValueError(f"Group room={room}, receiver={receiver} only has {len(group_df)} RIRs; need {num_voice}")

    voice_rirs = group_df.sample(n=num_voice, random_state=seed).reset_index(drop=True)
    speech_paths = rng.sample(speech_files, k=num_voice) if len(speech_files) >= num_voice else [rng.choice(speech_files) for _ in range(num_voice)]

    conv_tracks = []
    dry_activity_masks = []
    directions = []
    pause_info = []

    for i in range(num_voice):
        rir_row = voice_rirs.iloc[i]
        speech = load_speech_mono(speech_paths[i], fs)
        speech_rep, activity_mask, pauses_s = repeat_speech_with_random_pauses(speech, fs, rng)
        rir = load_hoa_rir(rir_folder, rir_row["Name"], fs, Q)
        conv = convolve_hoa_track(speech_rep, rir)

        conv_tracks.append(conv)
        # Activity is aligned to dry signal start; pad with inactive reverb tail length.
        activity_padded = np.pad(activity_mask, (0, max(0, conv.shape[1] - len(activity_mask))), mode="constant")
        dry_activity_masks.append(activity_padded)
        directions.append(unit_direction_from_rir_row(rir_row))
        pause_info.append(pauses_s)

    max_len = max(t.shape[1] for t in conv_tracks)

    # Pick exactly one random 1 s window per room/receiver.
    # The start is guaranteed to be < duration - 1 s when the mixture is longer
    # than the requested segment length; otherwise we fall back to start 0 and pad.
    if max_len <= segment_len_samples:
        segment_start = 0
    else:
        segment_start = rng.randrange(0, max_len - segment_len_samples)

    seg_tracks = []
    activity_tracks = []
    for conv, mask in zip(conv_tracks, dry_activity_masks):
        if conv.shape[1] < max_len:
            conv = np.pad(conv, ((0, 0), (0, max_len - conv.shape[1])), mode="constant")
        seg = conv[:, segment_start:segment_start + segment_len_samples]
        if seg.shape[1] < segment_len_samples:
            seg = np.pad(seg, ((0, 0), (0, segment_len_samples - seg.shape[1])), mode="constant")
        seg_tracks.append(seg.astype(np.float32))
        activity_tracks.append(
            frame_activity_from_sample_mask(mask, segment_start, fs, segment_len_samples, n_frames=50)
        )

    mixture = np.sum(np.stack(seg_tracks, axis=0), axis=0).astype(np.float32)  # (Q, samples)
    peak = float(np.max(np.abs(mixture)))
    if peak > 1e-8:
        mixture *= (0.9 / peak)

    X = compute_hopiv_features(mixture, fs, order)  # (50, 513, C)
    X = minmax_normalize_features_0_1(X)  # per channel over all 50 frames and all frequency bins
    Y = build_accdoa_targets(activity_tracks, directions, max_output_sources=max_output_sources, n_frames=X.shape[0])

    activity = np.zeros((X.shape[0], max_output_sources), dtype=np.float32)
    for slot, act in enumerate(activity_tracks[:max_output_sources]):
        activity[:, slot] = act

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pt_path = output_dir / f"{sample_id}.pt"
    csv_path = output_dir / f"{sample_id}.csv"

    torch.save(
        {
            "X": torch.from_numpy(X),
            "Y": torch.from_numpy(Y),
            "activity": torch.from_numpy(activity),
            "meta": {
                "sample_id": sample_id,
                "room": int(room),
                "receiver": int(receiver),
                "fs": fs,
                "order": order,
                "num_voice": num_voice,
                "max_output_sources": max_output_sources,
                "segment_start_sample": int(segment_start),
                "segment_start_s": float(segment_start / fs),
                "segment_len_samples": int(segment_len_samples),
                "segment_len_s": float(segment_len_s),
                "feature": "HO-PIV+logmel",
                "normalization": "0-1 min-max per feature channel over full (time, frequency) grid",
                "normalization_axes": [0, 1],
                "X_shape": list(X.shape),
                "Y_shape": list(Y.shape),
                "activity_shape": list(activity.shape),
            },
        },
        pt_path,
    )

    meta_cols = [c for c in [
        "Name", "RT", "c50", "x_room", "y_room", "z_room",
        "x_source", "y_source", "z_source", "x_mic", "y_mic", "z_mic", "status"
    ] if c in voice_rirs.columns]

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = [
            "sample_id", "segment_idx", "segment_start_sample", "segment_start_s",
            "frame_idx", "frame_start_sample", "frame_start_s", "slot",
            "speech_path", "active", "accdoa_x", "accdoa_y", "accdoa_z",
            "dx", "dy", "dz", "az_deg", "el_deg",
            "pause_start_s", "pause_middle_s", "pause_end_s"
        ] + meta_cols
        writer.writerow(header)

        hop_length = 320
        segment_idx = 0
        for frame_idx in range(Y.shape[0]):
            frame_start_sample = int(segment_start + frame_idx * hop_length)
            frame_start_s = frame_start_sample / fs

            for slot in range(max_output_sources):
                if slot < num_voice:
                    dx, dy, dz = directions[slot].tolist()
                    az = float(np.degrees(np.arctan2(dy, dx)))
                    el = float(np.degrees(np.arcsin(dz / (np.linalg.norm(directions[slot]) + 1e-8))))
                    source_meta = [voice_rirs.iloc[slot][c] for c in meta_cols]
                    speech_path = speech_paths[slot]
                    pauses = pause_info[slot]
                else:
                    dx = dy = dz = 0.0
                    az = el = ""
                    source_meta = [""] * len(meta_cols)
                    speech_path = ""
                    pauses = ["", "", ""]

                writer.writerow([
                    sample_id,
                    int(segment_idx), int(segment_start), segment_start / fs,
                    int(frame_idx), frame_start_sample, frame_start_s,
                    int(slot), speech_path, int(activity[frame_idx, slot]),
                    float(Y[frame_idx, slot, 0]),
                    float(Y[frame_idx, slot, 1]),
                    float(Y[frame_idx, slot, 2]),
                    dx, dy, dz, round(az, 3) if az != "" else "", round(el, 3) if el != "" else "",
                    pauses[0], pauses[1], pauses[2],
                ] + source_meta)

    return str(pt_path), str(csv_path), X.shape, Y.shape, activity.shape, int(segment_start)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(csv_file, rir_folder, speech_folder, output_dir,
         order=4, fs=16000, segment_len_s=1.0,
         num_voice=3, max_output_sources=4, num_workers=None,
         max_groups=None):
    start_time = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_file)
    if "Name" not in df.columns:
        raise ValueError("CSV must contain a 'Name' column for RIR file lookup")

    # Expected naming pattern: R<room>_P<receiver>_S<source>...
    df["r"] = df["Name"].astype(str).str.extract(r"R(\d+)_")[0].astype(int)
    df["p"] = df["Name"].astype(str).str.extract(r"_P(\d+)_")[0].astype(int)
    df["s"] = df["Name"].astype(str).str.extract(r"_S(\d+)_")[0].astype(int)

    grouped_all = list(df.groupby(["r", "p"]))

    # Optional debug subset BEFORE SLURM splitting.
    # Therefore max_groups is the TOTAL number of samples across all array jobs,
    # not the number of samples per array job.
    if max_groups is not None:
        grouped_all = grouped_all[:max_groups]

    n_total = len(grouped_all)
    task_id, task_count, start_idx, end_idx = get_slurm_task_info(n_total)
    grouped = grouped_all[start_idx:end_idx]

    speech_files = find_wavs_recursive(speech_folder)
    print(f"Found {len(speech_files)} TIMIT wavs under {speech_folder}", file=sys.stderr)
    print(f"Total room/receiver groups after max_groups: {n_total}", file=sys.stderr)
    print(
        f"SLURM task {task_id}/{task_count} processing "
        f"groups {start_idx}:{end_idx} ({len(grouped)} groups)",
        file=sys.stderr,
    )

    if len(grouped) == 0:
        print(
            f"SLURM task {task_id}/{task_count} has no groups to process. "
            f"This can happen when task_count > n_total.",
            file=sys.stderr,
        )

    args_list = []
    for local_idx, ((room, receiver), group) in enumerate(grouped):
        # No task labels and no merging: one flat output pair per room/receiver.
        # sample_id stays global and deterministic, so different SLURM tasks can
        # safely write into the same output_dir as long as the group split is disjoint.
        sample_id = f"R{int(room):04d}_P{int(receiver):02d}"

        # Use a stable deterministic seed. Python's built-in hash() is deliberately
        # randomized between processes unless PYTHONHASHSEED is fixed, so avoid it.
        global_idx = start_idx + local_idx
        seed = (
            int(room) * 1_000_003
            + int(receiver) * 10_007
            + global_idx
        ) % (2 ** 32)

        args_list.append((
            sample_id, int(room), int(receiver), group.copy(), rir_folder, speech_files,
            str(output_dir), fs, order, num_voice, max_output_sources, segment_len_s, seed
        ))

    if num_workers is None:
        num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))

    results = []
    if args_list:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for res in tqdm(executor.map(process_one, args_list, chunksize=1), total=len(args_list)):
                results.append(res)

    # In a SLURM array, each task writes its own manifest to avoid races and
    # accidental overwrites. The .pt/.csv sample files themselves remain flat.
    if task_count > 1:
        manifest_path = output_dir / f"manifest_task{task_id:04d}.csv"
    else:
        manifest_path = output_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pt_path", "csv_path", "X_shape", "Y_shape", "activity_shape", "segment_start_sample"])
        for pt_path, csv_path, x_shape, y_shape, activity_shape, segment_start in results:
            writer.writerow([
                pt_path, csv_path,
                json.dumps(list(x_shape)),
                json.dumps(list(y_shape)),
                json.dumps(list(activity_shape)),
                int(segment_start),
            ])

    rusage_s = resource.getrusage(resource.RUSAGE_SELF)
    rusage_c = resource.getrusage(resource.RUSAGE_CHILDREN)
    print(
        f"Wrote {len(results)} room/receiver samples to {output_dir} "
        f"for SLURM task {task_id}/{task_count}",
        file=sys.stderr,
    )
    print(f"Manifest: {manifest_path}", file=sys.stderr)
    print(f"Walltime {time.time() - start_time:.2f}s", file=sys.stderr)
    print(f"User time {rusage_s.ru_utime + rusage_c.ru_utime:.2f}s", file=sys.stderr)
    print(f"System time {rusage_s.ru_stime + rusage_c.ru_stime:.2f}s", file=sys.stderr)
    print(f"MaxRSS {(rusage_s.ru_maxrss + rusage_c.ru_maxrss) / 2 ** 20:.3f} GiB", file=sys.stderr)


if __name__ == "__main__":
    BASE_DIR = "../../../data/harp/R2000_P10_S5_Order4_24kHz_shoeBox_rtProfilesMIT_ISM"
    DATASET_NAME = "hopiv_timit_1s_random_window_pt"

    main(
        csv_file=os.path.join(BASE_DIR, "Generated_HOA_SRIR_data.csv"),
        rir_folder=os.path.join(BASE_DIR, "RIRs"),
        speech_folder="../../../data/DARPA_TIMIT/data/TRAIN",
        output_dir=os.path.join(BASE_DIR, DATASET_NAME + "_debug"),
        order=4,
        fs=16000,
        segment_len_s=1.0,
        num_voice=3,
        max_output_sources=3,
        max_groups=10000,
    )
