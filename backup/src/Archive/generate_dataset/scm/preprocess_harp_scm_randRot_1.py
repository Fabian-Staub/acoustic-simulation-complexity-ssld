import os
import numpy as np
import pandas as pd
import pyfar as pf
import torch
from scipy.io import wavfile
import random
from scipy.signal import fftconvolve
import csv
import sys
import resource
import time
from tqdm import tqdm
import librosa
from concurrent.futures import ProcessPoolExecutor
from spharpy.transforms import SphericalHarmonicRotation
from spharpy import SphericalHarmonicSignal, SphericalHarmonicDefinition


# =========================
# Geometry utils
# =========================
def rotate_xy(x, y, theta):
    c, s = np.cos(theta), np.sin(theta)
    return x * c - y * s, x * s + y * c


def rotate_point(row, theta):
    x_s, y_s = rotate_xy(row['x_source'], row['y_source'], theta)
    x_m, y_m = rotate_xy(row['x_mic'], row['y_mic'], theta)

    row = row.copy()
    row['x_source'], row['y_source'] = x_s, y_s
    row['x_mic'], row['y_mic'] = x_m, y_m
    return row


# =========================
# Active region detection
# =========================
def detect_active_regions_relative(signal, fs, frame_len=512, hop_len=256, rel_threshold=0.1):
    energy = librosa.feature.rms(y=signal, frame_length=frame_len, hop_length=hop_len)[0]
    rel_energy = energy / (np.max(energy) + 1e-8)
    active_frames = np.where(rel_energy > rel_threshold)[0]

    if len(active_frames) == 0:
        return []

    active_regions = []
    start = active_frames[0]
    prev = active_frames[0]

    for f in active_frames[1:]:
        if f != prev + 1:
            active_regions.append((start * hop_len, (prev + 1) * hop_len))
            start = f
        prev = f

    active_regions.append((start * hop_len, (prev + 1) * hop_len))
    return active_regions


def random_segment_with_activity(sig, segment_len_samples, active_regions,
                                 min_active_ratio=0.6, max_trials=10):
    if len(active_regions) == 0:
        return 0

    for _ in range(max_trials):
        candidates = [r for r in active_regions if (r[1] - r[0]) >= segment_len_samples * min_active_ratio]

        if candidates:
            start, end = random.choice(candidates)
            max_start = max(start, end - segment_len_samples)
            start_sample = random.randint(start, max_start)
        else:
            start, end = max(active_regions, key=lambda r: r[1] - r[0])
            start_sample = start

        active_count = sum(
            max(0, min(r_end, start_sample + segment_len_samples)
                - max(r_start, start_sample))
            for r_start, r_end in active_regions
        )

        if active_count / segment_len_samples >= min_active_ratio:
            return start_sample

    return start_sample


# =========================
# SCM
# =========================
def compute_scm(signal_bands):
    n_bands, n_channels, n_samples = signal_bands.time.shape
    cov = np.zeros((n_bands, n_channels, n_channels), dtype=np.float32)

    for b in range(n_bands):
        X = signal_bands.time[b]
        cov[b] = X @ X.T

        mn, mx = cov[b].min(), cov[b].max()
        cov[b] = (cov[b] - mn) / (mx - mn + 1e-8)

    return cov


# =========================
# HOA rotation (CORRECT)
# =========================
def rotate_hoa(signal, theta, fs=24000):
    sh = SphericalHarmonicSignal(
        data=signal,
        sampling_rate=fs,
        basis_type="real",          # or "complex"
        normalization="N3D",
        channel_convention="ACN",
        condon_shortley=True,
        domain="time"
    )

    R = SphericalHarmonicRotation.from_euler('z', theta)

    rotated = R * sh
    return np.squeeze(rotated._data, axis=0).astype(np.float32)


# =========================
# RIR convolution
# =========================
def convolve_rirs(rirs, sigs, rir_folder):
    out_list = []

    for rir_row, sig in zip(rirs.itertuples(), sigs):
        rir_path = os.path.join(rir_folder, rir_row.Name)
        sr, rir = wavfile.read(rir_path)

        if np.issubdtype(rir.dtype, np.integer):
            rir = rir.astype(np.float32) / np.iinfo(rir.dtype).max
        else:
            rir = rir.astype(np.float32)

        if rir.ndim == 1:
            rir = rir[:, None]
        rir = rir.T  # (channels, samples)

        sig = sig.astype(np.float32)

        out = np.stack(
            [fftconvolve(sig, rir_ch, mode='full') for rir_ch in rir],
            axis=0
        )

        out_list.append(out.astype(np.float32))

    return out_list


# =========================
# PROCESS GROUP
# =========================
def process_group(args):
    (room, receiver, group, rir_folder, speech_files, noise_files,
     fs, segment_len_samples, min_voice, max_voice,
     max_noise, noise_volume_ratio, num_fractions, freq_range,
     repeat_idx, seed, random_rotation) = args

    random.seed(seed)
    np.random.seed(seed)

    theta = 0.0
    if random_rotation:
        theta = random.uniform(0, 2*np.pi)

    # -------------------------
    # Sample RIRs
    # -------------------------
    num_voice = random.randint(min_voice, max_voice)
    voice_rirs = group.sample(num_voice)
    remaining = group.drop(voice_rirs.index)

    num_noise = random.randint(0, min(max_noise, len(remaining)))
    noise_rirs = remaining.sample(num_noise) if num_noise > 0 else pd.DataFrame(columns=group.columns)

    # -------------------------
    # Signals
    # -------------------------
    voice_sigs = [
        0.9 * librosa.load(random.choice(speech_files), sr=fs)[0]
        for _ in range(num_voice)
    ]

    noise_sigs = [
        0.9 * noise_volume_ratio * librosa.load(random.choice(noise_files), sr=fs)[0]
        for _ in range(num_noise)
    ]

    # -------------------------
    # Convolution
    # -------------------------
    voice_conv = convolve_rirs(voice_rirs, voice_sigs, rir_folder)
    noise_conv = convolve_rirs(noise_rirs, noise_sigs, rir_folder) if num_noise > 0 else []

    # -------------------------
    # Segment extraction
    # -------------------------
    segments = []

    for sig_list in [voice_conv, noise_conv]:
        for sig in sig_list:
            mono = np.mean(sig, axis=0)

            start = random_segment_with_activity(
                mono, segment_len_samples,
                detect_active_regions_relative(mono, fs)
            )

            seg = sig[:, start:start + segment_len_samples]

            if seg.shape[1] < segment_len_samples:
                pad = segment_len_samples - seg.shape[1]
                seg = np.pad(seg, ((0, 0), (0, pad)))

            segments.append(seg.astype(np.float32))

    # -------------------------
    # HOA mix
    # -------------------------
    total_sum = np.sum(segments, axis=0).astype(np.float32)

    # -------------------------
    # HOA rotation (IMPORTANT)
    # -------------------------
    if random_rotation:
        total_sum = rotate_hoa(total_sum, theta)

    # safety normalization
    peak = np.max(np.abs(total_sum))
    if peak > 1.0:
        total_sum *= (0.99 / peak)

    ambi = total_sum.T.astype(np.float32)
    output_wav_path = os.path.join("output_ambisonics.wav")
    wavfile.write(output_wav_path, fs, ambi)
    print(f"Saved ambisonics WAV: {output_wav_path}")

    # -------------------------
    # SCM
    # -------------------------
    signal_obj = pf.Signal(total_sum, fs)
    bands = pf.dsp.filter.fractional_octave_bands(
        signal_obj,
        num_fractions=num_fractions,
        frequency_range=freq_range,
        order=15
    )
    cov = compute_scm(bands)

    # -------------------------
    # Voice directions
    # -------------------------
    voice_dirs = []
    for rir in voice_rirs.itertuples():
        dx = rir.x_source - rir.x_mic
        dy = rir.y_source - rir.y_mic
        dz = rir.z_source - rir.z_mic

        r = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-8
        x, y = dx / r, dy / r

        if random_rotation:
            x, y = rotate_xy(x, y, theta)

        voice_dirs.append([x, y, dz / r])

    while len(voice_dirs) < 3:
        voice_dirs.append([0.0, 0.0, 0.0])

    voice_dirs = np.array(voice_dirs, dtype=np.float32)

    # -------------------------
    # CSV metadata
    # -------------------------
    meta_cols = ['Name','RT','c50','x_room','y_room','z_room',
                 'x_source','y_source','z_source',
                 'x_mic','y_mic','z_mic','status']

    all_rirs = pd.concat([voice_rirs, noise_rirs], ignore_index=True)

    Y_sample = []
    for idx in range(5):
        row = np.zeros(1 + 3 + len(meta_cols), dtype=object)

        if idx < len(all_rirs):
            rir = all_rirs.iloc[idx]

            row[0] = 1.0 if idx < len(voice_rirs) else 0.0

            dx = rir.x_source - rir.x_mic
            dy = rir.y_source - rir.y_mic
            dz = rir.z_source - rir.z_mic

            r = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-8
            x, y = dx / r, dy / r

            if random_rotation:
                x, y = rotate_xy(x, y, theta)

            row[1:4] = [x, y, dz / r]
            row[4:] = rir[meta_cols].values

        else:
            row[1:4] = [0.0, 0.0, 0.0]

        Y_sample.append(row)

    return cov, np.array(Y_sample, dtype=object), voice_dirs


# =========================
# MAIN
# =========================
def main(csv_file, csv_output_file, rir_folder, speech_folder, noise_folder, output_file,
         fs=24000, segment_len_s=1.0,
         min_voice=1, max_voice=3,
         max_noise=2,
         noise_volume_ratio=1.0,
         num_repeats=1,
         random_rotation=False,
         num_fractions=1,
         freq_range=(100, 8000)):

    segment_len_samples = int(segment_len_s * fs)

    df = pd.read_csv(csv_file, nrows=5)  # <-- adjust this if you want to use more/less RIRs
    df['r'] = df['Name'].str.extract(r'R(\d+)_')[0].astype(int)
    df['p'] = df['Name'].str.extract(r'_P(\d+)_')[0].astype(int)

    grouped = df.groupby(['r', 'p'])

    speech_files = [os.path.join(speech_folder, f) for f in os.listdir(speech_folder)]
    noise_files = [os.path.join(noise_folder, f) for f in os.listdir(noise_folder)]

    X_list, Y_list, V_list = [], [], []

    num_workers = os.cpu_count()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:

        args_list = []

        for (room, receiver), group in grouped:
            for rep in range(num_repeats):
                seed = hash((room, receiver, rep)) % (2**32)

                args_list.append(
                    (room, receiver, group, rir_folder,
                     speech_files, noise_files,
                     fs, segment_len_samples,
                     min_voice, max_voice,
                     max_noise, noise_volume_ratio,
                     num_fractions, freq_range,
                     rep, seed, random_rotation)
                )

        for cov, Y, V in tqdm(executor.map(process_group, args_list), total=len(args_list)):
            X_list.append(cov)
            Y_list.append(Y)
            V_list.append(V)

    torch.save({
        "X": torch.tensor(np.array(X_list, dtype=np.float32)),
        "Y": torch.tensor(np.array(V_list, dtype=np.float32))
    }, output_file)


# =========================
# RUN
# =========================
if __name__ == "__main__":
    csv_file = "../../../data/harp/R800_P10_S5_Order4_24kHz_shoeBox_ISM/Generated_HOA_SRIR_data.csv"
    csv_output_file = "../../../data/harp/R800_P10_S5_Order4_24kHz_shoeBox_ISM/harp_dataset_scm_multitrack_rot.csv"
    rir_folder = "../../../data/harp/R800_P10_S5_Order4_24kHz_shoeBox_ISM/RIRs"
    speech_folder = "../../../data/audio_exports_Harvard_Sets"
    noise_folder = "../../../data/noises"
    output_file = "../../../data/harp/R800_P10_S5_Order4_24kHz_shoeBox_ISM/harp_dataset_scm_multitrack_rot.pt"
    
    main(
        csv_file, csv_output_file, rir_folder, speech_folder, noise_folder, output_file,
        fs=24000, segment_len_s=1.0,
        min_voice=1, max_voice=1,
        max_noise=0,
        noise_volume_ratio=0,
        num_repeats=1,
        random_rotation=True,
        num_fractions=1,
        freq_range=(100, 8000)
    )