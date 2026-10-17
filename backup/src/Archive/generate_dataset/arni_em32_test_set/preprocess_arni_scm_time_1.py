import os
import numpy as np
import pandas as pd
import pyfar as pf
import torch
from scipy.io import wavfile
import random
from scipy.signal import fftconvolve
import sys
import resource
import time
from tqdm import tqdm
import librosa
from concurrent.futures import ProcessPoolExecutor
from scipy.signal import resample_poly

# =========================
# Activity detection
# =========================
def detect_active_regions_relative(signal, fs, frame_len=512, hop_len=256, rel_threshold=0.1):
    energy = librosa.feature.rms(y=signal, frame_length=frame_len, hop_length=hop_len)[0]
    max_energy = np.max(energy) + 1e-8
    rel_energy = energy / max_energy
    active_frames = np.where(rel_energy > rel_threshold)[0]

    if len(active_frames) == 0:
        return []

    active_regions = []
    start = active_frames[0]
    prev = active_frames[0]

    for f in active_frames[1:]:
        if f != prev + 1:
            active_regions.append((start*hop_len, (prev+1)*hop_len))
            start = f
        prev = f

    active_regions.append((start*hop_len, (prev+1)*hop_len))
    return active_regions


def random_segment_with_activity(sig, segment_len_samples, active_regions, min_active_ratio=0.6, max_trials=10):
    if len(active_regions) == 0:
        return 0

    for _ in range(max_trials):
        candidates = [r for r in active_regions if (r[1]-r[0]) >= segment_len_samples * min_active_ratio]

        if candidates:
            start, end = random.choice(candidates)
            max_start = max(start, end - segment_len_samples)
            start_sample = random.randint(start, max_start)
        else:
            start, end = max(active_regions, key=lambda r: r[1]-r[0])
            start_sample = start

        active_count = sum(
            max(0, min(r_end, start_sample + segment_len_samples) - max(r_start, start_sample))
            for r_start, r_end in active_regions
        )

        if active_count / segment_len_samples >= min_active_ratio:
            return start_sample

    return start_sample


# =========================
# SCM
# =========================
def compute_scm(total_sum, segment_splits):
    # =========================
    # Split signal + SCM → time series (upper triangle)
    # =========================
    n_channels, total_len = total_sum.shape
    split_len = total_len // segment_splits
    tri_idx = np.triu_indices(n_channels)

    scm_time_series = []

    for i in range(segment_splits):
        start = i * split_len
        end = start + split_len if i < segment_splits - 1 else total_len

        segment = total_sum[:, start:end]

        if segment.shape[1] < 2:
            cov = np.zeros((n_channels, n_channels), dtype=np.float32)
        else:
            cov = segment @ segment.T

            # Normalize per split
            min_val = cov.min()
            max_val = cov.max()
            cov = (cov - min_val) / (max_val - min_val + 1e-8)

        # Extract upper triangle (including diagonal)
        cov_vec = cov[tri_idx]  # shape: (num_scm_channels,)

        scm_time_series.append(cov_vec.astype(np.float32))

    # Shape: (time_steps, num_scm_channels)
    scm_time_series = np.stack(scm_time_series, axis=0)

    # Transpose → (num_scm_channels, time_steps)
    return scm_time_series.T


# =========================
# Process one group
# =========================
def process_group(args):
    (room, position, group, rir_folder, speech_files,
     fs, segment_len_samples, segment_splits, min_voice, max_voice,
     num_fractions, freq_range, seed) = args

    random.seed(seed)
    np.random.seed(seed)

    # Select RIRs
    num_voice = random.randint(min_voice, max_voice)
    selected_rirs = group.sample(n=min(num_voice, len(group)))

    # Load speech
    speech_signals = [
        librosa.load(random.choice(speech_files), sr=fs)[0]
        for _ in range(len(selected_rirs))
    ]

    speech_signals = [
        0.9 * sig / (np.max(np.abs(sig)) + 1e-8)
        for sig in speech_signals
    ]

    # Convolution
    convolved = []

    for rir_row, sig in zip(selected_rirs.itertuples(), speech_signals):
        rir_path = os.path.join(rir_folder, rir_row.Name)
        sr, rir = wavfile.read(rir_path)

        if np.issubdtype(rir.dtype, np.integer):
            rir = rir.astype(np.float32) / np.iinfo(rir.dtype).max
        else:
            rir = rir.astype(np.float32)

        if rir.ndim == 1:
            rir = rir[:, None]

        if sr != fs:
            rir = resample_poly(rir, fs, sr, axis=0)

        rir = rir.T
        sig = sig.astype(np.float32)

        out = np.stack([
            fftconvolve(sig, rir_ch, mode='full')
            for rir_ch in rir
        ], axis=0)

        convolved.append(out.astype(np.float32))

    # Segment extraction
    segments = []

    for sig in convolved:
        mono = np.mean(sig, axis=0)

        start_idx = random_segment_with_activity(
            mono, segment_len_samples,
            detect_active_regions_relative(mono, fs)
        )

        seg = sig[:, start_idx:start_idx+segment_len_samples]

        if seg.shape[1] < segment_len_samples:
            pad = segment_len_samples - seg.shape[1]
            seg = np.pad(seg, ((0,0),(0,pad)))

        segments.append(seg.astype(np.float32))

    # Mix
    total_sum = np.sum(segments, axis=0).astype(np.float32)

    max_peak = np.max(np.abs(total_sum))
    if max_peak > 1.0:
        total_sum *= 0.99 / max_peak

    '''
    ambi = total_sum.T.astype(np.float32)
    output_wav_path = os.path.join("output_ambisonics.wav")
    wavfile.write(output_wav_path, fs, ambi)
    print(f"Saved ambisonics WAV: {output_wav_path}")
    '''

    scm_time = compute_scm(total_sum, segment_splits)

    # CSV rows
    rows = []
    for rir in selected_rirs.itertuples():
        rows.append([1.0, rir.dx, rir.dy, rir.dz, rir.Name])

    while len(rows) < 3:
        rows.append([0.0, 0.0, 0.0, 0.0, "0"])

    return scm_time, np.array(rows, dtype=object)


# =========================
# MAIN
# =========================
def main(csv_file, rir_folder, speech_folder, output_file,
         fs=24000, segment_len_s=0.2, segment_splits=10,
         min_voice=1, max_voice=3,
         num_repeats=3,
         num_fractions=1,
         freq_range=(100, 8000)):

    start = time.time()
    segment_len_samples = int(segment_len_s * fs)

    df = pd.read_csv(csv_file)  # <-- adjust this if you want to use more/less RIRs

    # Extract room & position
    df['r'] = df['Name'].str.extract(r'R(\d+)_')[0].astype(int)
    df['p'] = df['Name'].str.extract(r'_P(\d+)_')[0].astype(int)

    grouped = df.groupby(['r', 'p'])

    speech_files = [
        os.path.join(speech_folder, f)
        for f in os.listdir(speech_folder)
        if f.endswith('.wav')
    ]

    X_list, Y_list, csv_rows = [], [], []

    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))

    with ProcessPoolExecutor(max_workers=num_workers) as executor:

        args_list = []

        for (room, position), group in grouped:
            for repeat_idx in range(num_repeats):
                seed = hash((room, position, repeat_idx)) % (2**32)

                args_list.append((
                    room, position, group,
                    rir_folder, speech_files,
                    fs, segment_len_samples, segment_splits,
                    min_voice, max_voice,
                    num_fractions, freq_range,
                    seed
                ))

        for scm_time, rows in tqdm(
            executor.map(process_group, args_list, chunksize=5),
            total=len(args_list)
        ):
            X_list.append(scm_time)

            dirs = np.array([r[1:4] for r in rows], dtype=np.float32)
            Y_list.append(dirs)

            csv_rows.append(rows)

    # Save tensors
    X_tensor = torch.tensor(np.array(X_list, dtype=np.float32))
    Y_tensor = torch.tensor(np.array(Y_list, dtype=np.float32))

    torch.save({"X": X_tensor, "Y": Y_tensor}, output_file)

    # =========================
    # SAVE CSV
    # =========================
    csv_output_file = output_file.replace(".pt", ".csv")

    with open(csv_output_file, "w") as f:
        f.write("voice_active,dx,dy,dz,az,el,Name\n")

        for sample in csv_rows:
            for row in sample:
                vx, vy, vz = row[1:4]

                az = np.degrees(np.arctan2(vy, vx))
                el = np.degrees(np.arcsin(vz / (np.linalg.norm([vx, vy, vz]) + 1e-8)))

                f.write(f"{row[0]},{vx},{vy},{vz},{az:.1f},{el:.1f},{row[4]}\n")

    # Stats
    rusage_s = resource.getrusage(resource.RUSAGE_SELF)
    rusage_c = resource.getrusage(resource.RUSAGE_CHILDREN)

    print(f'Walltime {time.time()-start:.2f}s', file=sys.stderr)
    print(f'User time {rusage_s.ru_utime+rusage_c.ru_utime:.2f}s', file=sys.stderr)
    print(f'System time {rusage_s.ru_stime+rusage_c.ru_stime:.2f}s', file=sys.stderr)
    print(f'MaxRSS {(rusage_s.ru_maxrss+rusage_c.ru_maxrss)/2**20:.3f} GiB', file=sys.stderr)


# =========================
# RUN
# =========================
if __name__ == "__main__":
    csv_file = "../../../data/arni_em32/sh_wavs/arni_directions.csv"
    rir_folder = "../../../data/arni_em32/sh_wavs/RIRs"
    speech_folder = "../../../data/audio_exports_Harvard_Sets"
    output_file = "../../../data/arni_em32/sh_wavs/arni_dataset_scm_time.pt"

    main(
        csv_file,
        rir_folder,
        speech_folder,
        output_file,
        fs=24000,
        segment_len_s=1,
        segment_splits=10,
        min_voice=1,
        max_voice=3,
        num_repeats=3,
        num_fractions=1,
        freq_range=(100, 8000)
    )