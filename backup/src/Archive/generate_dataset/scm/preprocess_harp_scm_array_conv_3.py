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

# =========================
# Hilfsfunktionen
# =========================

def detect_active_regions_relative(signal, fs, frame_len=1024, hop_len=512, rel_threshold=0.1):
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
        return 0  # fallback

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

def compute_scm(signal_bands):
    n_bands, n_channels, n_samples = signal_bands.time.shape
    cov_all_bands = np.zeros((n_bands, n_channels, n_channels), dtype=np.float32)

    for b in range(n_bands):
        X = signal_bands.time[b]
        cov_all_bands[b] = X @ X.T

    # Normalize globally
    #min_val = cov_all_bands.min()
    #max_val = cov_all_bands.max()
    #cov_all_bands = (cov_all_bands - min_val) / (max_val - min_val + 1e-8)

    # Normalize locally
    for b in range(cov_all_bands.shape[0]):
        min_val = cov_all_bands[b].min()
        max_val = cov_all_bands[b].max()
        cov_all_bands[b] = (cov_all_bands[b] - min_val) / (max_val - min_val + 1e-8)

    return cov_all_bands

# =========================
# Verarbeitung einer einzelnen RIR
# =========================
def process_rir(args):
    rir_row, rir_folder, speech_files, fs, segment_len_samples, num_fractions, freq_range = args
    random.seed()  # für Parallelität

    # Zufällige Sprachdatei auswählen
    sig = librosa.load(random.choice(speech_files), sr=fs)[0]
    sig = 0.9 * sig / (np.max(np.abs(sig)) + 1e-8)

    # RIR laden
    rir_path = os.path.join(rir_folder, rir_row.Name)
    sr, rir = wavfile.read(rir_path)
    if rir.ndim == 1: rir = rir[:, None]
    rir = rir.astype(np.float32) / (np.max(np.abs(rir)) + 1e-8)

    # Faltung
    conv_sig = np.stack([fftconvolve(sig, rir[:,ch], mode='full') for ch in range(rir.shape[1])], axis=0)

    # Segment extrahieren
    mono = np.mean(conv_sig, axis=0)
    start_idx = random_segment_with_activity(mono, segment_len_samples, detect_active_regions_relative(mono, fs))
    seg_multi = conv_sig[:, start_idx:start_idx+segment_len_samples]
    if seg_multi.shape[1] < segment_len_samples:
        pad = segment_len_samples - seg_multi.shape[1]
        seg_multi = np.pad(seg_multi, ((0,0),(0,pad)))

    # SCM berechnen
    signal_obj = pf.Signal(seg_multi, fs)
    signal_bands = pf.dsp.filter.fractional_octave_bands(signal_obj,
        num_fractions=num_fractions,
        frequency_range=freq_range,
        order=15
    )
    cov_all_bands = compute_scm(signal_bands)

    # Ground Truth Vektor
    dx, dy, dz = rir_row['x_source']-rir_row['x_mic'], rir_row['y_source']-rir_row['y_mic'], rir_row['z_source']-rir_row['z_mic']
    r_dist = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-8
    gt_vec = np.array([dx/r_dist, dy/r_dist, dz/r_dist], dtype=np.float32)

    return cov_all_bands, gt_vec

# =========================
# Main Funktion
# =========================
def main(csv_file, csv_output_file, rir_folder, speech_folder, output_file,
         fs=24000, segment_len_s=1.0, num_fractions=1, freq_range=(100,8000)):
    start = time.time()
    hop_len = int(0.02*fs)
    segment_len_samples = int(segment_len_s * fs + hop_len)

    df = pd.read_csv(csv_file)  # nrows
    speech_files = [os.path.join(speech_folder, f) for f in os.listdir(speech_folder) if f.endswith('.wav')]

    # Argumentliste für alle RIRs
    args_list = [
        (row, rir_folder, speech_files, fs, segment_len_samples, num_fractions, freq_range)
        for _, row in df.iterrows()
    ]

    # Parallel verarbeiten
    X_list, Y_list = [], []
    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for cov_all_bands, gt_vec in tqdm(executor.map(process_rir, args_list), total=len(args_list)):
            X_list.append(cov_all_bands)
            Y_list.append(gt_vec)

    # Tensoren speichern
    X_tensor = torch.tensor(np.array(X_list, dtype=np.float32))
    Y_tensor = torch.tensor(np.array(Y_list, dtype=np.float32))
    torch.save({"X": X_tensor, "Y": Y_tensor}, output_file)

    # CSV speichern
    meta_cols = ['Name','RT','c50','x_room','y_room','z_room',
                 'x_source','y_source','z_source','x_mic','y_mic','z_mic','status']
    with open(csv_output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ['voice_active','dx','dy','dz'] + meta_cols
        writer.writerow(header)
        for idx, row in enumerate(df.itertuples(index=False)):
            writer.writerow([1.0, *Y_list[idx], *row])

    # Performance stats
    rusage_s = resource.getrusage(resource.RUSAGE_SELF)
    rusage_c = resource.getrusage(resource.RUSAGE_CHILDREN)
    print(f'Walltime {time.time()-start:.2f}s', file=sys.stderr)
    print(f'User time {rusage_s.ru_utime+rusage_c.ru_utime:.2f}s', file=sys.stderr)
    print(f'System time {rusage_s.ru_stime+rusage_c.ru_stime:.2f}s', file=sys.stderr)
    print(f'MaxRSS {(rusage_s.ru_maxrss+rusage_c.ru_maxrss)/2**20:.3f} GiB', file=sys.stderr)

# ==============================
# Run
# ==============================
if __name__ == "__main__":
    csv_file = "../../../data/harp/R800_P10_S5_Order4_24kHz_shoeBox_ISM/Generated_HOA_SRIR_data.csv"
    csv_output_file = "../../../data/harp/R800_P10_S5_Order4_24kHz_shoeBox_ISM/harp_dataset_scm.csv"
    rir_folder = "../../../data/harp/R800_P10_S5_Order4_24kHz_shoeBox_ISM"
    speech_folder = "../../../data/audio_exports_Harvard_Sets"
    output_file = "../../../data/harp/R800_P10_S5_Order4_24kHz_shoeBox_ISM/harp_dataset_scm.pt"

    main(csv_file, csv_output_file, rir_folder, speech_folder, output_file,
         fs=24000, segment_len_s=1.0, num_fractions=1, freq_range=(100,8000))