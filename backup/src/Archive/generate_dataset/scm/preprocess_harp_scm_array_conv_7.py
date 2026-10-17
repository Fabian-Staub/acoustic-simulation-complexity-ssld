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
# Process a single group (parallelized)
# =========================
def process_group(args):
    (room, receiver, group, rir_folder, speech_files, noise_files,
     fs, segment_len_samples, min_voice, max_voice,
     max_noise, noise_volume_ratio, num_fractions, freq_range,
     repeat_idx, seed) = args  # unpack repeat_idx and seed

    random.seed(seed)        # deterministic random for Python
    np.random.seed(seed)     # deterministic random for NumPy

    # Sample RIRs for voice and noise
    num_voice = random.randint(min_voice, max_voice)
    voice_rirs = group.sample(num_voice)
    remaining_rirs = group.drop(voice_rirs.index)
    num_noise = random.randint(0, min(max_noise, len(remaining_rirs)))
    noise_rirs = remaining_rirs.sample(num_noise) if num_noise > 0 else pd.DataFrame(columns=group.columns)

    # Load and normalize signals
    voice_signals = [librosa.load(random.choice(speech_files), sr=fs)[0] for _ in range(num_voice)]
    voice_signals = [0.9 * sig / (np.max(np.abs(sig)) + 1e-8) for sig in voice_signals]

    noise_signals = [librosa.load(random.choice(noise_files), sr=fs)[0] for _ in range(num_noise)]
    noise_signals = [(0.9*noise_volume_ratio) * sig / (np.max(np.abs(sig))+1e-8) for sig in noise_signals]

    # --- Convolve RIRs robust und in float32 ---
    def convolve_rirs(rirs, sigs):
        convolved = []
        for rir_row, sig in zip(rirs.itertuples(), sigs):
            rir_path = os.path.join(rir_folder, rir_row.Name)
            sr, rir = wavfile.read(rir_path)

            # --- Robust conversion ---
            if rir.dtype != np.float32:
                # int16 / int32 → float32 im Bereich [-1,1]
                if np.issubdtype(rir.dtype, np.integer):
                    max_val = np.iinfo(rir.dtype).max
                    rir = rir.astype(np.float32) / max_val
                else:
                    rir = rir.astype(np.float32)

            # Sicherstellen: 2D (channels, samples)
            if rir.ndim == 1:
                rir = rir[:, None]
            rir = rir.T  # shape: (channels, samples)

            sig = sig.astype(np.float32)  # Quelle ebenfalls float32

            # FFT-Convolve für jeden Kanal
            out = np.stack([fftconvolve(sig, rir_ch, mode='full') for rir_ch in rir], axis=0)
            convolved.append(out.astype(np.float32))
        return convolved

    voice_conv = convolve_rirs(voice_rirs, voice_signals)
    noise_conv = convolve_rirs(noise_rirs, noise_signals) if num_noise>0 else []

    # Segment Extraktion
    segments = []
    for sig_list in [voice_conv, noise_conv]:
        for sig in sig_list:
            sig = sig.astype(np.float32)
            mono = np.mean(sig, axis=0)
            start_idx = random_segment_with_activity(
                mono, segment_len_samples,
                detect_active_regions_relative(mono, fs)
            )
            seg_multi = sig[:, start_idx:start_idx+segment_len_samples]
            if seg_multi.shape[1] < segment_len_samples:
                pad = segment_len_samples - seg_multi.shape[1]
                seg_multi = np.pad(seg_multi, ((0,0),(0,pad)), mode='constant', constant_values=0)
            segments.append(seg_multi.astype(np.float32))

    # Summe aller Segmente
    total_sum = np.sum(segments, axis=0).astype(np.float32)

    # Safe normalization, relative Lautstärke bleibt erhalten
    max_peak = np.max(np.abs(total_sum))
    safety_factor = 0.99
    if max_peak > 1.0:
        scale = safety_factor / max_peak
        total_sum *= scale

    
    ambi = total_sum.T.astype(np.float32)
    output_wav_path = os.path.join("output_ambisonics.wav")
    wavfile.write(output_wav_path, fs, ambi)
    print(f"Saved ambisonics WAV: {output_wav_path}")
    

    # =========================
    # SCM computation
    # =========================
    signal_obj = pf.Signal(total_sum, fs)
    signal_bands = pf.dsp.filter.fractional_octave_bands(
        signal_obj,
        num_fractions=num_fractions,
        frequency_range=freq_range,
        order=15
    )
    cov_all_bands = compute_scm(signal_bands)

    # Build Y metadata
    meta_cols = ['Name','RT','c50','x_room','y_room','z_room',
                 'x_source','y_source','z_source','x_mic','y_mic','z_mic','status']
    all_rirs = pd.concat([voice_rirs, noise_rirs], ignore_index=True) if num_noise>0 else voice_rirs
    Y_sample = []

    # --- Voices only for tensor ---
    voice_dirs = []
    for rir in voice_rirs.itertuples():
        dx = rir.x_mic - rir.x_source
        dy = rir.y_source - rir.y_mic
        dz = rir.z_source - rir.z_mic
        r_dist = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-8
        voice_dirs.append([dx/r_dist, dy/r_dist, dz/r_dist])

    # Pad to max_voice
    while len(voice_dirs) < 3:
        voice_dirs.append([0.0, 0.0, 0.0])

    #voice_dirs = voice_dirs[:max_voice]  # safety

    # --- CSV still uses full metadata (UNCHANGED behavior) ---
    num_tracks = 5
    all_rirs = pd.concat([voice_rirs, noise_rirs], ignore_index=True) if num_noise>0 else voice_rirs

    for idx in range(num_tracks):
        row_vec = np.zeros(1+3+len(meta_cols), dtype=object)
        if idx < len(all_rirs):
            rir = all_rirs.iloc[idx]
            row_vec[0] = 1.0 if idx < len(voice_rirs) else 0.0
            dx, dy, dz = rir['x_mic']-rir['x_source'], rir['y_source']-rir['y_mic'], rir['z_source']-rir['z_mic']
            r_dist = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-8
            row_vec[1:4] = [dx/r_dist, dy/r_dist, dz/r_dist]
            row_vec[4:] = rir[meta_cols].values
        else:
            row_vec[0] = 0.0
            row_vec[1:4] = [0.0,0.0,0.0]
        Y_sample.append(row_vec)

    return cov_all_bands, np.array(Y_sample, dtype=object), np.array(voice_dirs, dtype=np.float32)

# =========================
# Main function
# =========================
def main(csv_file, csv_output_file, wav_folder, speech_folder, noise_folder, output_file,
         order=4, fs=24000, segment_len_s=1.0,
         min_voice=1, max_voice=3,
         max_noise=2,
         noise_volume_ratio=1.0,
         num_repeats=1,
         num_fractions=1,
         freq_range=(100, 8000)):
    start = time.time()
    segment_len_samples = int(segment_len_s * fs)

    df = pd.read_csv(csv_file, nrows=40)  # <-- adjust this if you want to use more/less RIRs
    df['r'] = df['Name'].str.extract(r'R(\d+)_')[0].astype(int)
    df['p'] = df['Name'].str.extract(r'_P(\d+)_')[0].astype(int)
    df['s'] = df['Name'].str.extract(r'_S(\d+)_')[0].astype(int)

    grouped = df.groupby(['r','p'])
    speech_files = [os.path.join(speech_folder, f) for f in os.listdir(speech_folder) if f.endswith('.wav')]
    noise_files  = [os.path.join(noise_folder, f) for f in os.listdir(noise_folder) if f.endswith('.wav')]

    # =========================
    # Parallel processing with repeats
    # =========================
    X_list, Y_list, voice_dir_list = [], [], []

    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        args_list = []

        for (room, receiver), group in grouped:
            for repeat_idx in range(num_repeats):
                seed = hash((room, receiver, repeat_idx)) % (2**32)  # deterministic seed
                args_list.append(
                    (room, receiver, group, wav_folder, speech_files, noise_files,
                    fs, segment_len_samples, min_voice, max_voice,
                    max_noise, noise_volume_ratio, num_fractions, freq_range,
                    repeat_idx, seed)  # add repeat_idx and seed
                )

        for cov_all_bands, Y_sample, voice_dirs in tqdm(
            executor.map(process_group, args_list, chunksize=5),
            total=len(args_list)
        ):
            X_list.append(cov_all_bands)
            Y_list.append(Y_sample)
            voice_dir_list.append(voice_dirs)

    # Convert to tensors
    X_tensor = torch.tensor(np.array(X_list, dtype=np.float32))
    Y_tensor = torch.tensor(np.array(voice_dir_list, dtype=np.float32))
    torch.save({"X": X_tensor, "Y": Y_tensor}, output_file)

    # Save CSV
    meta_cols = ['Name','RT','c50','x_room','y_room','z_room',
                 'x_source','y_source','z_source','x_mic','y_mic','z_mic','status']
    with open(csv_output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ['voice_active','dx','dy','dz'] + meta_cols
        writer.writerow(header)
        for sample in Y_list:
            for row in sample:
                writer.writerow([str(item) for item in row])

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
    csv_file = "../../../data/harp/test/Generated_HOA_SRIR_data.csv"
    csv_output_file = "../../../data/harp/test/harp_dataset_scm_rot.csv"
    rir_folder = "../../../data/harp/test/RIRs"
    speech_folder = "../../../data/audio_exports_Harvard_Sets"
    noise_folder = "../../../data/noises"
    output_file = "../../../data/harp/test/harp_dataset_scm_rot.pt"

    main(
        csv_file, csv_output_file, rir_folder, speech_folder, noise_folder, output_file,
        order=4, fs=24000, segment_len_s=1,
        min_voice=1, max_voice=1,
        max_noise=0,
        noise_volume_ratio=0,
        num_repeats=1,
        num_fractions=1,
        freq_range=(100, 8000)
    )