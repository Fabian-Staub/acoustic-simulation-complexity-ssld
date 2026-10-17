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
from spharpy.transforms import SphericalHarmonicRotation, rotation_z_axis_real
from spharpy import SphericalHarmonicSignal, SphericalHarmonicDefinition
from scipy.signal import resample_poly

def detect_active_regions_relative(signal, fs, frame_len=512, hop_len=256, rel_threshold=0.01):
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

def random_segment_with_activity(sig, segment_len_samples, active_regions, min_active_ratio=0.8, max_trials=1000):
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
# HOA rotation
# =========================
'''
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

    R = rotation_z_axis_real(n_max, theta)

    rotated = R * sh
    return np.squeeze(rotated._data, axis=0).astype(np.float32)
'''

def rotate_hoa(signal, theta):
    n_max = int(np.sqrt(signal.shape[0]) - 1)

    R = rotation_z_axis_real(n_max, theta)
    return (R @ signal).astype(np.float32)

def rotate_vectors_z(vectors, theta):
    c = np.cos(theta)
    s = np.sin(theta)

    R = np.array([
        [ c, -s, 0],
        [ s,  c, 0],
        [ 0,  0, 1]
    ], dtype=np.float32)

    return (R @ vectors.T).T


def apply_order_weighting(cov_all_bands):
    n_bands, n_channels, _ = cov_all_bands.shape

    order_weights = {
        0: 1.0,
        1: 1.0,
        2: 0.75,
        3: 0.5,
        4: 0.25,
    }

    # Build channel-wise weight vector
    weights = np.ones(n_channels, dtype=np.float32)

    n_max = int(np.sqrt(n_channels) - 1)

    for n in range(n_max + 1):
        start = n**2
        end = (n+1)**2
        w = order_weights.get(n, 1.0)
        weights[start:end] = w

    # ---- process each band ----
    for b in range(n_bands):
        C = cov_all_bands[b].astype(np.float32)

        # 1) normalize to [-1, 1]
        c_min, c_max = C.min(), C.max()
        if c_max > c_min:
            C = 2.0 * (C - c_min) / (c_max - c_min) - 1.0
        else:
            C = np.zeros_like(C)

        # 2) apply weighting (W C W)
        C = weights[:, None] * C * weights[None, :]

        # 3) normalize to [0, 1]
        c_min, c_max = C.min(), C.max()
        if c_max > c_min:
            C = (C - c_min) / (c_max - c_min)
        else:
            C = np.zeros_like(C)

        cov_all_bands[b] = C

    return cov_all_bands


def get_task_range(n_items: int):
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    task_count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

    items_per_task = n_items // task_count
    remainder = n_items % task_count

    start = task_id * items_per_task + min(task_id, remainder)
    end = start + items_per_task + (1 if task_id < remainder else 0)

    return start, end


# =========================
# Process a single group (parallelized)
# =========================
def process_group(args):
    (room, receiver, group, rir_folder, speech_files, noise_files,
     fs, segment_len_samples, min_voice, max_voice,
     max_noise, noise_volume_ratio, num_fractions, freq_range, random_rotation,
     repeat_idx, seed) = args  # unpack repeat_idx and seed

    random.seed(seed)        # deterministic random for Python
    np.random.seed(seed)     # deterministic random for NumPy

    theta = 0.0
    if random_rotation:
        #theta = random.uniform(0, 2*np.pi)
        theta = random.choice([0, np.pi/2, np.pi, 3/2*np.pi])

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

            if sr != fs:
                rir = resample_poly(rir, fs, sr, axis=0)

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

    if random_rotation:
        total_sum = rotate_hoa(total_sum, theta)

    # Safe normalization, relative Lautstärke bleibt erhalten
    max_peak = np.max(np.abs(total_sum))
    safety_factor = 0.9
    if max_peak > 0.9:
        scale = safety_factor / max_peak
        total_sum *= scale

    '''
    ambi = total_sum.T.astype(np.float32)
    output_wav_path = os.path.join(
        "output_ambisonics",
        f"ambisonics_r{room}_p{receiver}_rep{repeat_idx}.wav"
    )
    os.makedirs(os.path.dirname(output_wav_path), exist_ok=True)
    wavfile.write(output_wav_path, fs, ambi)
    '''
    '''
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
    '''

    ########################################
    signal_obj = pf.Signal(total_sum, fs)

    # Bandpass filter
    signal_filtered = pf.dsp.filter.butterworth(
        signal_obj,
        N=4,  # filter order (you can tune this)
        frequency=[2000, 6000],
        btype='bandpass'
    )

    # Add fake band dimension to match compute_scm input
    signal_filtered.time = signal_filtered.time[None, ...]

    cov_all_bands = compute_scm(signal_filtered)
    cov_all_bands = apply_order_weighting(cov_all_bands)
    ########################################
    

    # Build Y metadata
    meta_cols = ['Name','RT','c50','x_room','y_room','z_room',
                 'x_source','y_source','z_source','x_mic','y_mic','z_mic','status']
    all_rirs = pd.concat([voice_rirs, noise_rirs], ignore_index=True) if num_noise>0 else voice_rirs
    Y_sample = []

    # --- Voices only for tensor ---
    voice_dirs = []

    for rir in voice_rirs.itertuples():
        dx = rir.x_source - rir.x_mic
        dy = rir.y_source - rir.y_mic
        dz = rir.z_source - rir.z_mic
        r_dist = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-8
        voice_dirs.append([dx/r_dist, dy/r_dist, dz/r_dist])

    voice_dirs = np.array(voice_dirs, dtype=np.float32)

    if random_rotation:
        voice_dirs = rotate_vectors_z(voice_dirs, theta)

    # pad safely (NumPy way)
    if len(voice_dirs) < 3:
        pad = np.zeros((3 - len(voice_dirs), 3), dtype=np.float32)
        voice_dirs = np.vstack([voice_dirs, pad])

    #voice_dirs = voice_dirs[:max_voice]  # safety

    # --- CSV still uses full metadata (UNCHANGED behavior) ---
    num_tracks = 5
    all_rirs = pd.concat([voice_rirs, noise_rirs], ignore_index=True) if num_noise>0 else voice_rirs

    for idx in range(num_tracks):
        row_vec = np.zeros(1+3+len(meta_cols), dtype=object)
        if idx < len(all_rirs):
            rir = all_rirs.iloc[idx]
            row_vec[0] = 1.0 if idx < len(voice_rirs) else 0.0

            dx = rir['x_source'] - rir['x_mic']
            dy = rir['y_source'] - rir['y_mic']
            dz = rir['z_source'] - rir['z_mic']
            vec = np.array([dx, dy, dz], dtype=np.float32)
            vec = vec / (np.linalg.norm(vec) + 1e-8)
            if random_rotation:
                vec = rotate_vectors_z(vec.reshape(1, 3), theta)[0]
            row_vec[1:4] = vec

            row_vec[4:] = rir[meta_cols].values
        else:
            row_vec[0] = 0.0
            row_vec[1:4] = [0.0,0.0,0.0]
        Y_sample.append(row_vec)

    return cov_all_bands, np.array(Y_sample, dtype=object), np.array(voice_dirs, dtype=np.float32), theta

# =========================
# Main function
# =========================
def main(csv_file, csv_output_file, wav_folder, speech_folder, noise_folder, output_file,
         order=4, fs=24000, segment_len_s=1.0,
         min_voice=1, max_voice=3,
         max_noise=2,
         noise_volume_ratio=1.0,
         num_repeats=1,
         random_rotation=False,
         num_fractions=1,
         freq_range=(100, 8000)):
    start = time.time()

    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    output_file = output_file.replace(".pt", f"_task_{task_id}.pt")
    csv_output_file = csv_output_file.replace(".csv", f"_task_{task_id}.csv")

    segment_len_samples = int(segment_len_s * fs)

    df = pd.read_csv(csv_file)  # <-- adjust this if you want to use more/less RIRs
    df['r'] = df['Name'].str.extract(r'R(\d+)_')[0].astype(int)
    df['p'] = df['Name'].str.extract(r'_P(\d+)_')[0].astype(int)
    df['s'] = df['Name'].str.extract(r'_S(\d+)_')[0].astype(int)

    grouped = list(df.groupby(['r','p']))
    start_idx, end_idx = get_task_range(len(grouped))
    grouped_subset = grouped[start_idx:end_idx]

    speech_files = [os.path.join(speech_folder, f) for f in os.listdir(speech_folder) if f.endswith('.wav')]
    noise_files  = [os.path.join(noise_folder, f) for f in os.listdir(noise_folder) if f.endswith('.wav')]

    # =========================
    # Parallel processing with repeats
    # =========================
    X_list, Y_list, voice_dir_list, rotation_list = [], [], [], []

    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        args_list = []

        for (room, receiver), group in grouped_subset:
            for repeat_idx in range(num_repeats):
                seed = hash((room, receiver, repeat_idx)) % (2**32)  # deterministic seed
                args_list.append(
                    (room, receiver, group, wav_folder, speech_files, noise_files,
                    fs, segment_len_samples, min_voice, max_voice,
                    max_noise, noise_volume_ratio, num_fractions, freq_range, random_rotation,
                    repeat_idx, seed)  # add repeat_idx and seed
                )

        for cov_all_bands, Y_sample, voice_dirs, theta in tqdm(
            executor.map(process_group, args_list, chunksize=5),
            total=len(args_list)
        ):
            X_list.append(cov_all_bands)
            Y_list.append(Y_sample)
            voice_dir_list.append(voice_dirs)
            rotation_list.append(theta)

    # Convert to tensors
    X_tensor = torch.tensor(np.array(X_list, dtype=np.float32))
    Y_tensor = torch.tensor(np.array(voice_dir_list, dtype=np.float32))
    torch.save({"X": X_tensor, "Y": Y_tensor}, output_file)

    # Save CSV
    meta_cols = ['Name','RT','c50','x_room','y_room','z_room',
                 'x_source','y_source','z_source','x_mic','y_mic','z_mic','status']
    with open(csv_output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ['voice_active','dx','dy','dz','rotation', 'az','el'] + meta_cols
        writer.writerow(header)
        for theta, sample in zip(rotation_list, Y_list):
            for row in sample:
                dx, dy, dz = float(row[1]), float(row[2]), float(row[3])
                # azimuth (xy-plane)
                az = np.degrees(np.arctan2(dy, dx))
                # elevation
                r = np.sqrt(dx*dx + dy*dy + dz*dz) + 1e-8
                el = np.degrees(np.arcsin(dz / r))
                writer.writerow(
                    [
                        str(row[0]),
                        str(dx), str(dy), str(dz),
                        str(round(np.degrees(theta), 1)),
                        str(round(az, 1)),
                        str(round(el, 1))
                    ]
                    + [str(item) for item in row[4:]]
                )

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
    csv_file = "../../../data/harp/R2000_P10_S5_Order4_24kHz_shoeBox_anisotropic_ISM/Generated_HOA_SRIR_data.csv"
    csv_output_file = "../../../data/harp/R2000_P10_S5_Order4_24kHz_shoeBox_anisotropic_ISM/harp_dataset_2-6kHz_multiSource.csv"
    rir_folder = "../../../data/harp/R2000_P10_S5_Order4_24kHz_shoeBox_anisotropic_ISM/RIRs"
    speech_folder = "../../../data/audio_exports_Harvard_Sets"
    noise_folder = "../../../data/noises"
    output_file = "../../../data/harp/R2000_P10_S5_Order4_24kHz_shoeBox_anisotropic_ISM/harp_dataset_2-6kHz_multiSource.pt"

    main(
        csv_file, csv_output_file, rir_folder, speech_folder, noise_folder, output_file,
        order=4, fs=24000, segment_len_s=1,
        min_voice=1, max_voice=3,
        max_noise=0,
        noise_volume_ratio=0,
        num_repeats=3,
        random_rotation=False,
        #num_fractions=1,
        #freq_range=(100, 8000)
    )