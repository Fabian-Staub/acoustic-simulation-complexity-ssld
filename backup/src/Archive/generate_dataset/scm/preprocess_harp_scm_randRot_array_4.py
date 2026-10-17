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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from utils import detect_active_regions_relative, random_segment_with_activity, compute_scm, compute_scm_avg_log, convolve_rirs, apply_tikhonov_difference_filters


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
     max_noise, noise_volume_ratio, num_scms, freq_range, random_rotation, highpass,
     repeat_idx, seed) = args  # unpack repeat_idx and seed

    random.seed(seed)        # deterministic random for Python
    np.random.seed(seed)     # deterministic random for NumPy

    theta = 0.0
    if random_rotation:
        theta = random.uniform(0, 2*np.pi)
        #theta = random.choice([0, np.pi/2, np.pi, 3/2*np.pi])

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

    voice_conv = convolve_rirs(voice_rirs, voice_signals, rir_folder, fs)
    noise_conv = convolve_rirs(noise_rirs, noise_signals, rir_folder, fs) if num_noise>0 else []

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

    if highpass:
        total_sum = apply_tikhonov_difference_filters(signal=total_sum, fs=fs)

    # Safe normalization, relative Lautstärke bleibt erhalten
    max_peak = np.max(np.abs(total_sum))
    safety_factor = 0.9
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
    '''

    cov_all_bands = compute_scm_avg_log(total_sum, fs, freq_range, num_scms)
    #cov_all_bands = apply_order_weighting(cov_all_bands)
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
    num_tracks = 3
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
         highpass=False,
         num_scms=1,
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
                    max_noise, noise_volume_ratio, num_scms, freq_range, random_rotation, highpass,
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
            writer.writerow([])  # empty line between samples

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
# ==============================
# Run
# ==============================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate HOA SCM dataset from RIRs, speech, and noise."
    )

    # Paths / dataset config
    parser.add_argument(
        "--base-dir",
        type=str,
        default="../../../data/harp/R2000_P10_S5_Order4_24kHz_shoeBox_rtProfilesMIT",
        help="Base directory containing Generated_HOA_SRIR_data.csv and RIRs/"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="harp_dataset_2-8kHz_highpass_10-SCM_randRot_10-repeats",
        help="Name used for output .pt and .csv files"
    )
    parser.add_argument(
        "--speech-folder",
        type=str,
        default="../../../data/audio_exports_Harvard_Sets",
        help="Folder containing speech .wav files"
    )
    parser.add_argument(
        "--noise-folder",
        type=str,
        default="../../../data/noises",
        help="Folder containing noise .wav files"
    )

    # Processing params
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--fs", type=int, default=24000)
    parser.add_argument("--segment-len-s", type=float, default=1.0)

    parser.add_argument("--min-voice", type=int, default=1)
    parser.add_argument("--max-voice", type=int, default=3)

    parser.add_argument("--max-noise", type=int, default=0)
    parser.add_argument("--noise-volume-ratio", type=float, default=0.0)

    parser.add_argument("--num-repeats", type=int, default=10)
    parser.add_argument("--num-scms", type=int, default=10)

    parser.add_argument(
        "--freq-range",
        type=float,
        nargs=2,
        default=(2000, 8000),
        metavar=("F_LOW", "F_HIGH"),
        help="Frequency range, e.g. --freq-range 2000 8000"
    )

    # Boolean flags
    parser.add_argument(
        "--random-rotation",
        action="store_true",
        default=True,
        help="Enable random HOA rotation"
    )
    parser.add_argument(
        "--no-random-rotation",
        action="store_false",
        dest="random_rotation",
        help="Disable random HOA rotation"
    )

    parser.add_argument(
        "--highpass",
        action="store_true",
        default=True,
        help="Enable highpass / Tikhonov difference filters"
    )
    parser.add_argument(
        "--no-highpass",
        action="store_false",
        dest="highpass",
        help="Disable highpass / Tikhonov difference filters"
    )

    args = parser.parse_args()

    csv_file = os.path.join(args.base_dir, "Generated_HOA_SRIR_data.csv")
    output_file = os.path.join(args.base_dir, f"{args.dataset_name}.pt")
    rir_folder = os.path.join(args.base_dir, "RIRs")
    csv_output_file = os.path.join(args.base_dir, f"{args.dataset_name}.csv")

    main(
        csv_file,
        csv_output_file,
        rir_folder,
        args.speech_folder,
        args.noise_folder,
        output_file,
        order=args.order,
        fs=args.fs,
        segment_len_s=args.segment_len_s,
        min_voice=args.min_voice,
        max_voice=args.max_voice,
        max_noise=args.max_noise,
        noise_volume_ratio=args.noise_volume_ratio,
        num_repeats=args.num_repeats,
        random_rotation=args.random_rotation,
        highpass=args.highpass,
        num_scms=args.num_scms,
        freq_range=tuple(args.freq_range),
    )