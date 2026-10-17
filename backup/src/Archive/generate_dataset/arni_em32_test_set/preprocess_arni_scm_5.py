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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from utils import detect_active_regions_relative, random_segment_with_activity, compute_scm, compute_scm_avg_log, convolve_rirs, apply_order_weighting


# =========================
# Process one group
# =========================
def process_group(args):
    (room, position, group, rir_folder, speech_files,
     fs, segment_len_samples, min_voice, max_voice,
     num_scms, freq_range, seed, repeat_idx) = args

    random.seed(seed)
    np.random.seed(seed)

    # Select RIRs
    num_voice = random.randint(min_voice, max_voice)
    selected_rirs = group.sample(n=min(num_voice, len(group)))

    # Load speech
    speech_signals = [librosa.load(random.choice(speech_files), sr=fs)[0] for _ in range(len(selected_rirs))]
    speech_signals = [0.9 * sig / (np.max(np.abs(sig)) + 1e-8) for sig in speech_signals]

    convolved = convolve_rirs(selected_rirs, speech_signals, rir_folder, fs)

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
    total_sum *= 0.9 / max_peak

    
    ambi = total_sum.T.astype(np.float32)
    output_wav_path = os.path.join(
        "output_ambisonics",
        f"ambisonics_{room}_P{position}_rep{repeat_idx}.wav"
    )
    os.makedirs(os.path.dirname(output_wav_path), exist_ok=True)
    wavfile.write(output_wav_path, fs, ambi)

    '''
    # SCM
    signal_obj = pf.Signal(total_sum, fs)

    signal_bands = pf.dsp.filter.fractional_octave_bands(
        signal_obj,
        num_fractions=num_fractions,
        frequency_range=freq_range,
        order=15
    )

    cov_all_bands = compute_scm(signal_bands)
    

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


    # CSV rows
    rows = []
    for rir in selected_rirs.itertuples():
        rows.append([1.0, rir.dx, rir.dy, rir.dz, rir.rt60, rir.Name])

    while len(rows) < 3:
        rows.append([0.0, 0.0, 0.0, 0.0, 0.0, "0"])

    return cov_all_bands, np.array(rows, dtype=object)


# =========================
# MAIN
# =========================
def main(csv_file, rir_folder, speech_folder, output_file,
         fs=24000, segment_len_s=1,
         min_voice=1, max_voice=3,
         num_repeats=1,
         num_scms=10,
         freq_range=(2000, 7000),
         global_seed=42):

    start = time.time()
    segment_len_samples = int(segment_len_s * fs)

    df = pd.read_csv(csv_file)  # <-- adjust this if you want to use more/less RIRs

    # Extract room name and position
    df['room'] = df['Name'].str.extract(r'^(.*?)_P')[0]
    df['position'] = df['Name'].str.extract(r'_P(\d+)_')[0].astype(int)

    grouped = df.groupby(['room', 'position'])

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
                seed = (hash((room, position, repeat_idx)) + global_seed) % (2**32)

                args_list.append((
                    room, position, group,
                    rir_folder, speech_files,
                    fs, segment_len_samples,
                    min_voice, max_voice,
                    num_scms, freq_range,
                    seed, repeat_idx
                ))

        for cov_all_bands, rows in tqdm(
            executor.map(process_group, args_list, chunksize=5),
            total=len(args_list)
        ):
            X_list.append(cov_all_bands)

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
        f.write("voice_active,dx,dy,dz,az,el,rt60,Name\n")

        for sample in csv_rows:
            for row in sample:
                vx, vy, vz = row[1:4]
                rt60 = float(row[4])

                az = np.degrees(np.arctan2(vy, vx))
                el = np.degrees(np.arcsin(vz / (np.linalg.norm([vx, vy, vz]) + 1e-8)))

                f.write(f"{row[0]},{vx},{vy},{vz},{az:.1f},{el:.1f},{rt60},{row[5]}\n")
            f.write("\n") # blank line between samples

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
    csv_file = "../../../data/arni_em32/sh_wavs/arni_directions_rt60.csv"
    rir_folder = "../../../data/arni_em32/sh_wavs/RIRs_truncated"
    speech_folder = "../../../data/audio_exports_Harvard_Sets"
    output_file = "../../../data/arni_em32/sh_wavs/arni_dataset_2-7kHz_10-SCM_multiSource.pt"

    main(
        csv_file,
        rir_folder,
        speech_folder,
        output_file,
        fs=24000,
        segment_len_s=1,
        min_voice=1,
        max_voice=3,
        num_repeats=1,
        num_scms=10,
        freq_range=(2000, 7000)
    )