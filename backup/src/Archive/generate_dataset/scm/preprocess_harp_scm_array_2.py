import os
import numpy as np
import pandas as pd
import pyfar as pf
import torch
from scipy.io import wavfile
import random
import time
import resource
import sys
from tqdm import tqdm


def main(csv_file, wav_folder, output_file,
         order=4, fs=24000,
         min_voice=1, max_voice=1,
         vol_aug_voice=0,
         max_noise=2,
         vol_aug_noise=0.0,
         noise_volume_ratio=1.0,
         num_fractions=1,
         freq_range=(100, 8000)):
    """
    Generate HOA dataset with:
    - Mixed speech + noise
    - Spatial Covariance Matrix (SCM)
    - ACCDOA targets (voice only)
    - Active mask
    - Metadata
    """

    NUM_SH = (order + 1) ** 2

    # ==============================
    # Helper functions
    # ==============================
    def azel_to_accdoa(azi_list, el_list, max_voice):
        accdoa = np.zeros((max_voice, 3), dtype=np.float32)
        n = min(len(azi_list), max_voice)

        for i in range(n):
            x = np.cos(el_list[i]) * np.cos(azi_list[i])
            y = np.cos(el_list[i]) * np.sin(azi_list[i])
            z = np.sin(el_list[i])

            vec = np.array([x, y, z], dtype=np.float32)
            vec /= np.linalg.norm(vec) + 1e-8
            accdoa[i] = vec

        return accdoa

    def compute_scm(signal_bands):
        n_bands, n_channels, n_samples = signal_bands.time.shape
        cov_all_bands = np.zeros((n_bands, n_channels, n_channels), dtype=np.float32)

        for b in range(n_bands):
            X = signal_bands.time[b]
            cov_all_bands[b] = X @ X.T

        # Normalize globally
        min_val = cov_all_bands.min()
        max_val = cov_all_bands.max()
        cov_all_bands = (cov_all_bands - min_val) / (max_val - min_val + 1e-8)

        return cov_all_bands

    # ==============================
    # Load metadata
    # ==============================
    df = pd.read_csv(csv_file, nrows=10)  # remove nrows for full dataset

    df['r'] = df['Name'].str.extract(r'R(\d+)_')[0].astype(int)
    df['p'] = df['Name'].str.extract(r'_P(\d+)_')[0].astype(int)
    df['s'] = df['Name'].str.extract(r'_S(\d+)_')[0].astype(int)

    grouped = df.groupby(['r', 'p'])

    X_list, Y_list = [], []

    # ==============================
    # Main loop
    # ==============================
    for (room, receiver), group in tqdm(grouped, total=len(grouped), desc="Generating samples"):

        # --------------------------
        # Select voice sources
        # --------------------------
        num_sources = random.randint(min_voice, max_voice)
        voice_sources = group.sample(num_sources)

        remaining = group.drop(voice_sources.index)

        # --------------------------
        # Select noise sources
        # --------------------------
        num_noise = random.randint(0, min(max_noise, len(remaining)))
        if num_noise > 0:
            noise_sources = remaining.sample(num_noise)
        else:
            noise_sources = pd.DataFrame(columns=group.columns)

        mixed_wav = None
        azi_list, el_list = [], []

        # =========================
        # 🔊 VOICE SOURCES
        # =========================
        for _, row in voice_sources.iterrows():
            wav_path = os.path.join(wav_folder, row['Name'])
            fs_wav, wav = wavfile.read(wav_path)

            wav = wav[:, :NUM_SH].T.astype(np.float32)

            # Volume augmentation (speech)
            vol = random.uniform(max(0.0, 1.0 - vol_aug_voice), 1.0)
            wav *= vol

            if mixed_wav is None:
                mixed_wav = wav
            else:
                min_len = min(mixed_wav.shape[1], wav.shape[1])
                mixed_wav = mixed_wav[:, :min_len] + wav[:, :min_len]

            # Direction (only for voice)
            dx = row['x_source'] - row['x_mic']
            dy = row['y_source'] - row['y_mic']
            dz = row['z_source'] - row['z_mic']
            r_dist = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-8

            azi_list.append(np.arctan2(dy, dx))
            el_list.append(np.arcsin(dz / r_dist))

        # =========================
        # 🔉 NOISE SOURCES
        # =========================
        for _, row in noise_sources.iterrows():
            wav_path = os.path.join(wav_folder, row['Name'])
            fs_wav, wav = wavfile.read(wav_path)

            wav = wav[:, :NUM_SH].T.astype(np.float32)

            # Noise volume augmentation
            vol = random.uniform(max(0.0, 1.0 - vol_aug_noise), 1.0)

            # Apply global noise scaling
            wav *= vol * noise_volume_ratio

            if mixed_wav is None:
                mixed_wav = wav
            else:
                min_len = min(mixed_wav.shape[1], wav.shape[1])
                mixed_wav = mixed_wav[:, :min_len] + wav[:, :min_len]

        # --------------------------
        # Normalize to avoid clipping
        # --------------------------
        peak = np.max(np.abs(mixed_wav)) + 1e-8
        if peak > 1.0:
            mixed_wav = mixed_wav / peak

        # =========================
        # SCM computation
        # =========================
        signal_obj = pf.Signal(mixed_wav, fs)

        signal_bands = pf.dsp.filter.fractional_octave_bands(
            signal_obj,
            num_fractions=num_fractions,
            frequency_range=freq_range,
            order=15
        )

        cov_all_bands = compute_scm(signal_bands)
        X_list.append(cov_all_bands)

        # =========================
        # Build Y
        # =========================
        Y_sample = []
        meta_cols = ['Name','RT','c50','x_room','y_room','z_room',
                    'x_source','y_source','z_source','x_mic','y_mic','z_mic','status']

        if not noise_sources.empty:
            all_sources = pd.concat([voice_sources, noise_sources], ignore_index=True)
        else:
            all_sources = voice_sources.copy()

        num_tracks = 5  # always 5 tracks per sample

        for idx in range(num_tracks):
            row_vec = np.zeros(1 + 3 + len(meta_cols), dtype=object)

            if idx < len(voice_sources):
                # voice
                row_vec[0] = 1.0
                dx = all_sources.iloc[idx]['x_source'] - all_sources.iloc[idx]['x_mic']
                dy = all_sources.iloc[idx]['y_source'] - all_sources.iloc[idx]['y_mic']
                dz = all_sources.iloc[idx]['z_source'] - all_sources.iloc[idx]['z_mic']
                r_dist = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-8
                row_vec[1:4] = [dx/r_dist, dy/r_dist, dz/r_dist]
            elif idx < len(all_sources):
                # noise
                row_vec[0] = 0.0
                dx = all_sources.iloc[idx]['x_source'] - all_sources.iloc[idx]['x_mic']
                dy = all_sources.iloc[idx]['y_source'] - all_sources.iloc[idx]['y_mic']
                dz = all_sources.iloc[idx]['z_source'] - all_sources.iloc[idx]['z_mic']
                r_dist = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-8
                row_vec[1:4] = [dx/r_dist, dy/r_dist, dz/r_dist]
            else:
                # empty track
                row_vec[0] = 0.0
                row_vec[1:4] = [0.0, 0.0, 0.0]

            if idx < len(all_sources):
                row_vec[4:] = all_sources.iloc[idx][meta_cols].values

            Y_sample.append(row_vec)

        Y_list.append(np.array(Y_sample, dtype=object))

    # ==============================
    # Convert & save
    # ==============================
    X_tensor = torch.tensor(np.array(X_list, dtype=np.float32))
    # Only keep first 4 float values in Y and convert to torch tensor
    Y_tensor = torch.tensor(
        np.array([[row[:4] for row in sample] for sample in Y_list], dtype=np.float32)
    )

    torch.save({"X": X_tensor, "Y": Y_tensor}, output_file)

    print(f"Dataset saved as {output_file}")
    print("X shape:", X_tensor.shape)
    print("Y shape:", Y_tensor.shape)


# ==============================
# Run
# ==============================
if __name__ == "__main__":
    start = time.time()

    csv_file = "../../../data/harp/R800_P10_S5_Order4_24kHz_shoeBox_ISM_direct/Generated_HOA_SRIR_data.csv"
    wav_folder = "../../../data/harp/R800_P10_S5_Order4_24kHz_shoeBox_ISM_direct"
    output_file = "../../../data/harp/R800_P10_S5_Order4_24kHz_shoeBox_ISM_direct/harp_dataset_scm.pt"

    main(
        csv_file,
        wav_folder,
        output_file,
        min_voice=1,
        max_voice=1,
        max_noise=0,
        vol_aug_voice=0,
        vol_aug_noise=0.5,
        noise_volume_ratio=0
    )

    # ==============================
    # Performance summary
    # ==============================
    print(file=sys.stderr)
    rusage_s = resource.getrusage(resource.RUSAGE_SELF)
    rusage_c = resource.getrusage(resource.RUSAGE_CHILDREN)

    print(f'Walltime {time.time() - start:.2f} s', file=sys.stderr)
    print(f'User time: {rusage_s.ru_utime + rusage_c.ru_utime:.2f} s', file=sys.stderr)
    print(f'System time: {rusage_s.ru_stime + rusage_c.ru_stime:.2f} s', file=sys.stderr)
    print(f'MaxRSS: {(rusage_s.ru_maxrss + rusage_c.ru_maxrss)/2**20:.3f} GiB', file=sys.stderr)