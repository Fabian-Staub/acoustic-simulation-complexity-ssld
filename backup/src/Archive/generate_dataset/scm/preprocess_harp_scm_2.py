import os
import numpy as np
import pandas as pd
import pyfar as pf
import torch
from scipy.io import wavfile
import random

def generate_hoa_dataset(csv_file, wav_folder, output_file,
                         order=4, fs=24000, min_sources=1, max_sources=3,
                         vol_aug=0, num_fractions=1, freq_range=(100, 8000)):
    """
    Generate HOA dataset with mixed sources, SCM, ACCDOA, active mask, and full CSV metadata.

    Parameters:
    - csv_file: Path to Generated_HOA_SRIR_data.csv
    - wav_folder: Folder containing WAV files
    - output_file: Path to save the PyTorch dataset (.pt)
    - order: HOA order (default 4)
    - fs: Sampling frequency (default 16kHz)
    - min_sources: Minimum number of sources to mix
    - max_sources: Maximum number of sources to mix
    - vol_aug: Maximum volume attenuation factor (0 to vol_aug)
    - num_fractions: Fractional octave bands (1=1/1 octave)
    - freq_range: Frequency range for octave band filtering
    """

    NUM_SH = (order + 1) ** 2  # HOA channels

    # ==============================
    # Helper functions
    # ==============================
    def azel_to_accdoa(azi_list, el_list, max_sources=max_sources):
        accdoa = np.zeros((max_sources, 3), dtype=np.float32)
        n = min(len(azi_list), max_sources)
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
            X = signal_bands.time[b]  # (n_channels, n_samples)
            cov_all_bands[b] = X @ X.T
        return cov_all_bands

    # ==============================
    # Load CSV metadata
    # ==============================
    df = pd.read_csv(csv_file)

    # Extract room (r), receiver (p), source (s) from the file name
    df['r'] = df['Name'].str.extract(r'R(\d+)_')[0].astype(int)
    df['p'] = df['Name'].str.extract(r'_P(\d+)_')[0].astype(int)
    df['s'] = df['Name'].str.extract(r'_S(\d+)_')[0].astype(int)

    grouped = df.groupby(['r', 'p'])

    X_list, Y_list = [], []

    for (room, receiver), group in grouped:
        # Randomly select number of sources
        num_sources = random.randint(min_sources, max_sources)
        sources = group.sample(num_sources)

        mixed_wav = None
        azi_list, el_list = [], []

        # Mix sources
        for _, row in sources.iterrows():
            wav_path = os.path.join(wav_folder, row['Name'])
            fs_wav, wav = wavfile.read(wav_path)
            wav = wav[:, :NUM_SH].T  # transpose to (channels, samples)
            
            # Random volume attenuation
            vol = random.uniform(max(0.0, 1.0 - vol_aug), 1.0)
            wav = wav * vol

            # Mix
            if mixed_wav is None:
                mixed_wav = wav
            else:
                min_len = min(mixed_wav.shape[1], wav.shape[1])
                mixed_wav = mixed_wav[:, :min_len] + wav[:, :min_len]

            # Compute az/el from source to microphone
            dx = row['x_source'] - row['x_mic']
            dy = row['y_source'] - row['y_mic']
            dz = row['z_source'] - row['z_mic']
            r_dist = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-8
            azi_list.append(np.arctan2(dy, dx))
            el_list.append(np.arcsin(dz / r_dist))

        # Pyfar signal
        signal_obj = pf.Signal(mixed_wav, fs)
        signal_bands = pf.dsp.filter.fractional_octave_bands(
            signal_obj, num_fractions=num_fractions, frequency_range=freq_range, order=15
        )

        # SCM normalized between 0 and 1
        cov_all_bands = compute_scm(signal_bands)

        # ACCDOA
        accdoa_targets = azel_to_accdoa(azi_list, el_list, max_sources=max_sources)
        active_mask = np.zeros((max_sources,), dtype=np.float32)
        active_mask[:len(azi_list)] = 1.0

        # Collect full metadata for each source
        meta_columns = ['Name','RT','c50','x_room','y_room','z_room',
                        'x_source','y_source','z_source','x_mic','y_mic','z_mic','status']
        meta_array = sources[meta_columns].to_numpy(dtype=object)  # keep Name and status as object

        # Combine ACCDOA + active mask + metadata
        Y_sample = []
        for i in range(max_sources):
            row_vec = np.zeros(3 + 1 + len(meta_columns), dtype=object)
            # ACCDOA
            row_vec[:3] = accdoa_targets[i]
            row_vec[3] = active_mask[i]
            if i < len(azi_list):
                row_vec[4:] = meta_array[i]
            Y_sample.append(row_vec)
        Y_sample = np.array(Y_sample, dtype=object)

        X_list.append(cov_all_bands)
        Y_list.append(Y_sample)

    # Convert X to float tensor, Y to object tensor (or leave as numpy object)
    X_tensor = torch.tensor(np.array(X_list, dtype=np.float32))
    # For Y, we can leave it as numpy array of object dtype to preserve strings
    Y_array = np.array(Y_list, dtype=object)

    # Save dataset
    torch.save({"X": X_tensor, "Y": Y_array}, output_file)

    print(f"Dataset saved as {output_file}")
    print("X shape:", X_tensor.shape)
    print("Y shape:", Y_array.shape)
    return X_tensor, Y_array

csv_file = "../../../data/harp/Generated_HOA_IRs_ISM/Generated_HOA_SRIR_data.csv"
wav_folder = "../../../data/harp/Generated_HOA_IRs_ISM"
output_file = "../../../data/harp/Generated_HOA_IRs_ISM/harp_dataset.pt"

X, Y = generate_hoa_dataset(csv_file, wav_folder, output_file,
                            min_sources=1, max_sources=3, vol_aug=0)