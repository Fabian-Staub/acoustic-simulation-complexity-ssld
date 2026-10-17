import os
import numpy as np
import pandas as pd
import torch
from scipy.io import wavfile
import random
import librosa
from scipy.signal import fftconvolve
import csv

def generate_hoa_dataset_conv(csv_file, wav_folder, speech_folder, noise_folder, output_file,
                              order=4, fs=24000, segment_len_s=1.0,
                              min_voice=1, max_voice=3,
                              vol_aug_speech=0.0, vol_aug_noise=0.0,
                              max_noise=2,
                              noise_volume_ratio=1.0):
    """
    Generate HOA dataset by convolving multi-channel HOA RIRs with speech and noise signals,
    normalizing them, computing log-mel spectrograms, applying volume augmentation, 
    and saving voice/noise activity + ACCDOA + metadata.
    """

    NUM_SH = (order + 1) ** 2
    hop_len = int(0.02*fs)
    win_len = int(0.04*fs)
    segment_len_samples = int(segment_len_s * fs + hop_len)

    # Pre-context for convolution
    pre_context_s = 0.15
    pre_context = int(pre_context_s * fs)

    segment_total = segment_len_samples + pre_context

    df = pd.read_csv(csv_file)
    df['r'] = df['Name'].str.extract(r'R(\d+)_')[0].astype(int)
    df['p'] = df['Name'].str.extract(r'_P(\d+)_')[0].astype(int)
    df['s'] = df['Name'].str.extract(r'_S(\d+)_')[0].astype(int)

    grouped = df.groupby(['r','p'])

    rir_cache = {}

    # Available speech and noise files
    speech_files = [os.path.join(speech_folder, f) for f in os.listdir(speech_folder) if f.endswith('.wav')]
    noise_files  = [os.path.join(noise_folder, f) for f in os.listdir(noise_folder) if f.endswith('.wav')]

    X_list, Y_list = [], []

    for (room, receiver), group in grouped:
        rir_sources = group.sample(n=5)  # always 5 RIRs per mic

        # Voice sources
        num_voice = random.randint(min_voice, max_voice)
        voice_rirs = rir_sources.sample(num_voice)
        remaining_rirs = rir_sources.drop(voice_rirs.index)

        # Noise sources
        num_noise = random.randint(0, min(max_noise, len(remaining_rirs)))
        if num_noise > 0:
            noise_rirs = remaining_rirs.sample(num_noise)
        else:
            noise_rirs = pd.DataFrame(columns=rir_sources.columns)

        # Load random speech and noise files
        voice_signals = [librosa.load(random.choice(speech_files), sr=fs)[0] for _ in range(num_voice)]
        noise_signals = [librosa.load(random.choice(noise_files), sr=fs)[0] for _ in range(num_noise)]

        # Select random segments and zero pad beginning if necessary
        def random_segment(sig):

            if len(sig) <= segment_len_samples:
                padded = np.zeros(segment_total, dtype=sig.dtype)
                padded[pre_context:pre_context+len(sig)] = sig
                return padded

            start = random.randint(0, len(sig)-segment_len_samples)

            start_with_context = start - pre_context
            end_with_context = start + segment_len_samples

            segment = np.zeros(segment_total, dtype=sig.dtype)

            src_start = max(0, start_with_context)
            src_end = min(len(sig), end_with_context)

            dst_start = src_start - start_with_context
            dst_end = dst_start + (src_end - src_start)

            segment[dst_start:dst_end] = sig[src_start:src_end]

            return segment

        voice_segments = [random_segment(sig) for sig in voice_signals]
        noise_segments = [random_segment(sig) for sig in noise_signals]

        # Normalize segments and apply noise volume ratio
        def normalize_segment(sig, target_peak=0.99):
            # Avoid division by zero
            max_val = np.max(np.abs(sig)) + 1e-8
            normalized = sig / max_val * target_peak
            return normalized

        voice_segments = [normalize_segment(seg) for seg in voice_segments]
        noise_segments = [normalize_segment(seg) * noise_volume_ratio for seg in noise_segments]

        # Convolve multi-channel HOA RIRs
        def convolve_rirs(rirs, segments, vol_aug):
            out = np.zeros((NUM_SH, segment_len_samples), dtype=np.float32)
            for rir, seg in zip(rirs.itertuples(), segments):
                rir_path = os.path.join(wav_folder, rir.Name)
                # ---- RIR CACHE ----
                if rir_path not in rir_cache:
                    _, rir_wav = wavfile.read(rir_path)
                    if rir_wav.ndim == 1:
                        rir_wav = rir_wav[:, None]
                    rir_ch = rir_wav.shape[1]
                    if rir_ch < NUM_SH:
                        raise ValueError(
                            f"RIR {rir.Name} has too few channels ({rir_ch}), expected {NUM_SH}"
                        )
                    rir_cache[rir_path] = rir_wav[:, :NUM_SH].astype(np.float32)
                rir_wav = rir_cache[rir_path]
                # ---- FFT CONVOLUTION (vectorized) ----
                conv_full = np.zeros((NUM_SH, len(seg) + rir_wav.shape[0] - 1), dtype=np.float32)
                for ch in range(NUM_SH):
                    conv_full[ch] = fftconvolve(seg, rir_wav[:, ch], mode='full')
                # ---- crop pre-context ----
                conv = conv_full[:, pre_context:pre_context+segment_len_samples]
                # ---- volume augmentation ----
                vol = random.uniform(max(0.0, 1.0 - vol_aug), 1.0)
                out += conv * vol
            return out

        voice_sum = convolve_rirs(voice_rirs, voice_segments, vol_aug_speech)

        if num_noise > 0:
            noise_sum = convolve_rirs(noise_rirs, noise_segments, vol_aug_noise)
            total_sum = voice_sum + noise_sum
        else:
            total_sum = voice_sum

        peak = np.max(np.abs(total_sum)) + 1e-8
        if peak > 0.99:
            total_sum = total_sum / peak * 0.99  # skaliere alles auf ±0.99

        # Compute log-mel spectrogram per HOA channel
        X_mel = []
        for ch in range(NUM_SH):
            S = librosa.feature.melspectrogram(y=total_sum[ch], sr=fs, n_fft=win_len,
                                               hop_length=hop_len, center=False, n_mels=64)
            S_db = librosa.power_to_db(S, ref=np.max)
            X_mel.append(S_db.T)
        X_mel = np.stack(X_mel, axis=0)

        # Per-sample normalization across all channels
        X_min = X_mel.min()
        X_max = X_mel.max()
        X_mel_norm = (X_mel - X_min) / (X_max - X_min + 1e-8)

        X_list.append(X_mel_norm.astype(np.float32))

        # Prepare Y
        Y_sample = []
        meta_cols = ['Name','RT','c50','x_room','y_room','z_room',
                     'x_source','y_source','z_source','x_mic','y_mic','z_mic','status']

        for rir in rir_sources.itertuples():
            row_vec = np.zeros(1 + 3 + len(meta_cols), dtype=object)

            if rir.Index in voice_rirs.index:
                row_vec[0] = 1.0
                dx = rir.x_source - rir.x_mic
                dy = rir.y_source - rir.y_mic
                dz = rir.z_source - rir.z_mic
                r_dist = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-8
                row_vec[1:4] = [dx/r_dist, dy/r_dist, dz/r_dist]
            elif rir.Index in noise_rirs.index:
                row_vec[0] = 0.0
                dx = rir.x_source - rir.x_mic
                dy = rir.y_source - rir.y_mic
                dz = rir.z_source - rir.z_mic
                r_dist = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-8
                row_vec[1:4] = [dx/r_dist, dy/r_dist, dz/r_dist]
            else:
                row_vec[0] = 0.0
                row_vec[1:4] = [0.0, 0.0, 0.0]

            row_vec[4:] = rir_sources.loc[rir.Index, meta_cols].values
            Y_sample.append(row_vec)

        Y_list.append(np.array(Y_sample, dtype=object))

    X_tensor = torch.tensor(np.array(X_list, dtype=np.float32))
    # Only keep first 4 float values in Y and convert to torch tensor
    Y_tensor = torch.tensor(
        np.array([[row[:4] for row in sample] for sample in Y_list], dtype=np.float32)
    )

    torch.save({"X": X_tensor, "Y": Y_tensor}, output_file)
    print(f"Dataset saved as {output_file}")
    print("X shape:", X_tensor.shape)
    print("Y shape:", Y_tensor.shape)

    # After generating X_tensor and Y_list
    csv_output_file = "../../../data/harp/Generated_HOA_IRs_ISM/harp_dataset.csv"

    # Flatten Y_list: each sample has multiple sources (rows)
    with open(csv_output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        
        # Create header
        header = ['voice_active', 'dx', 'dy', 'dz', 'Name','RT','c50',
                'x_room','y_room','z_room','x_source','y_source','z_source',
                'x_mic','y_mic','z_mic','status']
        writer.writerow(header)
        
        # Write rows
        for sample in Y_list:
            for row in sample:
                # Convert all elements to str for safe CSV writing
                writer.writerow([str(item) for item in row])

    print(f"Y data saved to CSV as {csv_output_file}")

    return X_tensor, Y_tensor

# Example usage
csv_file = "../../../data/harp/Generated_HOA_IRs_ISM/Generated_HOA_SRIR_data.csv"
rir_folder = "../../../data/harp/Generated_HOA_IRs_ISM" 
output_file = "../../../data/harp/Generated_HOA_IRs_ISM/harp_dataset.pt" 
speech_folder = "../../../data/audio_exports_Harvard_Sets"
noise_folder = "../../../data/noises"

X, Y = generate_hoa_dataset_conv(
    csv_file, rir_folder, speech_folder, noise_folder, output_file,
    order=4, fs=24000, segment_len_s=1.0,
    min_voice=1, max_voice=3,
    vol_aug_speech=0.5, vol_aug_noise=0.75,
    max_noise=0,
    noise_volume_ratio=0.5
)