import os
import numpy as np
import pandas as pd
import torch
from scipy.io import wavfile
import random
import librosa

def generate_hoa_dataset_conv(csv_file, wav_folder, speech_folder, noise_folder, output_file,
                              order=4, fs=24000, segment_len_s=1.0,
                              min_voice=1, max_voice=3,
                              vol_aug_speech=0.0, vol_aug_noise=0.0,
                              max_noise=2):
    """
    Generate HOA dataset by convolving multi-channel HOA RIRs with speech and noise signals,
    computing log-mel spectrograms, applying volume augmentation, and saving voice/noise activity + ACCDOA + metadata.
    """

    NUM_SH = (order + 1) ** 2
    hop_len = int(0.02*fs)
    win_len = int(0.04*fs)
    segment_len_samples = int(segment_len_s * fs + hop_len)  # add hop_len for last frame

    df = pd.read_csv(csv_file)
    df['r'] = df['Name'].str.extract(r'R(\d+)_')[0].astype(int)
    df['p'] = df['Name'].str.extract(r'_P(\d+)_')[0].astype(int)
    df['s'] = df['Name'].str.extract(r'_S(\d+)_')[0].astype(int)

    grouped = df.groupby(['r','p'])

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

        # Select random segments
        def random_segment(sig):
            if len(sig) <= segment_len_samples:
                padded = np.zeros(segment_len_samples, dtype=sig.dtype)
                padded[:len(sig)] = sig
                return padded
            else:
                start = random.randint(0, len(sig)-segment_len_samples)
                return sig[start:start+segment_len_samples]

        voice_segments = [random_segment(sig) for sig in voice_signals]
        noise_segments = [random_segment(sig) for sig in noise_signals]

        # Convolve multi-channel HOA RIRs
        def convolve_rirs(rirs, segments, vol_aug):
            out = np.zeros((NUM_SH, segment_len_samples), dtype=np.float32)
            for rir, seg in zip(rirs.itertuples(), segments):
                rir_path = os.path.join(wav_folder, rir.Name)
                _, rir_wav = wavfile.read(rir_path)  # rir_wav: samples × NUM_SH
                if rir_wav.ndim == 1:
                    rir_wav = np.expand_dims(rir_wav, axis=1)
                rir_ch = rir_wav.shape[1]

                if rir_ch < NUM_SH:
                    raise ValueError(f"RIR {rir.Name} has too few channels ({rir_ch}), expected {NUM_SH}")
                if rir_ch != NUM_SH:
                    print(f"Warning: RIR {rir.Name} has {rir_ch} channels, expected {NUM_SH}. Truncating to first {NUM_SH} channels.")

                # truncate higher orders
                rir_wav = rir_wav[:, :NUM_SH]

                # Convolve each SH channel
                conv = np.zeros((NUM_SH, segment_len_samples), dtype=np.float32)
                for ch in range(NUM_SH):
                    c = np.convolve(seg, rir_wav[:, ch])[:segment_len_samples]
                    vol = random.uniform(max(0.0, 1.0 - vol_aug), 1.0)
                    conv[ch] = c * vol

                out += conv  # sum contributions from multiple sources
            return out

        voice_sum = convolve_rirs(voice_rirs, voice_segments, vol_aug_speech)

        if num_noise > 0:
            noise_sum = convolve_rirs(noise_rirs, noise_segments, vol_aug_noise)
            total_sum = voice_sum + noise_sum
        else:
            total_sum = voice_sum

        # Compute log-mel spectrogram per HOA channel
        X_mel = []
        for ch in range(NUM_SH):
            S = librosa.feature.melspectrogram(y=total_sum[ch], sr=fs, n_fft=win_len,
                                               hop_length=hop_len, center=False)
            S_db = librosa.power_to_db(S, ref=np.max)
            X_mel.append(S_db)
        X_mel = np.stack(X_mel, axis=0)

        # Per-sample normalization across all channels
        X_min = X_mel.min()
        X_max = X_mel.max()
        X_mel_norm = (X_mel - X_min) / (X_max - X_min + 1e-8)  # epsilon avoids div by zero

        X_list.append(X_mel_norm.astype(np.float32))

        # Prepare Y: always 5 rows
        Y_sample = []
        meta_cols = ['Name','RT','c50','x_room','y_room','z_room',
                     'x_source','y_source','z_source','x_mic','y_mic','z_mic','status']

        for rir in rir_sources.itertuples():
            row_vec = np.zeros(1 + 3 + len(meta_cols), dtype=object)  # activity + xyz + metadata

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
                row_vec[0] = 0.0  # unused
                row_vec[1:4] = [0.0, 0.0, 0.0]

            row_vec[4:] = rir_sources.loc[rir.Index, meta_cols].values
            Y_sample.append(row_vec)

        Y_list.append(np.array(Y_sample, dtype=object))

    X_tensor = torch.tensor(np.array(X_list, dtype=np.float32))
    Y_array  = np.array(Y_list, dtype=object)

    torch.save({"X": X_tensor, "Y": Y_array}, output_file)
    print(f"Dataset saved as {output_file}")
    print("X shape:", X_tensor.shape)
    print("Y shape:", Y_array.shape)
    return X_tensor, Y_array


# Example usage
csv_file = "../../data/harp/Generated_HOA_IRs_ISM/Generated_HOA_SRIR_data.csv"
rir_folder = "../../data/harp/Generated_HOA_IRs_ISM" 
output_file = "../../data/harp/Generated_HOA_IRs_ISM/harp_dataset.pt" 
speech_folder = "../../data/audio_exports_Harvard_Sets"
noise_folder = "../../data/noises"

X, Y = generate_hoa_dataset_conv(csv_file, rir_folder, speech_folder, noise_folder, output_file,
                              order=4, fs=24000, segment_len_s=1.0,
                              min_voice=1, max_voice=3,
                              vol_aug_speech=0.0, vol_aug_noise=0.0, max_noise=2)