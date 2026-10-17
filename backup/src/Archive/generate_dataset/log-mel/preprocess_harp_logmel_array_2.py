import os
import numpy as np
import pandas as pd
import torch
from scipy.io import wavfile
import random
import librosa
from scipy.signal import fftconvolve
import csv
import sys
import resource
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def main(csv_file, csv_output_file, wav_folder, speech_folder, noise_folder, output_file,
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
    start = time.time()

    NUM_SH = (order + 1) ** 2
    hop_len = int(0.02*fs)
    win_len = int(0.04*fs)
    segment_len_samples = int(segment_len_s * fs + hop_len)

    # Pre-context for convolution
    pre_context_s = 0.15
    pre_context = int(pre_context_s * fs)

    segment_total = segment_len_samples + pre_context

    df = pd.read_csv(csv_file, nrows=5)  # Read only first 5 rows for testing
    df['r'] = df['Name'].str.extract(r'R(\d+)_')[0].astype(int)
    df['p'] = df['Name'].str.extract(r'_P(\d+)_')[0].astype(int)
    df['s'] = df['Name'].str.extract(r'_S(\d+)_')[0].astype(int)

    grouped = df.groupby(['r','p'])

    # Available speech and noise files
    speech_files = [os.path.join(speech_folder, f) for f in os.listdir(speech_folder) if f.endswith('.wav')]
    noise_files  = [os.path.join(noise_folder, f) for f in os.listdir(noise_folder) if f.endswith('.wav')]

    X_list, Y_list = [], []

    for (room, receiver), group in tqdm(grouped, total=len(grouped), desc="Generating samples"):
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
                # ---- Load RIR ----
                _, rir_wav = wavfile.read(rir_path)

                if rir_wav.ndim == 1:
                    rir_wav = rir_wav[:, None]

                rir_ch = rir_wav.shape[1]
                if rir_ch < NUM_SH:
                    raise ValueError(
                        f"RIR {rir.Name} has too few channels ({rir_ch}), expected {NUM_SH}"
                    )

                rir_wav = rir_wav[:, :NUM_SH].astype(np.float32)
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
        total_sum = total_sum / peak * 0.9  # skaliere alles auf ±0.99
        

        # Transpose to (samples, channels)
        ambi = total_sum.T  # shape: (samples, 25)

        # Convert to float32 (recommended for multichannel)
        ambi = ambi.astype(np.float32)

        # Save
        output_wav_path = os.path.join("output_ambisonics.wav")
        wavfile.write(output_wav_path, fs, ambi)

        print(f"Saved ambisonics WAV: {output_wav_path}")


        # Compute log-mel spectrogram per HOA channel
        X_mel = []
        for ch in range(NUM_SH):
            S = librosa.feature.melspectrogram(y=total_sum[ch], sr=fs, n_fft=win_len,
                                               hop_length=hop_len, center=False, n_mels=64)
            X_mel.append(S.T)
        X_mel = np.stack(X_mel, axis=0)
        X_mel_db = librosa.power_to_db(X_mel, ref=np.max(X_mel), top_db=30.0)

        # Per-sample normalization across all channels
        X_min = X_mel_db.min()
        X_max = X_mel_db.max()
        X_mel_norm = (X_mel_db - X_min) / (X_max - X_min + 1e-8)

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

    # Performance summary
    print(file=sys.stderr)
    rusage_s = resource.getrusage(resource.RUSAGE_SELF)
    rusage_c = resource.getrusage(resource.RUSAGE_CHILDREN)
    print(f'Walltime {time.time() - start:.2f} s', file=sys.stderr)
    print(f'User time: {rusage_s.ru_utime + rusage_c.ru_utime:.2f} s', file=sys.stderr)
    print(f'System time: {rusage_s.ru_stime + rusage_c.ru_stime:.2f} s', file=sys.stderr)
    print(f'MaxRSS: {(rusage_s.ru_maxrss + rusage_c.ru_maxrss)/2**20:.3f} GiB', file=sys.stderr)


if __name__ == "__main__":
    csv_file = "../../../data/harp/R800_P10_S5_Order4_24kHz_shoeBox_ISM/Generated_HOA_SRIR_data.csv"
    csv_output_file = "../../../data/harp/R800_P10_S5_Order4_24kHz_shoeBox_ISM/harp_dataset.csv"
    rir_folder = "../../../data/harp/R800_P10_S5_Order4_24kHz_shoeBox_ISM" 
    output_file = "../../../data/harp/R800_P10_S5_Order4_24kHz_shoeBox_ISM/harp_dataset.pt" 
    speech_folder = "../../../data/audio_exports_Harvard_Sets"
    noise_folder = "../../../data/noises"

    main(
        csv_file, csv_output_file, rir_folder, speech_folder, noise_folder, output_file,
        order=4, fs=24000, segment_len_s=1.0,
        min_voice=1, max_voice=1,
        vol_aug_speech=0, vol_aug_noise=0.75,
        max_noise=0,
        noise_volume_ratio=0.5
    )


    '''
    parser = argparse.ArgumentParser(
        description="Generate HOA dataset using RIR convolution with speech and noise signals."
    )

    # Paths
    parser.add_argument("--csv_file", type=str, required=True,
                        help="CSV file containing RIR metadata")
    parser.add_argument("--rir_folder", type=str, required=True,
                        help="Folder containing HOA RIR wav files")
    parser.add_argument("--speech_folder", type=str, required=True,
                        help="Folder containing speech wav files")
    parser.add_argument("--noise_folder", type=str, required=True,
                        help="Folder containing noise wav files")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output .pt dataset file")

    # Dataset parameters
    parser.add_argument("--order", type=int, default=4,
                        help="HOA order (default: 4)")
    parser.add_argument("--fs", type=int, default=24000,
                        help="Sampling rate (default: 24000)")
    parser.add_argument("--segment_len_s", type=float, default=1.0,
                        help="Segment length in seconds (default: 1.0)")

    # Voice parameters
    parser.add_argument("--min_voice", type=int, default=1,
                        help="Minimum number of speech sources")
    parser.add_argument("--max_voice", type=int, default=3,
                        help="Maximum number of speech sources")

    # Noise parameters
    parser.add_argument("--max_noise", type=int, default=2,
                        help="Maximum number of noise sources")
    parser.add_argument("--noise_volume_ratio", type=float, default=1.0,
                        help="Relative volume scaling for noise")

    # Volume augmentation
    parser.add_argument("--vol_aug_speech", type=float, default=0.0,
                        help="Speech volume augmentation range")
    parser.add_argument("--vol_aug_noise", type=float, default=0.0,
                        help="Noise volume augmentation range")

    args = parser.parse_args()


    main(
        csv_file=args.csv_file,
        rir_folder=args.rir_folder,
        speech_folder=args.speech_folder,
        noise_folder=args.noise_folder,
        output_file=args.output_file,
        order=args.order,
        fs=args.fs,
        segment_len_s=args.segment_len_s,
        min_voice=args.min_voice,
        max_voice=args.max_voice,
        vol_aug_speech=args.vol_aug_speech,
        vol_aug_noise=args.vol_aug_noise,
        max_noise=args.max_noise,
        noise_volume_ratio=args.noise_volume_ratio
    )
    '''