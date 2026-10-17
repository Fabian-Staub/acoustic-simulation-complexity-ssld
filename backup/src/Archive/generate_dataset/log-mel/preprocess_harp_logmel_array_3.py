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
from tqdm import tqdm

def detect_active_regions_relative(signal, fs, frame_len=1024, hop_len=512, rel_threshold=0.1):
    """
    Returns a list of (start, end) indices of active speech regions.
    Active regions are detected based on RMS energy relative to the max RMS.
    """
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
        return 0  # fallback to start

    for _ in range(max_trials):
        candidates = [r for r in active_regions if (r[1]-r[0]) >= segment_len_samples * min_active_ratio]

        if candidates:
            start, end = random.choice(candidates)
            max_start = max(start, end - segment_len_samples)
            start_sample = random.randint(start, max_start)
        else:
            start, end = max(active_regions, key=lambda r: r[1]-r[0])
            start_sample = start

        # Compute actual active ratio
        active_count = sum(
            max(0, min(r_end, start_sample + segment_len_samples) - max(r_start, start_sample))
            for r_start, r_end in active_regions
        )

        if active_count / segment_len_samples >= min_active_ratio:
            return start_sample

    return start_sample  # fallback after max_trials

def main(csv_file, csv_output_file, wav_folder, speech_folder, noise_folder, output_file,
                              order=4, fs=24000, segment_len_s=1.0,
                              min_voice=1, max_voice=3,
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

    df = pd.read_csv(csv_file)
    df['r'] = df['Name'].str.extract(r'R(\d+)_')[0].astype(int)
    df['p'] = df['Name'].str.extract(r'_P(\d+)_')[0].astype(int)
    df['s'] = df['Name'].str.extract(r'_S(\d+)_')[0].astype(int)

    grouped = df.groupby(['r','p'])

    # Available speech and noise files
    speech_files = [os.path.join(speech_folder, f) for f in os.listdir(speech_folder) if f.endswith('.wav')]
    noise_files  = [os.path.join(noise_folder, f) for f in os.listdir(noise_folder) if f.endswith('.wav')]

    X_list, Y_list = [], []

    for (room, receiver), group in tqdm(grouped, total=len(grouped), desc="Generating samples"):
        rir_sources = group

        num_voice = random.randint(min_voice, max_voice)
        voice_rirs = rir_sources.sample(num_voice)
        remaining_rirs = rir_sources.drop(voice_rirs.index)

        num_noise = random.randint(0, min(max_noise, len(remaining_rirs)))
        if num_noise > 0:
            noise_rirs = remaining_rirs.sample(num_noise)
        else:
            noise_rirs = pd.DataFrame(columns=rir_sources.columns)

        # Load random speech and noise files
        voice_signals = [librosa.load(random.choice(speech_files), sr=fs)[0] for _ in range(num_voice)]
        noise_signals = [librosa.load(random.choice(noise_files), sr=fs)[0] for _ in range(num_noise)]

        # normalize voice signals to peak 0.9
        normalized_voice_signals = []
        for sig in voice_signals:
            peak = np.max(np.abs(sig)) + 1e-8
            normalized_voice_signals.append(0.9 * sig / peak)

        # normalize noise signals to peak 0.9 * noise_volume_ratio
        normalized_noise_signals = []
        for sig in noise_signals:
            peak = np.max(np.abs(sig)) + 1e-8
            normalized_noise_signals.append((0.9 * noise_volume_ratio) * sig / peak)

        # overwrite originals (optional)
        voice_signals = normalized_voice_signals
        noise_signals = normalized_noise_signals

        def convolve_rirs(rirs, sigs):
            """
            Convolve each signal with its corresponding HOA RIR.
            Returns: list of arrays [NUM_SH, T]
            """
            convolved = []

            for rir_row, sig in zip(rirs.itertuples(), sigs):
                rir_path = os.path.join(wav_folder, rir_row.Name)
                sr, rir = wavfile.read(rir_path)  # shape: (T, NUM_SH) or (T,) if mono

                if rir.ndim == 1:
                    rir = rir[:, None]

                rir = rir.astype(np.float32)

                # Normalize RIR (optional but recommended)
                rir = rir / (np.max(np.abs(rir)) + 1e-8)

                num_channels = rir.shape[1]
                sig_len = len(sig)

                # Convolve each HOA channel
                out = []
                for ch in range(num_channels):
                    conv = fftconvolve(sig, rir[:, ch], mode='full')
                    out.append(conv)

                out = np.stack(out, axis=0)  # [NUM_SH, T]
                convolved.append(out)

            return convolved


        # --- Convolution ---
        voice_convolved = convolve_rirs(voice_rirs, voice_signals)
        noise_convolved = convolve_rirs(noise_rirs, noise_signals) if num_noise > 0 else []

        # --- Segment extraction ---
        voice_segments = []
        for sig in voice_convolved:
            mono = np.mean(sig, axis=0)
            active_regions = detect_active_regions_relative(mono, fs)

            start_idx = random_segment_with_activity(
                mono,
                segment_len_samples,
                active_regions
            )

            seg_multi = sig[:, start_idx:start_idx + segment_len_samples]

            if seg_multi.shape[1] < segment_len_samples:
                pad_width = segment_len_samples - seg_multi.shape[1]
                seg_multi = np.pad(seg_multi, ((0, 0), (0, pad_width)))

            voice_segments.append(seg_multi)


        noise_segments = []
        for sig in noise_convolved:
            mono = np.mean(sig, axis=0)

            # For noise: just random crop
            if len(mono) > segment_len_samples:
                start_idx = random.randint(0, len(mono) - segment_len_samples)
            else:
                start_idx = 0

            seg_multi = sig[:, start_idx:start_idx + segment_len_samples]

            if seg_multi.shape[1] < segment_len_samples:
                pad_width = segment_len_samples - seg_multi.shape[1]
                seg_multi = np.pad(seg_multi, ((0, 0), (0, pad_width)))

            noise_segments.append(seg_multi)


        # --- Sum signals ---
        total_sum = np.zeros_like(voice_segments[0])

        for v in voice_segments:
            total_sum += v

        if num_noise > 0:
            for n in noise_segments:
                total_sum += n


        # --- Normalize final signal ---
        peak = np.max(np.abs(total_sum)) + 1e-8
        total_sum = total_sum / peak * 0.9

        '''
        ambi = total_sum.T.astype(np.float32)
        output_wav_path = os.path.join("output_ambisonics.wav")
        wavfile.write(output_wav_path, fs, ambi)
        print(f"Saved ambisonics WAV: {output_wav_path}")
        '''

        X_mel = []
        for ch in range(NUM_SH):
            S = librosa.feature.melspectrogram(y=total_sum[ch], sr=fs, n_fft=win_len,
                                               hop_length=hop_len, center=False, n_mels=64)
            X_mel.append(S.T)
        X_mel = np.stack(X_mel, axis=0)
        X_mel_db = librosa.power_to_db(X_mel, ref=np.max(X_mel), top_db=60.0)

        X_min = X_mel_db.min()
        X_max = X_mel_db.max()
        X_mel_norm = (X_mel_db - X_min) / (X_max - X_min + 1e-8)
        X_list.append(X_mel_norm.astype(np.float32))

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
    Y_tensor = torch.tensor(
        np.array([[row[:4] for row in sample] for sample in Y_list], dtype=np.float32)
    )

    torch.save({"X": X_tensor, "Y": Y_tensor}, output_file)
    print(f"Dataset saved as {output_file}")
    print("X shape:", X_tensor.shape)
    print("Y shape:", Y_tensor.shape)

    with open(csv_output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ['voice_active', 'dx', 'dy', 'dz', 'Name','RT','c50',
                  'x_room','y_room','z_room','x_source','y_source','z_source',
                  'x_mic','y_mic','z_mic','status']
        writer.writerow(header)
        for sample in Y_list:
            for row in sample:
                writer.writerow([str(item) for item in row])

    print(f"Y data saved to CSV as {csv_output_file}")

    rusage_s = resource.getrusage(resource.RUSAGE_SELF)
    rusage_c = resource.getrusage(resource.RUSAGE_CHILDREN)
    print(f'Walltime {time.time() - start:.2f} s', file=sys.stderr)
    print(f'User time: {rusage_s.ru_utime + rusage_c.ru_utime:.2f} s', file=sys.stderr)
    print(f'System time: {rusage_s.ru_stime + rusage_c.ru_stime:.2f} s', file=sys.stderr)
    print(f'MaxRSS: {(rusage_s.ru_maxrss + rusage_c.ru_maxrss)/2**20:.3f} GiB', file=sys.stderr)


if __name__ == "__main__":
    csv_file = "../../../data/harp/R800_P10_S5_Order4_24kHz_shoeBox_ISM_direct/Generated_HOA_SRIR_data.csv"
    csv_output_file = "../../../data/harp/R800_P10_S5_Order4_24kHz_shoeBox_ISM_direct/harp_dataset.csv"
    rir_folder = "../../../data/harp/R800_P10_S5_Order4_24kHz_shoeBox_ISM_direct" 
    output_file = "../../../data/harp/R800_P10_S5_Order4_24kHz_shoeBox_ISM_direct/harp_dataset.pt" 
    speech_folder = "../../../data/audio_exports_Harvard_Sets"
    noise_folder = "../../../data/noises"

    main(
        csv_file, csv_output_file, rir_folder, speech_folder, noise_folder, output_file,
        order=4, fs=24000, segment_len_s=1.0,
        min_voice=1, max_voice=1,
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