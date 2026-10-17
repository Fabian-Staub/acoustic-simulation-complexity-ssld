import os
import random
import numpy as np
import librosa
from tqdm import tqdm


def float_to_pcm16(signal):
    """
    Convert mono float signal to int16 PCM for WebRTC VAD.
    """
    signal = np.asarray(signal, dtype=np.float32)

    if signal.ndim > 1:
        signal = np.mean(signal, axis=0)

    signal = np.nan_to_num(signal)
    signal = np.clip(signal, -1.0, 1.0)

    return (signal * 32767.0).astype(np.int16)


def make_repeated_speech_track(speech, fs, rng):
    """
    silence_start + speech + silence_middle + speech + silence_end

    Each silence length is random between 0 and 0.5 s.
    """
    silence_start_len = int(rng.uniform(0.0, 4) * fs)
    silence_middle_len = int(rng.uniform(0.0, 2) * fs)
    silence_end_len = int(rng.uniform(0.0, 4) * fs)

    return np.concatenate(
        [
            np.zeros(silence_start_len, dtype=np.float32),
            speech.astype(np.float32),
            np.zeros(silence_middle_len, dtype=np.float32),
            speech.astype(np.float32),
            np.zeros(silence_end_len, dtype=np.float32),
        ]
    ).astype(np.float32)


def pad_tracks_to_same_length(tracks):
    max_len = max(len(track) for track in tracks)

    padded_tracks = []
    for track in tracks:
        if len(track) < max_len:
            track = np.pad(track, (0, max_len - len(track)), mode="constant")
        padded_tracks.append(track.astype(np.float32))

    return padded_tracks


def pad_frame_masks_to_same_length(frame_masks):
    max_len = max(len(mask) for mask in frame_masks)

    padded_masks = []
    for mask in frame_masks:
        if len(mask) < max_len:
            mask = np.pad(mask, (0, max_len - len(mask)), mode="constant")
        padded_masks.append(mask.astype(bool))

    return padded_masks


def energy_vad_on_stft_time_frames(
    signal,
    fs=16000,
    hop_length=320,
    threshold_db=-40.0,
    adaptive=True,
    margin_db=6.0,
):
    """
    Lightweight energy-based VAD with one decision per STFT hop frame.

    Returns:
        Boolean array with one activity decision per frame.
    """
    signal = np.asarray(signal, dtype=np.float32)
    signal = np.nan_to_num(signal)

    remainder = len(signal) % hop_length
    if remainder != 0:
        signal = np.pad(signal, (0, hop_length - remainder), mode="constant")

    frames = signal.reshape(-1, hop_length)

    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    db = 20.0 * np.log10(rms + 1e-12)

    if adaptive:
        # Estimate noise floor from quieter frames.
        noise_floor = np.percentile(db, 20)
        threshold = noise_floor + margin_db
    else:
        threshold = threshold_db

    return db > threshold


def estimate_concurrent_speech_percentages_per_stft_frame(
    speech_folder,
    fs=16000,
    n_trials=10000,
    n_speakers=3,
    seed=0,
    hop_length=320,
    vad_mode=2,
):
    rng = np.random.default_rng(seed)
    random.seed(seed)

    speech_files = [
        os.path.join(speech_folder, f)
        for f in os.listdir(speech_folder)
        if f.lower().endswith(".wav")
    ]

    if len(speech_files) < n_speakers:
        raise ValueError(
            f"Need at least {n_speakers} wav files in speech_folder, "
            f"but found {len(speech_files)}."
        )

    counts = np.zeros(n_speakers + 1, dtype=np.int64)

    for _ in tqdm(range(n_trials), desc="Running frame-level overlap simulation"):
        selected_files = rng.choice(speech_files, size=n_speakers, replace=False)

        tracks = []

        for wav_path in selected_files:
            # Mono only, resampled to 16 kHz.
            speech, _ = librosa.load(wav_path, sr=fs, mono=True)

            speech = 0.9 * speech / (np.max(np.abs(speech)) + 1e-8)

            track = make_repeated_speech_track(speech, fs, rng)
            tracks.append(track)

        tracks = pad_tracks_to_same_length(tracks)

        frame_masks = []

        for track in tracks:
            frame_mask = energy_vad_on_stft_time_frames(
                track,
                fs=fs,
                hop_length=hop_length,
                adaptive=True,
                margin_db=6.0,
            )

            frame_masks.append(frame_mask)

        frame_masks = pad_frame_masks_to_same_length(frame_masks)
        frame_masks = np.stack(frame_masks, axis=0)

        active_per_frame = np.sum(frame_masks, axis=0)

        for k in range(n_speakers + 1):
            counts[k] += np.sum(active_per_frame == k)

    percentages = 100.0 * counts / np.sum(counts)

    print("\nPercentage of STFT time frames with k active speakers:")
    for k, percentage in enumerate(percentages):
        print(f"{k} active speakers: {percentage:.2f}%")

    return percentages, counts


if __name__ == "__main__":
    speech_folder = "../../data/audio_exports_Harvard_Sets"

    estimate_concurrent_speech_percentages_per_stft_frame(
        speech_folder=speech_folder,
        fs=16000,
        n_trials=100,
        n_speakers=3,
        seed=0,
        hop_length=320,
        vad_mode=2,
    )