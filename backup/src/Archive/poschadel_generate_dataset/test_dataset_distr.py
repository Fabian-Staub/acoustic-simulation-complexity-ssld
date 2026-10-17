import os
import random
import numpy as np
import librosa
from tqdm import tqdm
import matplotlib.pyplot as plt


def detect_speech_activity_from_waveform(
    speech,
    fs,
    hop_length=320,
    frame_length=640,
    top_db=35.0,
):
    """
    Energy-based VAD from the speech waveform itself.

    Quiet parts inside the speech sample are marked inactive.

    top_db:
        Frames more than `top_db` dB below the peak RMS are inactive.
        Smaller value = stricter VAD.
        Larger value = more permissive VAD.
    """
    speech = np.asarray(speech, dtype=np.float32)

    if len(speech) == 0:
        return np.zeros(0, dtype=bool)

    rms = librosa.feature.rms(
        y=speech,
        frame_length=frame_length,
        hop_length=hop_length,
        center=False,
    )[0]

    if len(rms) == 0:
        return np.zeros(len(speech), dtype=bool)

    peak_rms = np.max(rms)

    if peak_rms <= 0:
        return np.zeros(len(speech), dtype=bool)

    threshold = peak_rms * (10.0 ** (-top_db / 20.0))

    frame_mask = rms > threshold

    sample_mask = np.zeros(len(speech), dtype=bool)

    for frame_idx, active in enumerate(frame_mask):
        if active:
            start = frame_idx * hop_length
            end = min(start + frame_length, len(speech))
            sample_mask[start:end] = True

    return sample_mask


def randomly_trim_speech_edges(
    speech,
    fs,
    rng,
    max_start_trim_seconds=0.5,
    max_end_trim_seconds=0.5,
):
    """
    Randomly shorten a speech sample at the beginning and end.

    The start trim is sampled between 0 and `max_start_trim_seconds` seconds.
    The end trim is sampled between 0 and `max_end_trim_seconds` seconds.
    Trimming is applied before the sample is doubled in
    `make_repeated_speech_track`.
    """
    speech = np.asarray(speech, dtype=np.float32)

    max_start_trim_seconds = max(0.0, float(max_start_trim_seconds))
    max_end_trim_seconds = max(0.0, float(max_end_trim_seconds))

    if len(speech) <= 1 or (max_start_trim_seconds <= 0 and max_end_trim_seconds <= 0):
        return speech

    start_trim_len = int(rng.uniform(0.0, max_start_trim_seconds) * fs)
    end_trim_len = int(rng.uniform(0.0, max_end_trim_seconds) * fs)

    # Keep at least one sample if the file is very short.
    total_trim_len = start_trim_len + end_trim_len

    if total_trim_len >= len(speech):
        max_total_trim_len = len(speech) - 1

        if max_total_trim_len <= 0:
            return speech

        if total_trim_len > 0:
            scale = max_total_trim_len / total_trim_len
            start_trim_len = int(start_trim_len * scale)
            end_trim_len = max_total_trim_len - start_trim_len

    end_idx = len(speech) - end_trim_len

    return speech[start_trim_len:end_idx].astype(np.float32)


def make_repeated_speech_track(
    speech,
    fs,
    rng,
    hop_length=320,
    frame_length=640,
    vad_top_db=35.0,
):
    """
    Create:
        silence_start + speech + silence_middle + speech + silence_end

    Start/end silence lengths are random between 0 and 4 s.
    Middle silence length is random between 0 and 2 s.

    The speech activity mask is detected from the waveform itself, so quiet
    regions inside the speech sample are classified as inactive.

    Returns:
        track:
            Float waveform.

        sample_mask:
            Boolean sample-level speech activity mask.
            True means this sample belongs to detected active speech.
    """
    silence_start_len = int(rng.uniform(0.0, 0.0) * fs)
    silence_middle_len = int(rng.uniform(0.0, 0.0) * fs)
    silence_end_len = int(rng.uniform(0.0, 0.0) * fs)

    speech = np.asarray(speech, dtype=np.float32)

    speech_vad_mask = detect_speech_activity_from_waveform(
        speech=speech,
        fs=fs,
        hop_length=hop_length,
        frame_length=frame_length,
        top_db=vad_top_db,
    )

    track = np.concatenate(
        [
            np.zeros(silence_start_len, dtype=np.float32),
            speech,
            np.zeros(silence_middle_len, dtype=np.float32),
            speech,
            np.zeros(silence_end_len, dtype=np.float32),
        ]
    ).astype(np.float32)

    sample_mask = np.concatenate(
        [
            np.zeros(silence_start_len, dtype=bool),
            speech_vad_mask,
            np.zeros(silence_middle_len, dtype=bool),
            speech_vad_mask,
            np.zeros(silence_end_len, dtype=bool),
        ]
    )

    return track, sample_mask


def pad_tracks_to_same_length(tracks):
    max_len = max(len(track) for track in tracks)

    padded_tracks = []

    for track in tracks:
        if len(track) < max_len:
            track = np.pad(track, (0, max_len - len(track)), mode="constant")

        padded_tracks.append(track.astype(np.float32))

    return padded_tracks


def pad_sample_masks_to_same_length(sample_masks, target_len):
    padded_masks = []

    for mask in sample_masks:
        if len(mask) < target_len:
            mask = np.pad(
                mask,
                (0, target_len - len(mask)),
                mode="constant",
                constant_values=False,
            )

        padded_masks.append(mask.astype(bool))

    return padded_masks


def pad_frame_masks_to_same_length(frame_masks):
    max_len = max(len(mask) for mask in frame_masks)

    padded_masks = []

    for mask in frame_masks:
        if len(mask) < max_len:
            mask = np.pad(
                mask,
                (0, max_len - len(mask)),
                mode="constant",
                constant_values=False,
            )

        padded_masks.append(mask.astype(bool))

    return padded_masks


def sample_mask_to_frame_mask(sample_mask, hop_length=320):
    """
    Convert sample-level speech mask to frame-level mask.

    A frame is active if any sample in that frame belongs to detected speech.
    """
    sample_mask = np.asarray(sample_mask, dtype=bool)

    remainder = len(sample_mask) % hop_length

    if remainder != 0:
        sample_mask = np.pad(
            sample_mask,
            (0, hop_length - remainder),
            mode="constant",
            constant_values=False,
        )

    frames = sample_mask.reshape(-1, hop_length)

    return np.any(frames, axis=1)


def frame_mask_to_segments(frame_mask, fs, hop_length):
    """
    Convert a frame-level boolean mask into contiguous time segments.
    """
    frame_mask = np.asarray(frame_mask, dtype=bool)

    segments = []
    in_segment = False
    start_frame = None

    for frame_idx, active in enumerate(frame_mask):
        if active and not in_segment:
            in_segment = True
            start_frame = frame_idx

        elif not active and in_segment:
            end_frame = frame_idx
            start_time = start_frame * hop_length / fs
            end_time = end_frame * hop_length / fs
            segments.append((start_time, end_time))
            in_segment = False

    if in_segment:
        end_frame = len(frame_mask)
        start_time = start_frame * hop_length / fs
        end_time = end_frame * hop_length / fs
        segments.append((start_time, end_time))

    return segments


def plot_example_tracks_with_vad(
    tracks,
    frame_masks,
    fs,
    hop_length,
    output_png="example_tracks_with_vad.png",
):
    """
    Plot one example of the tracks and their frame-level speech activity masks.
    """
    n_speakers = len(tracks)

    fig, axes = plt.subplots(
        n_speakers,
        1,
        figsize=(14, 2.5 * n_speakers),
        sharex=True,
    )

    if n_speakers == 1:
        axes = [axes]

    for speaker_idx, ax in enumerate(axes):
        track = tracks[speaker_idx]
        frame_mask = frame_masks[speaker_idx]

        waveform_time = np.arange(len(track)) / fs

        ax.plot(waveform_time, track, linewidth=0.8)

        for start_time, end_time in frame_mask_to_segments(
            frame_mask,
            fs=fs,
            hop_length=hop_length,
        ):
            ax.axvspan(start_time, end_time, alpha=0.25)

        ax.set_ylabel(f"Speaker {speaker_idx + 1}")
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")

    fig.suptitle("Example Tracks with Energy-Based Ground-Truth VAD")
    fig.tight_layout()
    fig.savefig(output_png, dpi=200)
    plt.close(fig)

    print(f"Saved example VAD plot to: {output_png}")


def estimate_concurrent_speech_percentages_per_stft_frame(
    speech_folder,
    fs=16000,
    n_trials=10000,
    n_speakers=3,
    seed=0,
    hop_length=320,
    vad_frame_length=640,
    vad_top_db=35.0,
    max_start_trim_seconds=0.5,
    max_end_trim_seconds=0.5,
    save_example_plot=True,
    example_plot_path="example_tracks_with_vad.png",
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

    for trial_idx in tqdm(
        range(n_trials),
        desc="Running frame-level overlap simulation",
    ):
        selected_files = rng.choice(
            speech_files,
            size=n_speakers,
            replace=False,
        )

        tracks = []
        sample_masks = []

        for wav_path in selected_files:
            speech, _ = librosa.load(wav_path, sr=fs, mono=True)

            speech = np.asarray(speech, dtype=np.float32)

            peak = np.max(np.abs(speech))

            if peak > 0:
                speech = 0.9 * speech / peak

            speech = speech.astype(np.float32)

            speech = randomly_trim_speech_edges(
                speech=speech,
                fs=fs,
                rng=rng,
                max_start_trim_seconds=max_start_trim_seconds,
                max_end_trim_seconds=max_end_trim_seconds,
            )

            track, sample_mask = make_repeated_speech_track(
                speech=speech,
                fs=fs,
                rng=rng,
                hop_length=hop_length,
                frame_length=vad_frame_length,
                vad_top_db=vad_top_db,
            )

            tracks.append(track)
            sample_masks.append(sample_mask)

        tracks = pad_tracks_to_same_length(tracks)

        sample_masks = pad_sample_masks_to_same_length(
            sample_masks,
            target_len=len(tracks[0]),
        )

        frame_masks = []

        for sample_mask in sample_masks:
            frame_mask = sample_mask_to_frame_mask(
                sample_mask,
                hop_length=hop_length,
            )
            frame_masks.append(frame_mask)

        frame_masks = pad_frame_masks_to_same_length(frame_masks)
        frame_masks = np.stack(frame_masks, axis=0)

        if save_example_plot and trial_idx == 0:
            plot_example_tracks_with_vad(
                tracks=tracks,
                frame_masks=frame_masks,
                fs=fs,
                hop_length=hop_length,
                output_png=example_plot_path,
            )

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
        n_trials=1000,
        n_speakers=3,
        hop_length=320,
        vad_frame_length=640,
        vad_top_db=30.0,
        max_start_trim_seconds=1,
        max_end_trim_seconds=1,
        save_example_plot=True,
        example_plot_path="example_tracks_with_vad.png",
    )