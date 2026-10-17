import os
import random
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from scipy.io import wavfile
from scipy.signal import fftconvolve
from spharpy.transforms import rotation_z_axis_real


# =========================
# Fixed configuration
# =========================

N_SPEAKERS = 3


# =========================
# VAD / repeated speech track
# =========================

def detect_speech_activity_from_waveform(
    speech,
    fs,
    hop_length=320,
    frame_length=640,
    top_db=35.0,
):
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
    speech = np.asarray(speech, dtype=np.float32)

    if len(speech) <= 1:
        return speech

    max_start_trim_seconds = max(0.0, float(max_start_trim_seconds))
    max_end_trim_seconds = max(0.0, float(max_end_trim_seconds))

    start_trim_len = int(rng.uniform(0.0, max_start_trim_seconds) * fs)
    end_trim_len = int(rng.uniform(0.0, max_end_trim_seconds) * fs)

    total_trim_len = start_trim_len + end_trim_len

    if total_trim_len >= len(speech):
        max_total_trim_len = len(speech) - 1

        if max_total_trim_len <= 0:
            return speech

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
    Creates:

        speech + speech

    No extra start, middle, or end silence is added.
    """

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
            speech,
            speech,
        ]
    ).astype(np.float32)

    sample_mask = np.concatenate(
        [
            speech_vad_mask,
            speech_vad_mask,
        ]
    )

    return track, sample_mask


# =========================
# HOA helpers
# =========================

def infer_hoa_order(num_channels):
    """
    Infer HOA order from number of channels.

    ACN HOA channel count must be:

        num_channels = (order + 1)^2

    Examples:
        1  channel  -> order 0
        4  channels -> order 1
        9  channels -> order 2
        16 channels -> order 3
        25 channels -> order 4
    """
    sqrt_channels = int(round(np.sqrt(num_channels)))

    if sqrt_channels * sqrt_channels != num_channels:
        raise ValueError(
            f"Invalid HOA channel count: {num_channels}. "
            f"Expected a perfect square: 1, 4, 9, 16, 25, ..."
        )

    order = sqrt_channels - 1

    if order < 0:
        raise ValueError(
            f"Invalid HOA order inferred from {num_channels} channels."
        )

    return order


def rotate_hoa(signal, theta):
    """
    Rotate HOA / ambisonics signal around the z-axis.

    Args:
        signal:
            HOA signal with shape (channels, samples)

        theta:
            Rotation angle in radians.

    Returns:
        Rotated HOA signal with shape (channels, samples)
    """
    signal = np.asarray(signal, dtype=np.float32)

    if signal.ndim != 2:
        raise ValueError(
            f"Expected HOA signal with shape (channels, samples), "
            f"but got shape {signal.shape}."
        )

    num_channels = signal.shape[0]
    hoa_order = infer_hoa_order(num_channels)

    R = rotation_z_axis_real(hoa_order, theta)

    return (R @ signal).astype(np.float32)


def check_same_hoa_channel_count(signals):
    """
    Ensures all convolved tracks have the same HOA channel count.
    """
    channel_counts = [sig.shape[0] for sig in signals]
    unique_channel_counts = sorted(set(channel_counts))

    if len(unique_channel_counts) != 1:
        raise ValueError(
            f"All RIRs must have the same HOA channel count. "
            f"Got channel counts: {unique_channel_counts}"
        )

    return unique_channel_counts[0]


# =========================
# Audio helpers
# =========================

def load_mono_speech(path, fs):
    speech, _ = librosa.load(path, sr=fs, mono=True)
    speech = np.asarray(speech, dtype=np.float32)

    peak = np.max(np.abs(speech))

    if peak > 0:
        speech = 0.9 * speech / peak

    return speech.astype(np.float32)


def read_wav_float(path, target_fs):
    """
    Reads mono or multichannel WAV as float32.

    Returns:
        audio with shape (channels, samples)
    """
    fs_in, audio = wavfile.read(path)

    audio = np.asarray(audio)

    if audio.dtype.kind in {"i", "u"}:
        max_val = np.iinfo(audio.dtype).max
        audio = audio.astype(np.float32) / max_val
    else:
        audio = audio.astype(np.float32)

    if audio.ndim == 1:
        audio = audio[None, :]
    else:
        # scipy reads multichannel as (samples, channels)
        audio = audio.T

    if fs_in != target_fs:
        resampled = []

        for ch in audio:
            ch_rs = librosa.resample(
                ch,
                orig_sr=fs_in,
                target_sr=target_fs,
            )
            resampled.append(ch_rs.astype(np.float32))

        audio = np.stack(resampled, axis=0)

    return audio.astype(np.float32)


def convolve_mono_with_multichannel_rir(
    track,
    rir,
    keep_original_length=True,
):
    """
    Args:
        track:
            Mono signal, shape (samples,)

        rir:
            Multichannel HOA RIR, shape (channels, samples)

        keep_original_length:
            If True, the convolved signal is cropped back to len(track).
            This prevents the RIR tail from increasing the output duration.

    Returns:
        Convolved signal, shape (channels, samples)

        If keep_original_length=True:
            output length = len(track)

        If keep_original_length=False:
            output length = len(track) + len(rir) - 1
    """
    track = np.asarray(track, dtype=np.float32)
    rir = np.asarray(rir, dtype=np.float32)

    if track.ndim != 1:
        raise ValueError(
            f"Expected mono track with shape (samples,), "
            f"but got shape {track.shape}."
        )

    if rir.ndim != 2:
        raise ValueError(
            f"Expected RIR with shape (channels, samples), "
            f"but got shape {rir.shape}."
        )

    # Validate that the RIR channel count is HOA-compatible.
    infer_hoa_order(rir.shape[0])

    original_track_len = len(track)

    convolved = []

    for ch in range(rir.shape[0]):
        y = fftconvolve(track, rir[ch], mode="full").astype(np.float32)

        if keep_original_length:
            y = y[:original_track_len]

        convolved.append(y.astype(np.float32))

    return np.stack(convolved, axis=0).astype(np.float32)


def pad_multichannel_to_length(signal, target_len):
    """
    Args:
        signal:
            Shape (channels, samples)

    Returns:
        Signal padded or cropped to target_len.
    """
    signal = np.asarray(signal, dtype=np.float32)

    if signal.shape[1] >= target_len:
        return signal[:, :target_len].astype(np.float32)

    pad_len = target_len - signal.shape[1]

    return np.pad(
        signal,
        ((0, 0), (0, pad_len)),
        mode="constant",
    ).astype(np.float32)


def normalize_peak(signal, peak_value=0.9):
    signal = np.asarray(signal, dtype=np.float32)

    max_peak = np.max(np.abs(signal))

    if max_peak > 0:
        signal = peak_value * signal / max_peak

    return signal.astype(np.float32)


# =========================
# Main dataset generation
# =========================

def generate_ambisonics_wavs(
    csv_file,
    rir_folder,
    speech_folder,
    output_folder,
    fs=24000,
    seed=0,
    hop_length=320,
    vad_frame_length=640,
    vad_top_db=30.0,
    max_start_trim_seconds=1.0,
    max_end_trim_seconds=1.0,
    random_rotation=True,
):
    rng = np.random.default_rng(seed)
    random.seed(seed)

    os.makedirs(output_folder, exist_ok=True)

    df = pd.read_csv(csv_file, nrows=30)

    df["r"] = df["Name"].str.extract(r"R(\d+)_")[0].astype(int)
    df["p"] = df["Name"].str.extract(r"_P(\d+)_")[0].astype(int)
    df["s"] = df["Name"].str.extract(r"_S(\d+)_")[0].astype(int)

    speech_files = [
        os.path.join(speech_folder, f)
        for f in os.listdir(speech_folder)
        if f.lower().endswith(".wav")
    ]

    if len(speech_files) < N_SPEAKERS:
        raise ValueError(
            f"Need at least {N_SPEAKERS} speech wav files, "
            f"but found {len(speech_files)}."
        )

    grouped = list(df.groupby(["r", "p"]))

    metadata_rows = []

    for (room, position), group in tqdm(
        grouped,
        desc="Generating ambisonics WAVs",
    ):
        group = group.sort_values("s")

        if len(group) < N_SPEAKERS:
            print(
                f"Skipping R{room}_P{position}: "
                f"expected {N_SPEAKERS} RIRs, found {len(group)}"
            )
            continue

        selected_rirs = []

        # Use S0, S1, S2
        for speaker_idx in range(N_SPEAKERS):
            rir_row = group[group["s"] == speaker_idx]

            if len(rir_row) == 0:
                print(
                    f"Skipping R{room}_P{position}: "
                    f"missing S{speaker_idx}"
                )
                selected_rirs = []
                break

            selected_rirs.append(rir_row.iloc[0])

        if len(selected_rirs) != N_SPEAKERS:
            continue

        selected_speech_files = rng.choice(
            speech_files,
            size=N_SPEAKERS,
            replace=False,
        )

        theta = 0.0

        if random_rotation:
            theta = rng.uniform(0.0, 2.0 * np.pi)

        convolved_tracks = []
        dry_track_lengths = []

        for speaker_idx in range(N_SPEAKERS):
            speech_path = selected_speech_files[speaker_idx]
            rir_row = selected_rirs[speaker_idx]

            rir_path = os.path.join(rir_folder, rir_row["Name"])

            if not os.path.exists(rir_path):
                raise FileNotFoundError(f"Missing RIR file: {rir_path}")

            speech = load_mono_speech(speech_path, fs=fs)

            speech = randomly_trim_speech_edges(
                speech=speech,
                fs=fs,
                rng=rng,
                max_start_trim_seconds=max_start_trim_seconds,
                max_end_trim_seconds=max_end_trim_seconds,
            )

            track, _ = make_repeated_speech_track(
                speech=speech,
                fs=fs,
                rng=rng,
                hop_length=hop_length,
                frame_length=vad_frame_length,
                vad_top_db=vad_top_db,
            )

            dry_track_lengths.append(len(track))

            rir = read_wav_float(rir_path, target_fs=fs)

            conv = convolve_mono_with_multichannel_rir(
                track=track,
                rir=rir,
                keep_original_length=True,
            )

            # Safety check: convolution output should now match dry track length.
            if conv.shape[1] != len(track):
                raise RuntimeError(
                    f"Unexpected convolved length for speaker {speaker_idx}: "
                    f"got {conv.shape[1]}, expected {len(track)}."
                )

            convolved_tracks.append(conv)

            metadata_rows.append(
                {
                    "output_wav": f"ambisonics_R{room}_P{position}.wav",
                    "room": room,
                    "position": position,
                    "speaker": speaker_idx,
                    "speech_file": os.path.basename(speech_path),
                    "rir_file": rir_row["Name"],
                    "dry_track_length_samples": len(track),
                    "dry_track_length_seconds": len(track) / fs,
                    "convolved_length_samples": conv.shape[1],
                    "convolved_length_seconds": conv.shape[1] / fs,
                    "rotation_rad": theta,
                    "rotation_deg": np.degrees(theta),
                    "RT": rir_row.get("RT", None),
                    "c50": rir_row.get("c50", None),
                    "x_room": rir_row.get("x_room", None),
                    "y_room": rir_row.get("y_room", None),
                    "z_room": rir_row.get("z_room", None),
                    "x_source": rir_row.get("x_source", None),
                    "y_source": rir_row.get("y_source", None),
                    "z_source": rir_row.get("z_source", None),
                    "x_mic": rir_row.get("x_mic", None),
                    "y_mic": rir_row.get("y_mic", None),
                    "z_mic": rir_row.get("z_mic", None),
                    "status": rir_row.get("status", None),
                }
            )

        num_channels = check_same_hoa_channel_count(convolved_tracks)
        hoa_order = infer_hoa_order(num_channels)

        # Final length is now based on the original dry track lengths,
        # not on track + RIR tail length.
        max_len = max(dry_track_lengths)

        convolved_tracks = [
            pad_multichannel_to_length(x, max_len)
            for x in convolved_tracks
        ]

        ambisonics = np.sum(convolved_tracks, axis=0).astype(np.float32)

        if random_rotation:
            ambisonics = rotate_hoa(ambisonics, theta)

        ambisonics = normalize_peak(
            ambisonics,
            peak_value=0.9,
        )

        # scipy writes multichannel WAVs as (samples, channels)
        ambisonics_for_wav = ambisonics.T.astype(np.float32)

        output_wav_path = os.path.join(
            output_folder,
            f"ambisonics_R{room}_P{position}.wav",
        )

        wavfile.write(
            output_wav_path,
            fs,
            ambisonics_for_wav,
        )

        print(
            f"Saved {output_wav_path} "
            f"| HOA order {hoa_order} "
            f"| channels {num_channels} "
            f"| dry max length {max_len / fs:.2f} s "
            f"| rotation {np.degrees(theta):.1f} deg"
        )

    metadata_csv = os.path.join(
        output_folder,
        "ambisonics_metadata.csv",
    )

    pd.DataFrame(metadata_rows).to_csv(
        metadata_csv,
        index=False,
    )

    print(f"Saved WAVs to: {output_folder}")
    print(f"Saved metadata to: {metadata_csv}")


# =========================
# Run
# =========================

if __name__ == "__main__":
    base_dir = "../../../data/harp/R4500_P8_S3_Order3_24kHz_shoeBox_rtProfilesMIT_ISMOrder30"

    csv_file = os.path.join(
        base_dir,
        "Generated_HOA_SRIR_data.csv",
    )

    rir_folder = os.path.join(
        base_dir,
        "RIRs",
    )

    speech_folder = "../../../data/audio_exports_Harvard_Sets"

    output_folder = "output_ambisonics_wavs"

    generate_ambisonics_wavs(
        csv_file=csv_file,
        rir_folder=rir_folder,
        speech_folder=speech_folder,
        output_folder=output_folder,
        fs=16000,
        seed=0,
        hop_length=320,
        vad_frame_length=640,
        vad_top_db=30.0,
        max_start_trim_seconds=1.0,
        max_end_trim_seconds=1.0,
        random_rotation=True,
    )