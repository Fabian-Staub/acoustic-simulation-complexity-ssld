import os
import json
import random
import numpy as np
import pandas as pd
import librosa
import torch

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


def rotate_vectors_z(vectors, theta):
    vectors = np.asarray(vectors, dtype=np.float32)

    c = np.cos(theta)
    s = np.sin(theta)

    R = np.array(
        [
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    return (R @ vectors.T).T.astype(np.float32)


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


def pad_bool_mask_to_length(mask, target_len):
    mask = np.asarray(mask, dtype=bool)

    if len(mask) >= target_len:
        return mask[:target_len]

    return np.pad(
        mask,
        (0, target_len - len(mask)),
        mode="constant",
        constant_values=False,
    )


def normalize_peak(signal, peak_value=0.9):
    signal = np.asarray(signal, dtype=np.float32)

    max_peak = np.max(np.abs(signal))

    if max_peak > 0:
        signal = peak_value * signal / max_peak

    return signal.astype(np.float32)


# =========================
# HO-PIV / ACCDOA helpers
# =========================

def get_unit_doa_from_rir_row(rir_row):
    dx = float(rir_row["x_source"]) - float(rir_row["x_mic"])
    dy = float(rir_row["y_source"]) - float(rir_row["y_mic"])
    dz = float(rir_row["z_source"]) - float(rir_row["z_mic"])

    vec = np.array([dx, dy, dz], dtype=np.float32)
    norm = np.linalg.norm(vec) + 1e-8

    return (vec / norm).astype(np.float32)


def vec_to_az_el_deg(vec):
    x, y, z = float(vec[0]), float(vec[1]), float(vec[2])

    az = np.degrees(np.arctan2(y, x))

    r = np.sqrt(x * x + y * y + z * z) + 1e-8
    el = np.degrees(np.arcsin(z / r))

    return float(az), float(el)


def make_sample_id(room, position):
    return f"R{int(room):04d}_P{int(position):02d}"


def stft_hoa_fixed_num_frames(
    hoa_signal,
    num_frames=50,
    hop_length=320,
    frame_length=640,
    n_fft=1024,
):
    """
    Custom STFT framing to force exactly 50 frames for one sample.

    Input:
        hoa_signal:
            shape (channels, samples)

    Output:
        complex STFT:
            shape (time_frames, frequency_bins, channels)
    """
    hoa_signal = np.asarray(hoa_signal, dtype=np.float32)

    if hoa_signal.ndim != 2:
        raise ValueError(
            f"Expected HOA signal with shape (channels, samples), "
            f"but got shape {hoa_signal.shape}."
        )

    num_channels, num_samples = hoa_signal.shape
    window = np.hanning(frame_length).astype(np.float32)

    stft = np.zeros(
        (num_frames, n_fft // 2 + 1, num_channels),
        dtype=np.complex64,
    )

    for frame_idx in range(num_frames):
        start = frame_idx * hop_length
        end = start + frame_length

        frame = np.zeros(
            (num_channels, frame_length),
            dtype=np.float32,
        )

        valid = max(0, min(end, num_samples) - start)

        if valid > 0:
            frame[:, :valid] = hoa_signal[:, start:start + valid]

        frame *= window[None, :]

        spec = np.fft.rfft(
            frame,
            n=n_fft,
            axis=1,
        ).astype(np.complex64)

        stft[frame_idx] = spec.T

    return stft


def compute_ho_piv_from_hoa_segment(
    hoa_segment,
    hop_length=320,
    frame_length=640,
    n_fft=1024,
    eps=1e-8,
):
    """
    Computes HO-PIV from one HOA segment.

    Formula:
        I_hat_N = -a0 * conj([a1, ..., aQ-1])

    Input:
        hoa_segment:
            shape (Q, samples), where Q = (N + 1)^2

    Output:
        ho_piv:
            shape (50, 513, 2 * (Q - 1))

            channels:
                first Q-1 -> active HO-PIV
                next Q-1  -> reactive HO-PIV
    """
    hoa_segment = np.asarray(hoa_segment, dtype=np.float32)

    if hoa_segment.ndim != 2:
        raise ValueError(
            f"Expected hoa_segment with shape (channels, samples), "
            f"got {hoa_segment.shape}."
        )

    num_channels = hoa_segment.shape[0]
    hoa_order = infer_hoa_order(num_channels)

    A = stft_hoa_fixed_num_frames(
        hoa_signal=hoa_segment,
        num_frames=50,
        hop_length=hop_length,
        frame_length=frame_length,
        n_fft=n_fft,
    )

    # A shape: (T, F, Q)
    a0 = A[:, :, 0:1]
    a_rest = A[:, :, 1:]

    I_hat = -a0 * np.conj(a_rest)

    I_active = np.real(I_hat).astype(np.float32)
    I_reactive = np.imag(I_hat).astype(np.float32)

    # Energy normalization:
    # E_N = sum_n 1/(2n+1) * sum_m |a_nm|^2
    E = np.zeros(
        A.shape[:2] + (1,),
        dtype=np.float32,
    )

    for n in range(hoa_order + 1):
        start_ch = n * n
        end_ch = (n + 1) * (n + 1)

        E += (
            np.sum(
                np.abs(A[:, :, start_ch:end_ch]) ** 2,
                axis=-1,
                keepdims=True,
            )
            / float(2 * n + 1)
        ).astype(np.float32)

    ho_piv = -np.concatenate(
        [I_active, I_reactive],
        axis=-1,
    ) / np.maximum(E, eps)

    return ho_piv.astype(np.float32)


def make_accdoa_for_segment(
    speaker_activity_masks,
    speaker_directions,
    segment_start_sample,
    num_frames=50,
    hop_length=320,
    frame_length=640,
):
    """
    Builds ACCDOA labels for one sample.

    Output:
        accdoa:
            shape (50, N_SPEAKERS, 3)

    Inactive speaker/frame = [0, 0, 0]
    Active speaker/frame   = unit DOA vector
    """
    accdoa = np.zeros(
        (num_frames, N_SPEAKERS, 3),
        dtype=np.float32,
    )

    for speaker_idx in range(N_SPEAKERS):
        mask = speaker_activity_masks[speaker_idx]
        doa = speaker_directions[speaker_idx]

        for frame_idx in range(num_frames):
            frame_start = segment_start_sample + frame_idx * hop_length
            frame_end = frame_start + frame_length

            if frame_start >= len(mask):
                continue

            active = np.any(mask[frame_start:min(frame_end, len(mask))])

            if active:
                accdoa[frame_idx, speaker_idx, :] = doa

    return accdoa



def save_hopiv_samples_for_mixture(
    ambisonics,
    speaker_activity_masks,
    speaker_directions,
    speaker_metadata,
    room,
    position,
    fs,
    theta,
    output_wav_name,
    pt_batch_name,
    hop_length=320,
    frame_length=640,
    n_fft=1024,
):
    """
    Splits one full ambisonics mixture into 1 s samples.

    New behavior:
        - does NOT save one .pt per segment
        - returns sample records so the caller can save 100 rooms in one .pt
        - returns CSV rows so the caller can save one .csv per room

    CSV format:
        one row per frame per speaker slot
        50 frames * 3 speaker slots = 150 rows per segment

    Final chunk behavior:
        - drop if < 25 frames
        - zero-pad if >= 25 frames
    """
    segment_len_samples = int(fs)
    min_keep_samples = 25 * hop_length

    total_len = ambisonics.shape[1]
    sample_id = make_sample_id(room, position)

    sample_records = []
    csv_rows = []

    start = 0
    segment_idx = 0

    while start < total_len:
        remaining = total_len - start

        if remaining < segment_len_samples:
            if remaining < min_keep_samples:
                break

            padded = True
            segment = np.zeros(
                (ambisonics.shape[0], segment_len_samples),
                dtype=np.float32,
            )
            segment[:, :remaining] = ambisonics[:, start:]
        else:
            padded = False
            segment = ambisonics[:, start:start + segment_len_samples]

        X = compute_ho_piv_from_hoa_segment(
            hoa_segment=segment,
            hop_length=hop_length,
            frame_length=frame_length,
            n_fft=n_fft,
        )

        accdoa = make_accdoa_for_segment(
            speaker_activity_masks=speaker_activity_masks,
            speaker_directions=speaker_directions,
            segment_start_sample=start,
            num_frames=50,
            hop_length=hop_length,
            frame_length=frame_length,
        )

        base_name = f"{sample_id}_segment{segment_idx:04d}"

        sample_record = {
            "X": torch.from_numpy(X),              # (50, 513, 2*(Q-1))
            "accdoa": torch.from_numpy(accdoa),    # (50, 3, 3)

            "sample_id": sample_id,
            "segment_idx": int(segment_idx),
            "base_name": base_name,

            "sample_rate": int(fs),
            "feature": "HO-PIV",

            "frame_hop_samples": int(hop_length),
            "frame_length_samples": int(frame_length),
            "n_fft": int(n_fft),

            "segment_start_sample": int(start),
            "segment_start_s": float(start / fs),

            "num_valid_samples": int(min(remaining, segment_len_samples)),
            "padded": bool(padded),

            "rotation_rad": float(theta),
            "rotation_deg": float(np.degrees(theta)),

            "room": int(room),
            "position": int(position),

            "output_wav": output_wav_name,
            "pt_batch_file": pt_batch_name,
        }
        sample_records.append(sample_record)

        for frame_idx in range(50):
            frame_start_sample = start + frame_idx * hop_length
            frame_start_s = frame_start_sample / fs

            for slot in range(N_SPEAKERS):
                meta = speaker_metadata[slot]
                rir_meta = meta["rir_metadata"]

                dx = float(speaker_directions[slot][0])
                dy = float(speaker_directions[slot][1])
                dz = float(speaker_directions[slot][2])

                active = int(np.linalg.norm(accdoa[frame_idx, slot, :]) > 0.5)

                accdoa_x = float(accdoa[frame_idx, slot, 0])
                accdoa_y = float(accdoa[frame_idx, slot, 1])
                accdoa_z = float(accdoa[frame_idx, slot, 2])

                az_deg, el_deg = vec_to_az_el_deg([dx, dy, dz])

                csv_rows.append(
                    {
                        "pt_batch_file": pt_batch_name,
                        "sample_id": sample_id,
                        "segment_idx": int(segment_idx),
                        "base_name": base_name,

                        "segment_start_sample": int(start),
                        "segment_start_s": float(start / fs),

                        "frame_idx": int(frame_idx),
                        "frame_start_sample": int(frame_start_sample),
                        "frame_start_s": float(frame_start_s),

                        "slot": int(slot),
                        "speech_path": meta["speech_path"],

                        "active": int(active),

                        "accdoa_x": accdoa_x,
                        "accdoa_y": accdoa_y,
                        "accdoa_z": accdoa_z,

                        "dx": dx,
                        "dy": dy,
                        "dz": dz,

                        "az_deg": round(float(az_deg), 3),
                        "el_deg": round(float(el_deg), 3),

                        "pause_start_s": float(meta.get("pause_start_s", 0.0)),
                        "pause_middle_s": float(meta.get("pause_middle_s", 0.0)),
                        "pause_end_s": float(meta.get("pause_end_s", 0.0)),

                        "Name": rir_meta.get("Name", meta["rir_file"]),
                        "RT": rir_meta.get("RT", None),
                        "c50": rir_meta.get("c50", None),

                        "x_room": rir_meta.get("x_room", None),
                        "y_room": rir_meta.get("y_room", None),
                        "z_room": rir_meta.get("z_room", None),

                        "x_source": rir_meta.get("x_source", None),
                        "y_source": rir_meta.get("y_source", None),
                        "z_source": rir_meta.get("z_source", None),

                        "x_mic": rir_meta.get("x_mic", None),
                        "y_mic": rir_meta.get("y_mic", None),
                        "z_mic": rir_meta.get("z_mic", None),

                        "status": rir_meta.get("status", None),
                    }
                )

        segment_idx += 1
        start += segment_len_samples

    return sample_records, csv_rows


def save_room_batch_pt(pt_batch_folder, batch_idx, room_records):
    """
    Saves exactly one .pt for a batch of up to 100 rooms.
    Full batches contain 100 rooms. The final batch may contain fewer.
    """
    if len(room_records) == 0:
        return None

    os.makedirs(pt_batch_folder, exist_ok=True)

    rooms = [int(room_record["room"]) for room_record in room_records]
    batch_name = f"rooms_batch{batch_idx:04d}.pt"
    batch_path = os.path.join(pt_batch_folder, batch_name)

    torch.save(
        {
            "batch_idx": int(batch_idx),
            "num_rooms": int(len(room_records)),
            "rooms": rooms,
            "samples": room_records,
        },
        batch_path,
    )

    return batch_name


# =========================
# Main dataset generation
# =========================

def generate_ambisonics_wavs(
    csv_file,
    rir_folder,
    speech_folder,
    output_folder,
    fs=16000,
    seed=0,
    hop_length=320,
    vad_frame_length=640,
    vad_top_db=30.0,
    max_start_trim_seconds=1.0,
    max_end_trim_seconds=1.0,
    random_rotation=True,
    n_fft=1024,
    nrows=30,
    write_wavs=True,
    rooms_per_pt=100,
):
    rng = np.random.default_rng(seed)
    random.seed(seed)

    os.makedirs(output_folder, exist_ok=True)

    wav_folder = os.path.join(output_folder, "wav")
    room_csv_folder = os.path.join(output_folder, "csv")
    pt_batch_folder = os.path.join(output_folder, "pt")

    os.makedirs(wav_folder, exist_ok=True)
    os.makedirs(room_csv_folder, exist_ok=True)
    os.makedirs(pt_batch_folder, exist_ok=True)

    df = pd.read_csv(csv_file, nrows=nrows)

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

    rooms = sorted(df["r"].unique())

    metadata_rows = []
    current_batch_room_records = []
    current_batch_room_count = 0
    batch_idx = 0

    for room in tqdm(
        rooms,
        desc="Generating ambisonics WAVs, room CSVs, and 100-room .pt batches",
    ):
        room_group = df[df["r"] == room]
        positions = sorted(room_group["p"].unique())

        pt_batch_name = f"rooms_batch{batch_idx:04d}.pt"
        room_sample_records = []
        room_csv_rows = []

        for position in positions:
            group = room_group[room_group["p"] == position].sort_values("s")

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

            speaker_activity_masks = []
            speaker_directions = []
            speaker_metadata = []

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

                track, sample_mask = make_repeated_speech_track(
                    speech=speech,
                    fs=fs,
                    rng=rng,
                    hop_length=hop_length,
                    frame_length=vad_frame_length,
                    vad_top_db=vad_top_db,
                )

                dry_track_lengths.append(len(track))
                speaker_activity_masks.append(sample_mask)
                speaker_directions.append(get_unit_doa_from_rir_row(rir_row))

                speaker_metadata.append(
                    {
                        "speech_path": speech_path,
                        "speech_file": os.path.basename(speech_path),
                        "rir_file": rir_row["Name"],

                        "pause_start_s": 0.0,
                        "pause_middle_s": 0.0,
                        "pause_end_s": 0.0,

                        "rir_metadata": {
                            "Name": rir_row.get("Name", None),
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
                        },
                    }
                )

                rir = read_wav_float(rir_path, target_fs=fs)

                conv = convolve_mono_with_multichannel_rir(
                    track=track,
                    rir=rir,
                    keep_original_length=True,
                )

                if conv.shape[1] != len(track):
                    raise RuntimeError(
                        f"Unexpected convolved length for speaker {speaker_idx}: "
                        f"got {conv.shape[1]}, expected {len(track)}."
                    )

                convolved_tracks.append(conv)

            num_channels = check_same_hoa_channel_count(convolved_tracks)
            hoa_order = infer_hoa_order(num_channels)

            max_len = max(dry_track_lengths)

            convolved_tracks = [
                pad_multichannel_to_length(x, max_len)
                for x in convolved_tracks
            ]

            speaker_activity_masks = [
                pad_bool_mask_to_length(mask, max_len)
                for mask in speaker_activity_masks
            ]

            speaker_directions = np.asarray(
                speaker_directions,
                dtype=np.float32,
            )

            ambisonics = np.sum(convolved_tracks, axis=0).astype(np.float32)

            if random_rotation:
                ambisonics = rotate_hoa(ambisonics, theta)
                speaker_directions = rotate_vectors_z(
                    speaker_directions,
                    theta,
                )

            ambisonics = normalize_peak(
                ambisonics,
                peak_value=0.9,
            )

            output_wav_name = f"ambisonics_R{room}_P{position}.wav"
            output_wav_path = os.path.join(wav_folder, output_wav_name)

            if write_wavs:
                wavfile.write(
                    output_wav_path,
                    fs,
                    ambisonics.T.astype(np.float32),
                )

            sample_records, csv_rows = save_hopiv_samples_for_mixture(
                ambisonics=ambisonics,
                speaker_activity_masks=speaker_activity_masks,
                speaker_directions=speaker_directions,
                speaker_metadata=speaker_metadata,
                room=room,
                position=position,
                fs=fs,
                theta=theta,
                output_wav_name=output_wav_name,
                pt_batch_name=pt_batch_name,
                hop_length=hop_length,
                frame_length=vad_frame_length,
                n_fft=n_fft,
            )

            room_sample_records.extend(sample_records)
            room_csv_rows.extend(csv_rows)

            print(
                f"Prepared {output_wav_path} "
                f"| HOA order {hoa_order} "
                f"| channels {num_channels} "
                f"| dry max length {max_len / fs:.2f} s "
                f"| rotation {np.degrees(theta):.1f} deg "
                f"| samples {len(sample_records)}"
            )

        if len(room_sample_records) == 0:
            print(f"Skipping room R{room}: no valid samples")
            continue

        # One CSV per room.
        room_csv_name = f"room_R{int(room):04d}.csv"
        room_csv_path = os.path.join(room_csv_folder, room_csv_name)
        pd.DataFrame(room_csv_rows).to_csv(room_csv_path, index=False)

        room_record = {
            "room": int(room),
            "room_csv": room_csv_name,
            "num_samples": int(len(room_sample_records)),
            "samples": room_sample_records,
        }

        current_batch_room_records.append(room_record)
        current_batch_room_count += 1

        for row in room_csv_rows:
            metadata_rows.append(
                {
                    "room_csv": room_csv_name,
                    **row,
                }
            )

        # Save exactly 100 rooms per .pt for full batches.
        if current_batch_room_count == rooms_per_pt:
            final_batch_name = save_room_batch_pt(
                pt_batch_folder=pt_batch_folder,
                batch_idx=batch_idx,
                room_records=current_batch_room_records,
            )
            print(
                f"Saved .pt batch: {final_batch_name} "
                f"with {current_batch_room_count} rooms"
            )

            current_batch_room_records = []
            current_batch_room_count = 0
            batch_idx += 1

    # Save final partial batch, if any.
    if current_batch_room_count > 0:
        final_batch_name = save_room_batch_pt(
            pt_batch_folder=pt_batch_folder,
            batch_idx=batch_idx,
            room_records=current_batch_room_records,
        )
        print(
            f"Saved final .pt batch: {final_batch_name} "
            f"with {current_batch_room_count} rooms"
        )

    metadata_csv = os.path.join(
        output_folder,
        "hopiv_accdoa_metadata_all_frames.csv",
    )

    pd.DataFrame(metadata_rows).to_csv(
        metadata_csv,
        index=False,
    )

    print(f"Saved WAVs to: {wav_folder}")
    print(f"Saved room CSVs to: {room_csv_folder}")
    print(f"Saved 100-room .pt batches to: {pt_batch_folder}")
    print(f"Saved master metadata CSV to: {metadata_csv}")


# =========================
# Run
# =========================

if __name__ == "__main__":
    base_dir = "../../data/harp/R4500_P8_S3_Order3_24kHz_shoeBox_rtProfilesMIT_ISMOrder30"

    csv_file = os.path.join(
        base_dir,
        "Generated_HOA_SRIR_data.csv",
    )

    rir_folder = os.path.join(
        base_dir,
        "RIRs",
    )

    speech_folder = "../../data/audio_exports_Harvard_Sets"

    output_folder = "../../data/harp/R4500_P8_S3_Order3_24kHz_shoeBox_rtProfilesMIT_ISMOrder30/hopiv_accdoa"

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
        n_fft=1024,
        nrows=30,
        write_wavs=True,
        rooms_per_pt=100,
    )