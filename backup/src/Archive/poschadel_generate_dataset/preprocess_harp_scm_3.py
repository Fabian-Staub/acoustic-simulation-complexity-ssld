# Corrected: HO-PIV + per-HOA-channel log-mel spectrograms,
# using a true mel axis, e.g. n_mels=128.
#
# Main change:
#   - HO-PIV is first computed on the STFT frequency bins: (50, 513, 2*(Q-1))
#   - HO-PIV is then mel-binned to:                      (50, n_mels, 2*(Q-1))
#   - Per-HOA-channel log-mel is computed as:             (50, n_mels, Q)
#   - Final feature is:                                   (50, n_mels, 2*(Q-1)+Q)
#
# This avoids librosa's "Empty filters detected" warning caused by n_mels=513
# at fs=16 kHz / n_fft=1024.

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

    track = np.concatenate([speech, speech]).astype(np.float32)
    sample_mask = np.concatenate([speech_vad_mask, speech_vad_mask])

    return track, sample_mask


# =========================
# HOA helpers
# =========================

def infer_hoa_order(num_channels):
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
        speech = speech / peak

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


def estimate_rir_delay_samples(rir, threshold_db=-40.0):
    rir = np.asarray(rir, dtype=np.float32)

    if rir.ndim != 2 or rir.shape[1] == 0:
        return 0

    envelope = np.sqrt(np.mean(rir ** 2, axis=0))
    peak = float(np.max(envelope))

    if peak <= 0.0:
        return 0

    threshold = peak * (10.0 ** (float(threshold_db) / 20.0))
    above = np.where(envelope >= threshold)[0]

    if len(above) == 0:
        return 0

    return int(above[0])


def get_rt60_seconds_from_rir_row(rir_row, default_rt60=0.0):
    for key in ("RT60", "rt60", "RT", "rt"):
        if key in rir_row:
            value = rir_row.get(key)

            try:
                value = float(value)
            except (TypeError, ValueError):
                continue

            if np.isfinite(value) and value >= 0.0:
                return value

    return float(default_rt60)


def delay_and_extend_activity_mask(
    mask,
    delay_samples,
    extension_samples,
    target_len=None,
):
    mask = np.asarray(mask, dtype=bool)

    if target_len is None:
        target_len = len(mask)

    target_len = int(target_len)
    delay_samples = max(0, int(delay_samples))
    extension_samples = max(0, int(extension_samples))

    corrected = np.zeros(target_len, dtype=bool)

    if target_len <= 0 or len(mask) == 0 or delay_samples >= target_len:
        return corrected

    padded = np.concatenate([[False], mask, [False]])
    changes = np.diff(padded.astype(np.int8))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    for start, end in zip(starts, ends):
        corrected_start = start + delay_samples
        corrected_end = end + delay_samples + extension_samples

        corrected_start = max(0, min(corrected_start, target_len))
        corrected_end = max(0, min(corrected_end, target_len))

        if corrected_end > corrected_start:
            corrected[corrected_start:corrected_end] = True

    return corrected


def convolve_mono_with_multichannel_rir(
    track,
    rir,
    keep_original_length=True,
):
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


def normalize_peak(signal, peak_value=1.0):
    signal = np.asarray(signal, dtype=np.float32)

    max_peak = np.max(np.abs(signal))

    if max_peak > 0:
        signal = peak_value * signal / max_peak

    return signal.astype(np.float32)


# =========================
# Feature standardization helpers
# =========================

class FeatureStatsAccumulator:
    """
    Accumulates dataset-level statistics for mel-binned HO-PIV + log-mel.

    Feature layout:
        [0 : 2*(Q-1)]          -> mel-binned HO-PIV active + reactive
        [2*(Q-1) : 2*(Q-1)+Q]  -> log-mel per HOA channel

    Standardization:
        HO-PIV:
            one scalar mean/std jointly across time, mel bins, active/reactive,
            and all HO-PIV channels.

        Log-mel:
            one mean/std per mel bin, shared across all HOA channels.
    """

    def __init__(self, num_hoa_channels, n_mels, eps=1e-8):
        self.num_hoa_channels = int(num_hoa_channels)
        self.hopiv_channels = int(2 * (self.num_hoa_channels - 1))
        self.logmel_channels = int(self.num_hoa_channels)
        self.n_mels = int(n_mels)
        self.eps = float(eps)

        self.hopiv_sum = 0.0
        self.hopiv_sumsq = 0.0
        self.hopiv_count = 0

        self.logmel_sum = np.zeros(self.n_mels, dtype=np.float64)
        self.logmel_sumsq = np.zeros(self.n_mels, dtype=np.float64)
        self.logmel_count = np.zeros(self.n_mels, dtype=np.int64)

    def update(self, X):
        X = np.asarray(X, dtype=np.float32)

        if X.ndim != 3:
            raise ValueError(f"Expected X with shape (T, M, C), got {X.shape}.")

        if X.shape[1] != self.n_mels:
            raise ValueError(
                f"Expected {self.n_mels} mel bins, got {X.shape[1]}."
            )

        expected_channels = self.hopiv_channels + self.logmel_channels

        if X.shape[2] != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} feature channels, got {X.shape[2]}."
            )

        hopiv = X[:, :, :self.hopiv_channels].astype(np.float64)

        self.hopiv_sum += float(np.sum(hopiv))
        self.hopiv_sumsq += float(np.sum(hopiv ** 2))
        self.hopiv_count += int(hopiv.size)

        logmel = X[:, :, self.hopiv_channels:].astype(np.float64)

        self.logmel_sum += np.sum(logmel, axis=(0, 2))
        self.logmel_sumsq += np.sum(logmel ** 2, axis=(0, 2))
        self.logmel_count += logmel.shape[0] * logmel.shape[2]

    def finalize(self):
        if self.hopiv_count <= 0:
            raise RuntimeError("No HO-PIV samples were accumulated.")

        if np.any(self.logmel_count <= 0):
            raise RuntimeError("No log-mel samples were accumulated.")

        hopiv_mean = self.hopiv_sum / self.hopiv_count
        hopiv_var = self.hopiv_sumsq / self.hopiv_count - hopiv_mean ** 2
        hopiv_std = np.sqrt(max(hopiv_var, self.eps))

        logmel_mean = self.logmel_sum / self.logmel_count
        logmel_var = self.logmel_sumsq / self.logmel_count - logmel_mean ** 2
        logmel_std = np.sqrt(np.maximum(logmel_var, self.eps))

        return {
            "num_hoa_channels": self.num_hoa_channels,
            "hopiv_channels": self.hopiv_channels,
            "logmel_channels": self.logmel_channels,
            "n_mels": self.n_mels,
            "eps": self.eps,
            "hopiv_mean": float(hopiv_mean),
            "hopiv_std": float(hopiv_std),
            "logmel_mean": logmel_mean.astype(float).tolist(),
            "logmel_std": logmel_std.astype(float).tolist(),
            "description": {
                "hopiv": "single joint z-score over all mel-binned HO-PIV values",
                "logmel": "mel-bin-dependent z-score shared across HOA channels",
            },
        }


class FeatureNormalizer:
    def __init__(self, stats):
        self.stats = stats

        self.num_hoa_channels = int(stats["num_hoa_channels"])
        self.hopiv_channels = int(stats["hopiv_channels"])
        self.logmel_channels = int(stats["logmel_channels"])
        self.n_mels = int(stats["n_mels"])
        self.eps = float(stats.get("eps", 1e-8))

        self.hopiv_mean = np.float32(stats["hopiv_mean"])
        self.hopiv_std = np.float32(max(float(stats["hopiv_std"]), self.eps))

        self.logmel_mean = np.asarray(stats["logmel_mean"], dtype=np.float32)
        self.logmel_std = np.asarray(stats["logmel_std"], dtype=np.float32)
        self.logmel_std = np.maximum(self.logmel_std, self.eps).astype(np.float32)

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32).copy()

        expected_channels = self.hopiv_channels + self.logmel_channels

        if X.ndim != 3:
            raise ValueError(f"Expected X with shape (T, M, C), got {X.shape}.")

        if X.shape[1] != self.n_mels:
            raise ValueError(
                f"Expected {self.n_mels} mel bins, got {X.shape[1]}."
            )

        if X.shape[2] != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} feature channels, got {X.shape[2]}."
            )

        X[:, :, :self.hopiv_channels] = (
            X[:, :, :self.hopiv_channels] - self.hopiv_mean
        ) / self.hopiv_std

        X[:, :, self.hopiv_channels:] = (
            X[:, :, self.hopiv_channels:] - self.logmel_mean[None, :, None]
        ) / self.logmel_std[None, :, None]

        return X.astype(np.float32)


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
    Custom STFT framing to force exactly 50 frames.

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


def make_mel_basis(
    fs,
    n_fft=1024,
    n_mels=128,
    fmin=0.0,
    fmax=None,
):
    """
    Builds a mel basis for mapping STFT-frequency-bin features to mel bins.

    Returns:
        mel_basis:
            shape (n_mels, n_fft//2 + 1)
    """
    if fmax is None:
        fmax = fs / 2.0

    mel_basis = librosa.filters.mel(
        sr=fs,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        norm=None,
        dtype=np.float32,
    )

    empty = np.where(np.sum(mel_basis, axis=1) <= 0.0)[0]

    if len(empty) > 0:
        raise ValueError(
            f"Mel basis contains {len(empty)} empty filters. "
            f"Reduce n_mels, increase n_fft, or increase fs/fmax. "
            f"Current: fs={fs}, n_fft={n_fft}, n_mels={n_mels}, fmax={fmax}."
        )

    return mel_basis.astype(np.float32)


def compute_hopiv_logmel_from_hoa_segment(
    hoa_segment,
    fs,
    hop_length=320,
    frame_length=640,
    n_fft=1024,
    n_mels=128,
    eps=1e-8,
):
    """
    Computes mel-binned HO-PIV and per-HOA-channel log-mel spectrograms.

    Input:
        hoa_segment:
            Shape (Q, samples), where Q = (N + 1)^2.

    Output:
        X:
            Shape (50, n_mels, 2 * (Q - 1) + Q)

            channels/features:
                [0 : Q-1]                 -> active HO-PIV, mel-binned
                [Q-1 : 2*(Q-1)]           -> reactive HO-PIV, mel-binned
                [2*(Q-1) : 2*(Q-1)+Q]     -> log-mel per HOA channel

    Notes:
        HO-PIV itself is still computed using the original STFT bins and the
        original HO-PIV energy normalization E_N(k,l). After that, the
        frequency-bin HO-PIV feature is projected onto the mel axis so it can
        be concatenated with log-mel on the same second dimension.
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

    ho_piv_freq = -np.concatenate(
        [I_active, I_reactive],
        axis=-1,
    ) / np.maximum(E, eps)

    # ho_piv_freq shape: (T, F, 2*(Q-1))

    mel_basis = make_mel_basis(
        fs=fs,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=0.0,
        fmax=fs / 2.0,
    )

    # Mel-bin HO-PIV:
    # mel_basis:   (M, F)
    # ho_piv_freq: (T, F, C)
    # output:      (T, M, C)
    #
    # Because HO-PIV can be signed, we normalize the mel filters by their
    # row sums before applying them to avoid scale inflation.
    mel_basis_norm = mel_basis / np.maximum(
        np.sum(mel_basis, axis=1, keepdims=True),
        eps,
    )

    ho_piv_mel = np.einsum(
        "mf,tfc->tmc",
        mel_basis_norm,
        ho_piv_freq,
        optimize=True,
    ).astype(np.float32)

    # Per-HOA-channel log-mel power:
    # power:       (T, F, Q)
    # mel_basis:   (M, F)
    # mel_power:   (T, M, Q)
    power = (np.abs(A) ** 2).astype(np.float32)

    mel_power = np.einsum(
        "mf,tfq->tmq",
        mel_basis,
        power,
        optimize=True,
    ).astype(np.float32)

    logmel = np.log(np.maximum(mel_power, eps)).astype(np.float32)

    X = np.concatenate(
        [ho_piv_mel, logmel],
        axis=-1,
    ).astype(np.float32)

    return X


def make_accdoa_for_segment(
    speaker_activity_masks,
    speaker_directions,
    segment_start_sample,
    num_frames=50,
    hop_length=320,
    frame_length=640,
):
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
    n_mels=128,
    stats_accumulator=None,
    feature_normalizer=None,
    make_records=True,
):
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

        X = compute_hopiv_logmel_from_hoa_segment(
            hoa_segment=segment,
            fs=fs,
            hop_length=hop_length,
            frame_length=frame_length,
            n_fft=n_fft,
            n_mels=n_mels,
        )

        if stats_accumulator is not None:
            stats_accumulator.update(X)

        if not make_records:
            segment_idx += 1
            start += segment_len_samples
            continue

        if feature_normalizer is not None:
            X = feature_normalizer.transform(X)

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
            "X": torch.from_numpy(X),              # (50, n_mels, 2*(Q-1)+Q)
            "accdoa": torch.from_numpy(accdoa),    # (50, 3, 3)

            "sample_id": sample_id,
            "segment_idx": int(segment_idx),
            "base_name": base_name,

            "sample_rate": int(fs),
            "feature": "mel-HO-PIV+logmel_zscore",
            "num_mel_bins": int(n_mels),
            "num_feature_channels": int(X.shape[-1]),

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

                        "rir_delay_samples": int(meta.get("rir_delay_samples", 0)),
                        "rir_delay_s": float(meta.get("rir_delay_s", 0.0)),
                        "rt60_s": float(meta.get("rt60_s", 0.0)),
                        "vad_extension_samples": int(meta.get("vad_extension_samples", 0)),
                        "vad_extension_s": float(meta.get("vad_extension_s", 0.0)),

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
# One room/position synthesis helper
# =========================

def synthesize_one_position(
    group,
    room,
    position,
    rir_folder,
    speech_files,
    fs,
    rng,
    hop_length,
    vad_frame_length,
    vad_top_db,
    max_start_trim_seconds,
    max_end_trim_seconds,
    random_rotation,
    correct_vad_for_rir,
    rir_delay_threshold_db,
    rt60_extension_fraction,
):
    if len(group) < N_SPEAKERS:
        print(
            f"Skipping R{room}_P{position}: "
            f"expected {N_SPEAKERS} RIRs, found {len(group)}"
        )
        return None

    selected_rirs = []

    for speaker_idx in range(N_SPEAKERS):
        rir_row = group[group["s"] == speaker_idx]

        if len(rir_row) == 0:
            print(
                f"Skipping R{room}_P{position}: "
                f"missing S{speaker_idx}"
            )
            return None

        selected_rirs.append(rir_row.iloc[0])

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

        rir = read_wav_float(rir_path, target_fs=fs)

        rir_delay_samples = 0
        rir_delay_s = 0.0
        rt60_s = get_rt60_seconds_from_rir_row(rir_row)
        vad_extension_samples = 0
        vad_extension_s = 0.0

        if correct_vad_for_rir:
            rir_delay_samples = estimate_rir_delay_samples(
                rir=rir,
                threshold_db=rir_delay_threshold_db,
            )
            rir_delay_s = rir_delay_samples / float(fs)

            vad_extension_s = float(rt60_s) * float(rt60_extension_fraction)
            vad_extension_samples = int(round(vad_extension_s * fs))

            sample_mask = delay_and_extend_activity_mask(
                mask=sample_mask,
                delay_samples=rir_delay_samples,
                extension_samples=vad_extension_samples,
                target_len=len(track),
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

                "rir_delay_samples": int(rir_delay_samples),
                "rir_delay_s": float(rir_delay_s),
                "rt60_s": float(rt60_s),
                "vad_extension_samples": int(vad_extension_samples),
                "vad_extension_s": float(vad_extension_s),

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
        peak_value=1.0,
    )

    return {
        "ambisonics": ambisonics,
        "speaker_activity_masks": speaker_activity_masks,
        "speaker_directions": speaker_directions,
        "speaker_metadata": speaker_metadata,
        "theta": theta,
        "num_channels": num_channels,
        "hoa_order": hoa_order,
        "max_len": max_len,
    }


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
    n_mels=128,
    nrows=30,
    write_wavs=True,
    rooms_per_pt=100,
    correct_vad_for_rir=True,
    rir_delay_threshold_db=-40.0,
    rt60_extension_fraction=1.0 / 6.0,
):
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

    # Quick mel-basis validation before doing expensive generation.
    _ = make_mel_basis(
        fs=fs,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=0.0,
        fmax=fs / 2.0,
    )

    first_num_channels = None
    stats_accumulator = None

    # =========================
    # Pass 1: compute z-score stats
    # =========================

    print("Pass 1/2: computing feature normalization statistics...")

    rng = np.random.default_rng(seed)
    random.seed(seed)

    for room in tqdm(rooms, desc="Stats pass"):
        room_group = df[df["r"] == room]
        positions = sorted(room_group["p"].unique())

        for position in positions:
            group = room_group[room_group["p"] == position].sort_values("s")

            synthesized = synthesize_one_position(
                group=group,
                room=room,
                position=position,
                rir_folder=rir_folder,
                speech_files=speech_files,
                fs=fs,
                rng=rng,
                hop_length=hop_length,
                vad_frame_length=vad_frame_length,
                vad_top_db=vad_top_db,
                max_start_trim_seconds=max_start_trim_seconds,
                max_end_trim_seconds=max_end_trim_seconds,
                random_rotation=random_rotation,
                correct_vad_for_rir=correct_vad_for_rir,
                rir_delay_threshold_db=rir_delay_threshold_db,
                rt60_extension_fraction=rt60_extension_fraction,
            )

            if synthesized is None:
                continue

            num_channels = synthesized["num_channels"]

            if first_num_channels is None:
                first_num_channels = num_channels
                stats_accumulator = FeatureStatsAccumulator(
                    num_hoa_channels=first_num_channels,
                    n_mels=n_mels,
                    eps=1e-8,
                )
            elif num_channels != first_num_channels:
                raise ValueError(
                    f"Inconsistent HOA channel count. "
                    f"First={first_num_channels}, current={num_channels}."
                )

            output_wav_name = f"ambisonics_R{room}_P{position}.wav"
            pt_batch_name = "stats_pass_no_batch.pt"

            save_hopiv_samples_for_mixture(
                ambisonics=synthesized["ambisonics"],
                speaker_activity_masks=synthesized["speaker_activity_masks"],
                speaker_directions=synthesized["speaker_directions"],
                speaker_metadata=synthesized["speaker_metadata"],
                room=room,
                position=position,
                fs=fs,
                theta=synthesized["theta"],
                output_wav_name=output_wav_name,
                pt_batch_name=pt_batch_name,
                hop_length=hop_length,
                frame_length=vad_frame_length,
                n_fft=n_fft,
                n_mels=n_mels,
                stats_accumulator=stats_accumulator,
                feature_normalizer=None,
                make_records=False,
            )

    if stats_accumulator is None:
        raise RuntimeError("No valid samples were generated; cannot compute stats.")

    stats = stats_accumulator.finalize()
    stats_path = os.path.join(output_folder, "feature_normalization_stats.json")

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved feature normalization stats to: {stats_path}")
    print(f"HO-PIV mean: {stats['hopiv_mean']:.6g}")
    print(f"HO-PIV std:  {stats['hopiv_std']:.6g}")

    feature_normalizer = FeatureNormalizer(stats)

    # =========================
    # Pass 2: save normalized dataset
    # =========================

    print("Pass 2/2: generating normalized features and saving outputs...")

    rng = np.random.default_rng(seed)
    random.seed(seed)

    metadata_rows = []
    current_batch_room_records = []
    current_batch_room_count = 0
    batch_idx = 0

    for room in tqdm(
        rooms,
        desc="Generating ambisonics WAVs, room CSVs, and normalized .pt batches",
    ):
        room_group = df[df["r"] == room]
        positions = sorted(room_group["p"].unique())

        pt_batch_name = f"rooms_batch{batch_idx:04d}.pt"
        room_sample_records = []
        room_csv_rows = []

        for position in positions:
            group = room_group[room_group["p"] == position].sort_values("s")

            synthesized = synthesize_one_position(
                group=group,
                room=room,
                position=position,
                rir_folder=rir_folder,
                speech_files=speech_files,
                fs=fs,
                rng=rng,
                hop_length=hop_length,
                vad_frame_length=vad_frame_length,
                vad_top_db=vad_top_db,
                max_start_trim_seconds=max_start_trim_seconds,
                max_end_trim_seconds=max_end_trim_seconds,
                random_rotation=random_rotation,
                correct_vad_for_rir=correct_vad_for_rir,
                rir_delay_threshold_db=rir_delay_threshold_db,
                rt60_extension_fraction=rt60_extension_fraction,
            )

            if synthesized is None:
                continue

            num_channels = synthesized["num_channels"]
            hoa_order = synthesized["hoa_order"]
            max_len = synthesized["max_len"]
            theta = synthesized["theta"]
            ambisonics = synthesized["ambisonics"]

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
                speaker_activity_masks=synthesized["speaker_activity_masks"],
                speaker_directions=synthesized["speaker_directions"],
                speaker_metadata=synthesized["speaker_metadata"],
                room=room,
                position=position,
                fs=fs,
                theta=theta,
                output_wav_name=output_wav_name,
                pt_batch_name=pt_batch_name,
                hop_length=hop_length,
                frame_length=vad_frame_length,
                n_fft=n_fft,
                n_mels=n_mels,
                stats_accumulator=None,
                feature_normalizer=feature_normalizer,
                make_records=True,
            )

            room_sample_records.extend(sample_records)
            room_csv_rows.extend(csv_rows)

            print(
                f"Prepared {output_wav_path} "
                f"| HOA order {hoa_order} "
                f"| channels {num_channels} "
                f"| dry max length {max_len / fs:.2f} s "
                f"| rotation {np.degrees(theta):.1f} deg "
                f"| normalized samples {len(sample_records)} "
                f"| X shape {tuple(sample_records[0]['X'].shape) if sample_records else None}"
            )

        if len(room_sample_records) == 0:
            print(f"Skipping room R{room}: no valid samples")
            continue

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

        if current_batch_room_count == rooms_per_pt:
            final_batch_name = save_room_batch_pt(
                pt_batch_folder=pt_batch_folder,
                batch_idx=batch_idx,
                room_records=current_batch_room_records,
            )
            print(
                f"Saved normalized .pt batch: {final_batch_name} "
                f"with {current_batch_room_count} rooms"
            )

            current_batch_room_records = []
            current_batch_room_count = 0
            batch_idx += 1

    if current_batch_room_count > 0:
        final_batch_name = save_room_batch_pt(
            pt_batch_folder=pt_batch_folder,
            batch_idx=batch_idx,
            room_records=current_batch_room_records,
        )
        print(
            f"Saved final normalized .pt batch: {final_batch_name} "
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
    print(f"Saved normalized .pt batches to: {pt_batch_folder}")
    print(f"Saved master metadata CSV to: {metadata_csv}")
    print(f"Saved normalization stats to: {stats_path}")


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

    output_folder = os.path.join(
        base_dir,
        "hopiv_accdoa_mel128",
    )

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
        n_mels=128,
        rooms_per_pt=100,
        correct_vad_for_rir=True,
        rir_delay_threshold_db=-40.0,
        rt60_extension_fraction=1.0 / 6.0,
        nrows=300,
        write_wavs=False,
    )