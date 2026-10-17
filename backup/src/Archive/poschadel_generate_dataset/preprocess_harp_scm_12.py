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
import glob
import argparse
import random
from functools import lru_cache
import re

import numpy as np
import pandas as pd
import librosa
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.io import wavfile
from scipy.fft import next_fast_len, rfft, irfft
from spharpy.transforms import rotation_z_axis_real


# =========================
# Fixed configuration
# =========================

N_SPEAKERS = 3

# .pt output format:
#     (X, accdoa)
# where
#     X.shape      == (N, 50, n_mels, feature_channels)
#     accdoa.shape == (N, 50, 3, 3)
# No metadata dictionaries are stored in .pt files.


# =========================
# VAD / repeated speech track
# =========================

def get_timit_phn_path(wav_path):
    """Return the TIMIT .PHN path corresponding to a *.WAV.wav file."""
    wav_path = os.fspath(wav_path)
    suffix = ".WAV.wav"
    if not wav_path.lower().endswith(suffix.lower()):
        raise ValueError(f"Expected a TIMIT file ending in {suffix!r}: {wav_path}")
    return wav_path[:-len(suffix)] + ".PHN"


@lru_cache(maxsize=256)
def _load_timit_speech_and_activity_cached(wav_path, fs):
    """
    Load a mono 16-kHz TIMIT utterance and derive sample-level activity from
    its companion .PHN file. TIMIT interval end indices are exclusive.
    """
    fs_in, speech = wavfile.read(wav_path)
    if int(fs_in) != int(fs):
        raise ValueError(
            f"TIMIT speech must be {fs} Hz, but {wav_path} is {fs_in} Hz."
        )

    speech = np.asarray(speech)
    if speech.ndim != 1:
        raise ValueError(f"Expected mono TIMIT audio, got shape {speech.shape}.")

    if speech.dtype.kind in {"i", "u"}:
        if speech.dtype.kind == "u":
            info = np.iinfo(speech.dtype)
            midpoint = (info.max + 1) / 2.0
            speech = (speech.astype(np.float32) - midpoint) / midpoint
        else:
            scale = float(max(abs(np.iinfo(speech.dtype).min), np.iinfo(speech.dtype).max))
            speech = speech.astype(np.float32) / scale
    else:
        speech = speech.astype(np.float32, copy=False)

    peak = float(np.max(np.abs(speech))) if speech.size else 0.0
    if peak > 0.0:
        speech = speech / peak

    phn_path = get_timit_phn_path(wav_path)
    if not os.path.exists(phn_path):
        raise FileNotFoundError(f"Missing TIMIT phone annotation: {phn_path}")

    activity = np.zeros(len(speech), dtype=bool)
    silence_phones = {"h#", "pau", "epi"}

    with open(phn_path, "r", encoding="ascii") as phn_file:
        for line_number, line in enumerate(phn_file, start=1):
            fields = line.strip().split()
            if not fields:
                continue
            if len(fields) != 3:
                raise ValueError(
                    f"Malformed PHN line {line_number} in {phn_path}: {line!r}"
                )
            start_sample, end_sample = int(fields[0]), int(fields[1])
            phone = fields[2].lower()
            start_sample = max(0, min(start_sample, len(speech)))
            end_sample = max(start_sample, min(end_sample, len(speech)))
            if phone not in silence_phones:
                activity[start_sample:end_sample] = True

    return (
        np.ascontiguousarray(speech, dtype=np.float32),
        np.ascontiguousarray(activity, dtype=bool),
        phn_path,
    )


def load_timit_speech_and_activity(wav_path, fs=16000):
    return _load_timit_speech_and_activity_cached(os.fspath(wav_path), int(fs))


def make_repeated_speech_track(
    speech,
    speech_vad_mask,
    fs,
    rng,
    max_pause_seconds=0.5,
):
    """
    Repeat one TIMIT utterance twice and add independent random silences at
    the beginning, between repetitions, and at the end, as described by
    Poschadel et al. Each silence is uniformly sampled from [0, 0.5] s.
    """
    speech = np.asarray(speech, dtype=np.float32)
    speech_vad_mask = np.asarray(speech_vad_mask, dtype=bool)

    if speech.shape != speech_vad_mask.shape:
        raise ValueError(
            f"Speech and PHN activity must have equal shape, got "
            f"{speech.shape} and {speech_vad_mask.shape}."
        )

    max_pause_seconds = max(0.0, float(max_pause_seconds))
    pause_lengths = [
        int(round(rng.uniform(0.0, max_pause_seconds) * fs))
        for _ in range(3)
    ]
    pause_start, pause_middle, pause_end = [
        np.zeros(length, dtype=np.float32) for length in pause_lengths
    ]
    inactive_start, inactive_middle, inactive_end = [
        np.zeros(length, dtype=bool) for length in pause_lengths
    ]

    track = np.concatenate(
        [pause_start, speech, pause_middle, speech, pause_end]
    ).astype(np.float32)
    sample_mask = np.concatenate(
        [inactive_start, speech_vad_mask, inactive_middle,
         speech_vad_mask, inactive_end]
    )

    pauses_s = {
        "pause_start_s": pause_lengths[0] / float(fs),
        "pause_middle_s": pause_lengths[1] / float(fs),
        "pause_end_s": pause_lengths[2] / float(fs),
    }
    return track, sample_mask, pauses_s


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


def truncate_hoa_to_order(signal, target_order, signal_name="HOA signal"):
    """
    Keep only the ACN channels required for ``target_order``.

    HOA order N uses the first (N + 1)^2 ACN channels. Higher-order
    channels are ignored. An input with insufficient order raises an error.
    """
    signal = np.asarray(signal, dtype=np.float32)

    if signal.ndim != 2:
        raise ValueError(
            f"Expected {signal_name} with shape (channels, samples), "
            f"got {signal.shape}."
        )

    target_order = int(target_order)
    if target_order < 0:
        raise ValueError(
            f"HOA order must be non-negative, got {target_order}."
        )

    available_channels = int(signal.shape[0])
    available_order = infer_hoa_order(available_channels)
    required_channels = (target_order + 1) ** 2

    if available_channels < required_channels:
        raise ValueError(
            f"{signal_name} contains {available_channels} channels "
            f"(HOA order {available_order}), but requested order "
            f"{target_order} requires {required_channels} channels."
        )

    return np.ascontiguousarray(
        signal[:required_channels, :],
        dtype=np.float32,
    )


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

@lru_cache(maxsize=256)
def _load_mono_speech_cached(path, fs):
    speech, _ = librosa.load(path, sr=fs, mono=True)
    speech = np.asarray(speech, dtype=np.float32)

    peak = np.max(np.abs(speech))

    if peak > 0:
        speech = speech / peak

    return np.ascontiguousarray(speech, dtype=np.float32)


def load_mono_speech(path, fs):
    """
    Cached speech loading/resampling.

    The returned array must be treated as read-only. Functions in this script
    do not modify it in place.
    """
    return _load_mono_speech_cached(os.fspath(path), int(fs))


@lru_cache(maxsize=1024)
def _read_wav_float_cached(path, target_fs):
    """
    Reads and optionally resamples a WAV once, then caches the float32 result.

    Returns:
        audio with shape (channels, samples)
    """
    fs_in, audio = wavfile.read(path)
    audio = np.asarray(audio)

    if audio.dtype.kind in {"i", "u"}:
        if audio.dtype.kind == "u":
            # Unsigned PCM is normally centered around half the integer range.
            info = np.iinfo(audio.dtype)
            midpoint = (info.max + 1) / 2.0
            audio = (audio.astype(np.float32) - midpoint) / midpoint
        else:
            scale = float(max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max))
            audio = audio.astype(np.float32) / scale
    else:
        audio = audio.astype(np.float32, copy=False)

    if audio.ndim == 1:
        audio = audio[None, :]
    else:
        audio = audio.T

    if fs_in != target_fs:
        audio = librosa.resample(
            audio,
            orig_sr=fs_in,
            target_sr=target_fs,
            axis=1,
        ).astype(np.float32, copy=False)

    return np.ascontiguousarray(audio, dtype=np.float32)


def read_wav_float(path, target_fs):
    """
    Cached mono or multichannel WAV loading.

    The returned array must be treated as read-only.
    """
    return _read_wav_float_cached(os.fspath(path), int(target_fs))


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
    for key in ("rt60_normalized", "RT60", "rt60", "RT", "rt"):
        if key in rir_row.index:
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
    """
    FFT convolution of one mono track with all RIR channels at once.

    This is faster than calling scipy.signal.fftconvolve independently for
    every HOA channel because the speech FFT is computed only once.
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

    if track.size == 0 or rir.shape[1] == 0:
        output_len = track.size if keep_original_length else 0
        return np.zeros((rir.shape[0], output_len), dtype=np.float32)

    full_len = track.shape[0] + rir.shape[1] - 1
    fft_len = next_fast_len(full_len)

    track_fft = rfft(track, n=fft_len)
    rir_fft = rfft(rir, n=fft_len, axis=1)

    convolved = irfft(
        rir_fft * track_fft[None, :],
        n=fft_len,
        axis=1,
    )[:, :full_len]

    if keep_original_length:
        convolved = convolved[:, :track.shape[0]]

    return np.ascontiguousarray(convolved, dtype=np.float32)


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




def mean_active_power(signal, activity_mask, channel=0, eps=1e-12):
    """Mean-square power of one HOA channel over active samples only."""
    signal = np.asarray(signal, dtype=np.float32)
    activity_mask = np.asarray(activity_mask, dtype=bool)

    if signal.ndim != 2:
        raise ValueError(
            f"Expected signal with shape (channels, samples), got {signal.shape}."
        )
    if signal.shape[1] != activity_mask.size:
        raise ValueError(
            f"Signal and activity mask length differ: "
            f"{signal.shape[1]} != {activity_mask.size}."
        )
    if not 0 <= int(channel) < signal.shape[0]:
        raise ValueError(f"Invalid channel index {channel} for {signal.shape[0]} channels.")

    active = signal[int(channel), activity_mask].astype(np.float64)
    if active.size == 0:
        return 0.0

    power = float(np.mean(active * active))
    return power if np.isfinite(power) and power > eps else 0.0


def normalize_tracks_to_relative_sir(
    tracks,
    activity_masks,
    rng,
    sir_min_db=0.0,
    sir_max_db=10.0,
    reference_index=0,
    power_channel=0,
    eps=1e-12,
):
    """
    Scale every non-reference speaker relative to one reference speaker.

    Power is measured only over each speaker's own active samples, using the
    selected HOA channel (channel 0 / W by default). Thus inserted pauses do
    not affect the SIR normalization.
    """
    if len(tracks) != len(activity_masks):
        raise ValueError("tracks and activity_masks must have equal length.")
    if not tracks:
        return [], []

    reference_index = int(reference_index)
    if not 0 <= reference_index < len(tracks):
        raise ValueError(f"Invalid reference_index: {reference_index}.")

    scaled = [np.asarray(track, dtype=np.float32).copy() for track in tracks]
    powers_before = [
        mean_active_power(track, mask, channel=power_channel, eps=eps)
        for track, mask in zip(scaled, activity_masks)
    ]

    reference_power = powers_before[reference_index]
    if reference_power <= eps:
        raise ValueError(
            "Reference speaker has no measurable power in active samples."
        )

    normalization_info = []
    for speaker_index, (track, power) in enumerate(zip(scaled, powers_before)):
        if speaker_index == reference_index:
            target_sir_db = 0.0
            gain = 1.0
        elif power <= eps:
            target_sir_db = None
            gain = 0.0
            track *= 0.0
        else:
            target_sir_db = float(rng.uniform(sir_min_db, sir_max_db))
            gain = float(np.sqrt(
                reference_power / (power * 10.0 ** (target_sir_db / 10.0))
            ))
            track *= np.float32(gain)

        power_after = mean_active_power(
            track, activity_masks[speaker_index], channel=power_channel, eps=eps
        )
        achieved_sir_db = None
        if speaker_index != reference_index and power_after > eps:
            achieved_sir_db = float(
                10.0 * np.log10(reference_power / power_after)
            )

        normalization_info.append({
            "speaker_index": int(speaker_index),
            "reference_index": int(reference_index),
            "target_sir_db": target_sir_db,
            "achieved_sir_db": achieved_sir_db,
            "gain": float(gain),
            "active_power_before": float(power),
            "active_power_after": float(power_after),
            "power_channel": int(power_channel),
        })

    return scaled, normalization_info

def normalize_peak(signal, peak_value=0.9):
    signal = np.asarray(signal, dtype=np.float32)

    max_peak = np.max(np.abs(signal))

    if max_peak > 0:
        signal = peak_value * signal / max_peak

    return signal.astype(np.float32)


# =========================
# Diffuse babble-noise helpers
# =========================

def repeat_to_length(signal, target_len, random_offset=0, eps=1e-12):
    """Repeat a mono signal until ``target_len`` samples are available."""
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    target_len = int(target_len)
    if target_len <= 0:
        return np.zeros(0, dtype=np.float32)
    if signal.size == 0 or float(np.max(np.abs(signal))) <= eps:
        return np.zeros(target_len, dtype=np.float32)

    offset = int(random_offset) % signal.size
    if offset:
        signal = np.concatenate([signal[offset:], signal[:offset]])
    repeats = int(np.ceil(target_len / signal.size))
    return np.tile(signal, repeats)[:target_len].astype(np.float32, copy=False)


def create_repeated_babble_noise(
    speech_files,
    target_len,
    fs,
    rng,
    num_speakers=50,
):
    """
    Create an anechoic babble track by summing randomly selected TIMIT speakers.

    Every selected utterance is repeated to the full mixture length. A random
    circular starting offset prevents repetition boundaries from lining up.
    The sum is divided by sqrt(num_speakers), preserving approximately stable
    variance while retaining dense, practically pause-free babble.
    """
    if not speech_files:
        raise ValueError("No speech files available for babble generation.")
    num_speakers = int(num_speakers)
    if num_speakers <= 0:
        return np.zeros(int(target_len), dtype=np.float32), []

    replace = len(speech_files) < num_speakers
    selected = rng.choice(speech_files, size=num_speakers, replace=replace)
    babble = np.zeros(int(target_len), dtype=np.float64)

    for path in selected:
        utterance = load_mono_speech(path, fs)
        offset = int(rng.integers(0, max(1, len(utterance))))
        repeated = repeat_to_length(utterance, target_len, random_offset=offset)
        rms = float(np.sqrt(np.mean(repeated.astype(np.float64) ** 2)))
        if rms > 1e-12:
            repeated = repeated / np.float32(rms)
        babble += repeated.astype(np.float64)

    babble /= np.sqrt(float(num_speakers))
    babble -= float(np.mean(babble))
    return babble.astype(np.float32), [os.fspath(x) for x in selected]


def create_diffuse_gaussian_decay_rir(
    num_channels,
    rt60_s,
    fs,
    rng,
    max_duration_s=4.0,
):
    """
    Create an N3D/ACN diffuse late-reverberation model.

    Each HOA channel receives an independent Gaussian-noise decay. With N3D
    spherical harmonics an isotropic diffuse field has equal expected modal
    power, so no additional order-dependent gain is applied. There is no direct
    impulse and no deterministic early-reflection component.
    """
    num_channels = int(num_channels)
    infer_hoa_order(num_channels)
    rt60_s = max(float(rt60_s), 1.0 / float(fs))
    rir_len = max(2, int(round(min(float(max_duration_s), rt60_s) * fs)))
    t = np.arange(rir_len, dtype=np.float64) / float(fs)

    # Amplitude reaches -60 dB (= 1/1000) at t = RT60.
    envelope = np.exp(-np.log(1000.0) * t / rt60_s)
    noise = rng.standard_normal((num_channels, rir_len))
    rir = noise * envelope[None, :]

    # Remove any impulse-like common component and equalize channel energy.
    rir[:, 0] = 0.0
    rms = np.sqrt(np.mean(rir * rir, axis=1, keepdims=True))
    rir /= np.maximum(rms, 1e-12)
    rir *= envelope[None, :]
    return np.ascontiguousarray(rir, dtype=np.float32)


def add_diffuse_babble_noise(
    clean_ambisonics,
    speech_activity_mask,
    speech_files,
    rt60_s,
    fs,
    rng,
    num_babble_speakers=50,
    snr_db=20.0,
):
    """Add diffuse babble at the requested active-speech SNR.

    Power is measured in the N3D omnidirectional channel (ACN 0) only over
    samples where at least one target speaker is active. Despite the historical
    variable name "STI" in the request, this is an SNR/power calculation, not
    the Speech Transmission Index metric.
    """
    clean = np.asarray(clean_ambisonics, dtype=np.float32)
    activity = np.asarray(speech_activity_mask, dtype=bool)
    if clean.ndim != 2 or clean.shape[1] != activity.size:
        raise ValueError("clean_ambisonics and speech_activity_mask mismatch.")

    dry_babble, selected = create_repeated_babble_noise(
        speech_files=speech_files, target_len=clean.shape[1], fs=fs, rng=rng,
        num_speakers=num_babble_speakers,
    )
    diffuse_rir = create_diffuse_gaussian_decay_rir(
        num_channels=clean.shape[0], rt60_s=rt60_s, fs=fs, rng=rng,
    )
    diffuse_babble = convolve_mono_with_multichannel_rir(
        dry_babble, diffuse_rir, keep_original_length=True
    )

    if not np.any(activity):
        gain = 0.0
        speech_power = babble_power = 0.0
    else:
        speech_power = mean_active_power(clean, activity, channel=0)
        babble_power = mean_active_power(diffuse_babble, activity, channel=0)
        if speech_power <= 1e-12 or babble_power <= 1e-12:
            gain = 0.0
        else:
            gain = float(np.sqrt(
                speech_power / (babble_power * 10.0 ** (float(snr_db) / 10.0))
            ))
    diffuse_babble *= np.float32(gain)
    mixed = clean + diffuse_babble
    achieved_snr = None
    scaled_babble_power = mean_active_power(diffuse_babble, activity, channel=0)
    if speech_power > 1e-12 and scaled_babble_power > 1e-12:
        achieved_snr = float(10.0 * np.log10(speech_power / scaled_babble_power))

    info = {
        "enabled": True,
        "num_babble_speakers": int(num_babble_speakers),
        "rt60_s": float(rt60_s),
        "target_snr_db": float(snr_db),
        "achieved_snr_db": achieved_snr,
        "gain": float(gain),
        "speech_active_power_omni": float(speech_power),
        "babble_active_power_omni_before": float(babble_power),
        "babble_active_power_omni_after": float(scaled_babble_power),
        "selected_speech_files": selected,
    }
    return mixed.astype(np.float32), diffuse_babble.astype(np.float32), info


# =========================
# Visualization helper
# =========================

def plot_first_realigned_multitrack_sample(
    convolved_tracks,
    speaker_activity_masks,
    speaker_metadata,
    fs,
    output_path=None,
    max_seconds=None,
):
    """
    Plot the first synthesized multitrack example after RIR/VAD realignment.

    Each row shows the omnidirectional HOA channel (channel 0) of one
    reverberant speaker track. Corrected voice-active samples are shaded.
    """
    if len(convolved_tracks) != len(speaker_activity_masks):
        raise ValueError(
            "convolved_tracks and speaker_activity_masks must have equal length."
        )

    n_tracks = len(convolved_tracks)
    if n_tracks == 0:
        return

    n_samples = min(track.shape[1] for track in convolved_tracks)
    if max_seconds is not None:
        n_samples = min(n_samples, int(float(max_seconds) * fs))

    time_s = np.arange(n_samples, dtype=np.float32) / float(fs)
    fig, axes = plt.subplots(
        n_tracks,
        1,
        figsize=(15, 2.8 * n_tracks),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    for speaker_idx, (track, mask, meta) in enumerate(
        zip(convolved_tracks, speaker_activity_masks, speaker_metadata)
    ):
        waveform = np.asarray(track[0, :n_samples], dtype=np.float32)
        active = np.asarray(mask[:n_samples], dtype=bool)

        axes[speaker_idx].plot(
            time_s,
            waveform,
            linewidth=0.7,
            label=f"Speaker {speaker_idx + 1} waveform (HOA W channel)",
        )
        axes[speaker_idx].fill_between(
            time_s,
            -1.0,
            1.0,
            where=active,
            alpha=0.22,
            step="pre",
            label="Realigned voice activity",
        )
        axes[speaker_idx].set_ylim(-1.05, 1.05)
        axes[speaker_idx].set_ylabel(f"S{speaker_idx + 1}")
        axes[speaker_idx].grid(True, alpha=0.25)

        delay_ms = 1000.0 * float(meta.get("rir_delay_s", 0.0))
        extension_ms = 1000.0 * float(meta.get("vad_extension_s", 0.0))
        axes[speaker_idx].set_title(
            f"Speaker {speaker_idx + 1}: corrected VAD "
            f"(delay={delay_ms:.1f} ms, extension={extension_ms:.1f} ms)"
        )
        axes[speaker_idx].legend(loc="upper right")

    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(
        "First synthesized multitrack sample after RIR/VAD realignment",
        fontsize=14,
    )

    if output_path is not None:
        output_path = os.fspath(output_path)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        print(f"Saved first-sample waveform/VAD plot to: {output_path}")

    plt.show()
    plt.close(fig)


# =========================
# HO-PIV / ACCDOA helpers
# =========================

def get_unit_doa_from_rir_row(rir_row):
    direct_keys = ("dx", "dy", "dz")
    position_keys = (
        "x_source", "y_source", "z_source",
        "x_mic", "y_mic", "z_mic",
    )

    if all(key in rir_row.index and pd.notna(rir_row[key]) for key in direct_keys):
        vec = np.array(
            [float(rir_row["dx"]), float(rir_row["dy"]), float(rir_row["dz"])],
            dtype=np.float32,
        )
    elif all(key in rir_row.index and pd.notna(rir_row[key]) for key in position_keys):
        vec = np.array(
            [
                float(rir_row["x_source"]) - float(rir_row["x_mic"]),
                float(rir_row["y_source"]) - float(rir_row["y_mic"]),
                float(rir_row["z_source"]) - float(rir_row["z_mic"]),
            ],
            dtype=np.float32,
        )
    else:
        raise ValueError(
            f"RIR row {rir_row.get('Name', '<unknown>')} contains neither "
            "valid dx/dy/dz values nor valid source/microphone coordinates."
        )

    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        raise ValueError(
            f"Invalid zero-length DOA vector for RIR {rir_row.get('Name', '<unknown>')}"
        )
    return (vec / norm).astype(np.float32)


def vec_to_az_el_deg(vec):
    x, y, z = float(vec[0]), float(vec[1]), float(vec[2])

    az = np.degrees(np.arctan2(y, x))

    r = np.sqrt(x * x + y * y + z * z) + 1e-8
    el = np.degrees(np.arcsin(z / r))

    return float(az), float(el)


def make_sample_id(room, position):
    return f"R{int(room):04d}_P{int(position):02d}"


@lru_cache(maxsize=32)
def _cached_hann_window(frame_length):
    return np.hanning(int(frame_length)).astype(np.float32)


def stft_hoa_fixed_num_frames(
    hoa_signal,
    num_frames=50,
    hop_length=320,
    frame_length=640,
    n_fft=1024,
):
    """
    Vectorized fixed-frame STFT.

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

    if num_frames <= 0:
        return np.empty(
            (0, n_fft // 2 + 1, hoa_signal.shape[0]),
            dtype=np.complex64,
        )

    required_len = (num_frames - 1) * hop_length + frame_length
    num_samples = hoa_signal.shape[1]

    if num_samples < required_len:
        work = np.pad(
            hoa_signal,
            ((0, 0), (0, required_len - num_samples)),
            mode="constant",
        )
    else:
        work = hoa_signal[:, :required_len]

    frames = np.lib.stride_tricks.sliding_window_view(
        work,
        window_shape=frame_length,
        axis=1,
    )[:, ::hop_length, :][:, :num_frames, :]

    window = _cached_hann_window(frame_length)
    windowed = frames * window[None, None, :]

    spec = np.fft.rfft(windowed, n=n_fft, axis=-1)

    # (Q, T, F) -> (T, F, Q)
    return np.ascontiguousarray(
        spec.transpose(1, 2, 0),
        dtype=np.complex64,
    )


@lru_cache(maxsize=32)
def _make_mel_basis_cached(
    fs,
    n_fft,
    n_mels,
    fmin,
    fmax,
):
    mel_basis = librosa.filters.mel(
        sr=int(fs),
        n_fft=int(n_fft),
        n_mels=int(n_mels),
        fmin=float(fmin),
        fmax=float(fmax),
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

    return np.ascontiguousarray(mel_basis, dtype=np.float32)


def make_mel_basis(
    fs,
    n_fft=1024,
    n_mels=128,
    fmin=0.0,
    fmax=None,
):
    """
    Cached mel basis with shape (n_mels, n_fft//2 + 1).

    The returned array must be treated as read-only.
    """
    if fmax is None:
        fmax = fs / 2.0

    return _make_mel_basis_cached(
        int(fs),
        int(n_fft),
        int(n_mels),
        float(fmin),
        float(fmax),
    )


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
    """
    Vectorized frame activity calculation using cumulative sums.
    """
    directions = np.asarray(speaker_directions, dtype=np.float32)

    if directions.shape != (N_SPEAKERS, 3):
        raise ValueError(
            f"Expected speaker_directions shape {(N_SPEAKERS, 3)}, "
            f"got {directions.shape}."
        )

    frame_starts = (
        int(segment_start_sample)
        + np.arange(num_frames, dtype=np.int64) * int(hop_length)
    )
    frame_ends = frame_starts + int(frame_length)

    activity = np.zeros((num_frames, N_SPEAKERS), dtype=bool)

    for speaker_idx, mask in enumerate(speaker_activity_masks):
        mask = np.asarray(mask, dtype=np.uint8)
        cumulative = np.empty(mask.size + 1, dtype=np.int64)
        cumulative[0] = 0
        np.cumsum(mask, dtype=np.int64, out=cumulative[1:])

        starts = np.clip(frame_starts, 0, mask.size)
        ends = np.clip(frame_ends, 0, mask.size)

        active_samples = cumulative[ends] - cumulative[starts]
        frame_lengths = ends - starts

        activity[:, speaker_idx] = (
            active_samples >= 0.5 * frame_lengths
        )


    return (
        activity[:, :, None] * directions[None, :, :]
    ).astype(np.float32, copy=False)


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
    overlap_frame_counts=None,
):
    # One-second segment stride, but enough analysis context for all STFT frames.
    # With 50 frames, hop=320, and frame_length=640, the feature window spans
    # (50 - 1) * 320 + 640 = 16320 samples.  Using only 16000 samples would
    # zero-pad the second half of the final STFT frame while ACCDOA labels still
    # read real activity from the following samples, causing feature/label
    # misalignment.
    segment_stride_samples = int(fs)
    num_frames = 50
    analysis_len_samples = (num_frames - 1) * hop_length + frame_length
    min_keep_samples = 10 * hop_length

    total_len = ambisonics.shape[1]
    sample_id = make_sample_id(room, position)

    sample_records = []
    csv_rows = []

    start = 0
    segment_idx = 0

    while start < total_len:
        remaining = total_len - start

        # Keep the original one-second segmentation policy for deciding whether
        # to retain a final partial example, but provide enough right context for
        # all 50 analysis frames.  Any unavailable context at the very end of the
        # recording is zero-padded consistently for both features and labels.
        if remaining < segment_stride_samples and remaining < min_keep_samples:
            break

        available_analysis_samples = min(remaining, analysis_len_samples)
        padded = available_analysis_samples < analysis_len_samples

        segment = np.zeros(
            (ambisonics.shape[0], analysis_len_samples),
            dtype=np.float32,
        )
        segment[:, :available_analysis_samples] = ambisonics[
            :, start:start + available_analysis_samples
        ]

        X = compute_hopiv_logmel_from_hoa_segment(
            hoa_segment=segment,
            fs=fs,
            hop_length=hop_length,
            frame_length=frame_length,
            n_fft=n_fft,
            n_mels=n_mels,
        )


        accdoa = make_accdoa_for_segment(
            speaker_activity_masks=speaker_activity_masks,
            speaker_directions=speaker_directions,
            segment_start_sample=start,
            num_frames=num_frames,
            hop_length=hop_length,
            frame_length=frame_length,
        )

        # Count how many sources are active in each frame (0, 1, 2, or 3).
        # A source is active when its ACCDOA vector has non-zero magnitude.
        if overlap_frame_counts is not None:
            active_per_frame = np.sum(
                np.linalg.norm(accdoa, axis=-1) > 0.5,
                axis=1,
            )

            for overlap_count in range(N_SPEAKERS + 1):
                overlap_frame_counts[overlap_count] += int(
                    np.sum(active_per_frame == overlap_count)
                )

        base_name = f"{sample_id}_segment{segment_idx:04d}"

        # Store only the tensors needed for training.
        # Metadata remains available in the CSV output.
        sample_records.append(
            (
                torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(accdoa, dtype=np.float32)),
            )
        )

        for frame_idx in range(num_frames):
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
        start += segment_stride_samples

    return sample_records, csv_rows


def save_room_batch_pt(pt_batch_folder, batch_idx, room_records):
    """
    Save one batch as a tuple of two dense tensors:

        X:      (N, 50, n_mels, feature_channels)
        accdoa: (N, 50, 3, 3)

    No room/sample metadata is stored in the .pt file.
    """
    if len(room_records) == 0:
        return None

    os.makedirs(pt_batch_folder, exist_ok=True)

    samples = []
    for room_record in room_records:
        samples.extend(room_record["samples"])

    if not samples:
        return None

    X = torch.stack([sample[0] for sample in samples], dim=0).contiguous()
    accdoa = torch.stack([sample[1] for sample in samples], dim=0).contiguous()

    batch_name = f"rooms_batch{batch_idx:04d}.pt"
    batch_path = os.path.join(pt_batch_folder, batch_name)

    torch.save((X, accdoa), batch_path)

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
    max_pause_seconds,
    random_rotation,
    correct_vad_for_rir,
    rir_delay_threshold_db,
    rt60_extension_fraction,
    order,
    apply_relative_sir=True,
    sir_min_db=0.0,
    sir_max_db=10.0,
    add_babble=True,
    num_babble_speakers=50,
    babble_snr_db=20.0,
):
    selected_rirs = []
    for speaker_idx in range(N_SPEAKERS):
        matches = group[group["s"] == speaker_idx]
        selected_rirs.append(None if len(matches) == 0 else matches.iloc[0])

    real_source_indices = [i for i, row in enumerate(selected_rirs) if row is not None]
    if not real_source_indices:
        print(f"Skipping R{room}_P{position}: no valid sources found")
        return None

    selected_speech_files = [None] * N_SPEAKERS
    chosen = rng.choice(speech_files, size=len(real_source_indices), replace=False)
    for idx, path in zip(real_source_indices, chosen):
        selected_speech_files[idx] = path

    theta = rng.uniform(0.0, 2.0 * np.pi) if random_rotation else 0.0

    convolved_tracks = []
    dry_track_lengths = []
    speaker_activity_masks = []
    speaker_directions = []
    speaker_metadata = []

    for speaker_idx in range(N_SPEAKERS):
        rir_row = selected_rirs[speaker_idx]
        speech_path = selected_speech_files[speaker_idx]

        if rir_row is None:
            convolved_tracks.append(None)
            speaker_activity_masks.append(None)
            dry_track_lengths.append(0)
            speaker_directions.append(np.zeros(3, dtype=np.float32))
            speaker_metadata.append({
                "speech_path": None,
                "speech_file": None,
                "phn_path": None,
                "phn_file": None,
                "rir_file": None,
                "pause_start_s": 0.0,
                "pause_middle_s": 0.0,
                "pause_end_s": 0.0,
                "rir_delay_samples": 0,
                "rir_delay_s": 0.0,
                "rt60_s": 0.0,
                "vad_extension_samples": 0,
                "vad_extension_s": 0.0,
                "missing_source": True,
                "rir_metadata": {
                    "Name": None, "RT": None, "rt60": None, "c50": None,
                    "x_room": None, "y_room": None, "z_room": None,
                    "x_source": None, "y_source": None, "z_source": None,
                    "x_mic": None, "y_mic": None, "z_mic": None,
                    "dx": 0.0, "dy": 0.0, "dz": 0.0,
                    "az": None, "el": None, "status": "Missing",
                },
            })
            continue

        rir_path = os.path.join(rir_folder, rir_row["Name"])
        if not os.path.exists(rir_path):
            raise FileNotFoundError(f"Missing RIR file: {rir_path}")

        speech, speech_vad_mask, phn_path = load_timit_speech_and_activity(speech_path, fs=fs)
        track, sample_mask, pauses_s = make_repeated_speech_track(
            speech=speech,
            speech_vad_mask=speech_vad_mask,
            fs=fs,
            rng=rng,
            max_pause_seconds=max_pause_seconds,
        )
        rir = read_wav_float(rir_path, target_fs=fs)
        rir = truncate_hoa_to_order(
            rir,
            target_order=order,
            signal_name=f"RIR {rir_row['Name']}",
        )

        rir_delay_samples = 0
        rir_delay_s = 0.0
        rt60_s = get_rt60_seconds_from_rir_row(rir_row)
        vad_extension_samples = 0
        vad_extension_s = 0.0
        if correct_vad_for_rir:
            rir_delay_samples = estimate_rir_delay_samples(rir, threshold_db=rir_delay_threshold_db)
            rir_delay_s = rir_delay_samples / float(fs)
            vad_extension_s = float(rt60_s) * float(rt60_extension_fraction)
            vad_extension_samples = int(round(vad_extension_s * fs))
            sample_mask = delay_and_extend_activity_mask(
                sample_mask, rir_delay_samples, vad_extension_samples, target_len=len(track)
            )

        dry_track_lengths.append(len(track))
        speaker_activity_masks.append(sample_mask)
        speaker_directions.append(get_unit_doa_from_rir_row(rir_row))
        speaker_metadata.append({
            "speech_path": "DARPA_TIMIT/" + speech_path.split("DARPA_TIMIT/", 1)[-1],
            "speech_file": os.path.basename(speech_path),
            "phn_path": phn_path,
            "phn_file": os.path.basename(phn_path),
            "rir_file": rir_row["Name"],
            **pauses_s,
            "rir_delay_samples": int(rir_delay_samples),
            "rir_delay_s": float(rir_delay_s),
            "rt60_s": float(rt60_s),
            "vad_extension_samples": int(vad_extension_samples),
            "vad_extension_s": float(vad_extension_s),
            "missing_source": False,
            "rir_metadata": {
                "Name": rir_row.get("Name", None),
                "RT": float(rt60_s),
                "rt60": float(rt60_s),
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
                "dx": rir_row.get("dx", None),
                "dy": rir_row.get("dy", None),
                "dz": rir_row.get("dz", None),
                "az": rir_row.get("az", None),
                "el": rir_row.get("el", None),
                "status": rir_row.get("status", None),
            },
        })
        conv = convolve_mono_with_multichannel_rir(track, rir, keep_original_length=True)
        convolved_tracks.append(conv)

    real_tracks = [x for x in convolved_tracks if x is not None]
    num_channels = check_same_hoa_channel_count(real_tracks)
    hoa_order = infer_hoa_order(num_channels)
    max_len = max(length for length in dry_track_lengths if length > 0)

    for i in range(N_SPEAKERS):
        if convolved_tracks[i] is None:
            convolved_tracks[i] = np.zeros((num_channels, max_len), dtype=np.float32)
            speaker_activity_masks[i] = np.zeros(max_len, dtype=bool)
        else:
            convolved_tracks[i] = pad_multichannel_to_length(convolved_tracks[i], max_len)
            speaker_activity_masks[i] = pad_bool_mask_to_length(speaker_activity_masks[i], max_len)

    speaker_directions = np.asarray(speaker_directions, dtype=np.float32)

    sir_normalization = []
    if apply_relative_sir:
        convolved_tracks, sir_normalization = normalize_tracks_to_relative_sir(
            tracks=convolved_tracks,
            activity_masks=speaker_activity_masks,
            rng=rng,
            sir_min_db=sir_min_db,
            sir_max_db=sir_max_db,
            reference_index=0,
            power_channel=0,
        )
        for meta, info in zip(speaker_metadata, sir_normalization):
            meta.update({
                "sir_reference_speaker": int(info["reference_index"]),
                "target_sir_db": info["target_sir_db"],
                "achieved_sir_db": info["achieved_sir_db"],
                "sir_gain": float(info["gain"]),
                "active_power_before": float(info["active_power_before"]),
                "active_power_after": float(info["active_power_after"]),
            })

    ambisonics = np.sum(convolved_tracks, axis=0).astype(np.float32)
    babble_noise = np.zeros_like(ambisonics)
    babble_info = {"enabled": False}
    if add_babble:
        active_any = np.any(np.stack(speaker_activity_masks, axis=0), axis=0)
        loaded_rt60 = [
            float(meta.get("rt60_s", 0.0))
            for meta in speaker_metadata
            if not meta.get("missing_source", False)
            and np.isfinite(float(meta.get("rt60_s", 0.0)))
            and float(meta.get("rt60_s", 0.0)) > 0.0
        ]
        average_rt60 = float(np.mean(loaded_rt60)) if loaded_rt60 else 0.5
        ambisonics, babble_noise, babble_info = add_diffuse_babble_noise(
            clean_ambisonics=ambisonics,
            speech_activity_mask=active_any,
            speech_files=speech_files,
            rt60_s=average_rt60,
            fs=fs,
            rng=rng,
            num_babble_speakers=num_babble_speakers,
            snr_db=babble_snr_db,
        )

    if random_rotation:
        ambisonics = rotate_hoa(ambisonics, theta)
        speaker_directions = rotate_vectors_z(speaker_directions, theta)
    ambisonics = normalize_peak(ambisonics, peak_value=0.9)

    return {
        "ambisonics": ambisonics,
        "speaker_activity_masks": speaker_activity_masks,
        "speaker_directions": speaker_directions,
        "speaker_metadata": speaker_metadata,
        "convolved_tracks": convolved_tracks,
        "theta": theta,
        "num_channels": num_channels,
        "hoa_order": hoa_order,
        "max_len": max_len,
        "sir_normalization": sir_normalization,
        "babble_noise": babble_noise,
        "babble_info": babble_info,
    }


# =========================
# Main dataset generation
# =========================


def build_room_groups(csv_file, nrooms=None):
    df = pd.read_csv(csv_file)

    if "Name" not in df.columns:
        raise ValueError("CSV is missing required column: Name")

    if "status" in df.columns:
        valid = (
            df["status"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("success")
        )

        removed = int((~valid).sum())
        if removed:
            print(f"Ignoring {removed} CSV rows whose status is not Success.")

        df = df[valid].copy()

    if df.empty:
        raise ValueError("No valid CSV rows remain after status filtering.")

    names = df["Name"].astype(str).str.strip()

    # Accept filenames such as:
    #   anisotropic_P1_S1.wav
    #   room_R12_P3_S2.wav
    #   prefix_P4_S3_extra.wav
    #
    # The negative digit lookahead ensures that P12 or S10 is captured fully,
    # while allowing '.', '_', or the end of the string afterward.
    position = names.str.extract(
        r"(?:^|_)P(\d+)(?!\d)",
        expand=False,
    )
    speaker = names.str.extract(
        r"(?:^|_)S(\d+)(?!\d)",
        expand=False,
    )

    if position.isna().any():
        invalid_names = names[position.isna()].head(10).tolist()
        raise ValueError(
            f"Could not extract P<number> from: {invalid_names}"
        )

    if speaker.isna().any():
        invalid_names = names[speaker.isna()].head(10).tolist()
        raise ValueError(
            f"Could not extract S<number> from: {invalid_names}"
        )

    df["p"] = position.astype(int)

    raw_s = speaker.astype(int)
    min_s = int(raw_s.min())

    if min_s == 0:
        # Filenames already use zero-based speaker numbering:
        # S0, S1, S2
        df["s"] = raw_s
    elif min_s == 1:
        # Convert one-based numbering:
        # S1, S2, S3 -> 0, 1, 2
        df["s"] = raw_s - 1
    else:
        raise ValueError(
            f"Speaker numbering must start at S0 or S1, found S{min_s}."
        )

    invalid = (df["s"] < 0) | (df["s"] >= N_SPEAKERS)

    if invalid.any():
        invalid_entries = (
            df.loc[invalid, ["Name", "s"]]
            .head(10)
            .to_dict("records")
        )
        raise ValueError(
            f"Speaker indices must be in slots 0-{N_SPEAKERS - 1}. "
            f"Invalid entries: {invalid_entries}"
        )

    # Prefer an explicit R<number> token when present.
    explicit_room = names.str.extract(
        r"(?:^|_)R(\d+)(?!\d)",
        expand=False,
    )

    # Otherwise use everything before _P<number>_S<number> as the room or
    # dataset key. For example:
    #
    #   anisotropic_P1_S1.wav -> anisotropic
    #
    # The lookahead ensures that the speaker number is followed by an
    # underscore, a period, or the end of the filename.
    prefix = names.str.extract(
        r"^(.+?)_P\d+_S\d+(?=_|\.|$)",
        expand=False,
    )

    room_keys = []

    for row_index, name in names.items():
        if pd.notna(explicit_room.loc[row_index]):
            room_key = f"R{int(explicit_room.loc[row_index])}"
        elif pd.notna(prefix.loc[row_index]):
            room_key = str(prefix.loc[row_index])
        else:
            raise ValueError(
                f"Could not infer room/dataset key from filename: {name}"
            )

        room_keys.append(room_key)

    df["room_key"] = room_keys


    def natural_sort_key(value):
        return [
            int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", str(value))
        ]


    unique_keys = sorted(
        df["room_key"].unique().tolist(),
        key=natural_sort_key,
    )

    room_mapping = {
        room_key: room_index
        for room_index, room_key in enumerate(unique_keys)
    }

    df["r"] = df["room_key"].map(room_mapping).astype(int)


    if "RT60" in df.columns:
        rt60 = pd.to_numeric(df["RT60"], errors="coerce")
    elif "rt60" in df.columns:
        rt60 = pd.to_numeric(df["rt60"], errors="coerce")
    elif "RT" in df.columns:
        rt60 = pd.to_numeric(df["RT"], errors="coerce")
    elif "rt" in df.columns:
        rt60 = pd.to_numeric(df["rt"], errors="coerce")
    else:
        rt60 = pd.Series(
            0.0,
            index=df.index,
            dtype=np.float64,
        )

    df["rt60_normalized"] = (
        rt60
        .fillna(0.0)
        .clip(lower=0.0)
        .astype(float)
    )

    has_direct_doa = {"dx", "dy", "dz"}.issubset(df.columns)

    position_columns = {
        "x_source",
        "y_source",
        "z_source",
        "x_mic",
        "y_mic",
        "z_mic",
    }
    has_positions = position_columns.issubset(df.columns)

    if not has_direct_doa and not has_positions:
        raise ValueError(
            "CSV must contain either dx/dy/dz or "
            "source/microphone coordinates."
        )

    all_rooms = sorted(df["r"].unique().tolist())

    if nrooms is not None:
        nrooms = max(0, int(nrooms))
        all_rooms = all_rooms[:nrooms]
        df = df[df["r"].isin(all_rooms)].copy()

    position_groups = [
        (
            int(room),
            int(position_value),
            group.sort_values("s").copy(),
        )
        for (room, position_value), group in df.groupby(
            ["r", "p"],
            sort=True,
        )
    ]

    groups_by_room = {
        room: [
            (position_value, group)
            for group_room, position_value, group in position_groups
            if group_room == room
        ]
        for room in all_rooms
    }

    return all_rooms, groups_by_room


def make_position_rng(seed, room, position):
    # Stable regardless of task count, task assignment, or loop ordering.
    seed_sequence = np.random.SeedSequence(
        [int(seed), int(room), int(position)]
    )
    return np.random.default_rng(seed_sequence)


def generate_ambisonics_wavs(
    csv_file,
    rir_folder,
    speech_folder,
    output_folder,
    fs=16000,
    seed=0,
    hop_length=320,
    vad_frame_length=640,
    max_pause_seconds=0.5,
    random_rotation=True,
    n_fft=1024,
    n_mels=128,
    order=3,
    nrooms=None,
    write_wavs=True,
    rooms_per_pt=100,
    correct_vad_for_rir=True,
    rir_delay_threshold_db=-40.0,
    rt60_extension_fraction=1.0 / 6.0,
    plot_first_sample=False,
    first_sample_plot_max_seconds=None,
    task_id=0,
    num_tasks=1,
    apply_relative_sir=True,
    sir_min_db=0.0,
    sir_max_db=10.0,
    add_babble=True,
    num_babble_speakers=50,
    babble_snr_db=20.0,
):
    task_id = int(task_id)
    num_tasks = int(num_tasks)
    order = int(order)

    if order < 0:
        raise ValueError(f"order must be non-negative, got {order}.")

    target_num_channels = (order + 1) ** 2
    if num_tasks <= 0:
        raise ValueError("num_tasks must be positive.")
    if not 0 <= task_id < num_tasks:
        raise ValueError(
            f"task_id must be in [0, {num_tasks - 1}], got {task_id}."
        )

    all_rooms, all_groups_by_room = build_room_groups(csv_file, nrooms=nrooms)

    # Build final room batches globally, then assign complete batches to tasks.
    # Every task can write directly into shared pt/ and csv/ folders without
    # filename collisions or a merge step.
    all_room_batches = [
        all_rooms[start:start + rooms_per_pt]
        for start in range(0, len(all_rooms), rooms_per_pt)
    ]
    assigned_room_batches = all_room_batches[task_id::num_tasks]
    rooms = [room for batch in assigned_room_batches for room in batch]

    groups_by_room = {room: all_groups_by_room[room] for room in rooms}

    print(
        f"Task={task_id}/{num_tasks} | "
        f"assigned rooms={len(rooms)}/{len(all_rooms)}"
    )
    print(
        f"Assigned batches={len(assigned_room_batches)}/"
        f"{len(all_room_batches)}"
    )
    if rooms:
        print(f"First room={rooms[0]}, last room={rooms[-1]}")
    else:
        print("This task has no assigned rooms.")

    speech_files = sorted(
        path for path in glob.glob(
            os.path.join(speech_folder, "**", "*.WAV.wav"),
            recursive=True,
        )
        if os.path.isfile(path) and os.path.isfile(get_timit_phn_path(path))
    )
    if len(speech_files) < N_SPEAKERS:
        raise ValueError(
            f"Need at least {N_SPEAKERS} TIMIT *.WAV.wav/.PHN pairs, "
            f"but found {len(speech_files)}."
        )

    _ = make_mel_basis(
        fs=fs,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=0.0,
        fmax=fs / 2.0,
    )

    os.makedirs(output_folder, exist_ok=True)

    wav_folder = os.path.join(output_folder, "wav")
    room_csv_folder = os.path.join(output_folder, "csv")
    pt_batch_folder = os.path.join(output_folder, "pt")
    os.makedirs(wav_folder, exist_ok=True)
    os.makedirs(room_csv_folder, exist_ok=True)
    os.makedirs(pt_batch_folder, exist_ok=True)

    overlap_frame_counts = np.zeros(N_SPEAKERS + 1, dtype=np.int64)
    first_sample_plotted = False

    def save_batch(batch_rooms, batch_room_records, batch_csv_rows):
        if not batch_room_records:
            return

        first_room = int(batch_rooms[0])
        last_room = int(batch_rooms[-1])
        base_name = f"room_{first_room:04d}-{last_room:04d}"

        pt_path = os.path.join(pt_batch_folder, base_name + ".pt")
        csv_path = os.path.join(room_csv_folder, base_name + ".csv")

        samples = []
        for room_record in batch_room_records:
            samples.extend(room_record["samples"])

        if not samples:
            return

        X = torch.stack(
            [sample[0] for sample in samples],
            dim=0,
        ).contiguous()
        accdoa = torch.stack(
            [sample[1] for sample in samples],
            dim=0,
        ).contiguous()

        # The .pt file contains only training tensors, without dictionaries
        # or redundant metadata. Metadata remains in the companion CSV.
        torch.save((X, accdoa), pt_path)
        pd.DataFrame(batch_csv_rows).to_csv(csv_path, index=False)

        print(
            f"Saved {os.path.basename(pt_path)} and "
            f"{os.path.basename(csv_path)} with "
            f"{len(samples)} samples from "
            f"{len(batch_room_records)} rooms | "
            f"X={tuple(X.shape)} accdoa={tuple(accdoa.shape)}"
        )

    for batch_rooms in tqdm(
        assigned_room_batches,
        desc=f"Generate task {task_id} batches",
        position=0,
    ):
        batch_room_records = []
        batch_csv_rows = []
        batch_name = (
            f"room_{int(batch_rooms[0]):04d}-"
            f"{int(batch_rooms[-1]):04d}.pt"
        )

        for room in tqdm(
            batch_rooms,
            desc=f"Task {task_id} rooms",
            unit="room",
            leave=False,
            position=1,
        ):
            room_sample_records = []
            room_csv_rows = []

            for position_idx, group in groups_by_room[room]:
                rng = make_position_rng(seed, room, position_idx)

                synthesized = synthesize_one_position(
                    group=group,
                    room=room,
                    position=position_idx,
                    rir_folder=rir_folder,
                    speech_files=speech_files,
                    fs=fs,
                    rng=rng,
                    hop_length=hop_length,
                    vad_frame_length=vad_frame_length,
                    max_pause_seconds=max_pause_seconds,
                    random_rotation=random_rotation,
                    correct_vad_for_rir=correct_vad_for_rir,
                    rir_delay_threshold_db=rir_delay_threshold_db,
                    rt60_extension_fraction=rt60_extension_fraction,
                    order=order,
                    apply_relative_sir=apply_relative_sir,
                    sir_min_db=sir_min_db,
                    sir_max_db=sir_max_db,
                    add_babble=add_babble,
                    num_babble_speakers=num_babble_speakers,
                    babble_snr_db=babble_snr_db,
                )

                if synthesized is None:
                    continue

                if plot_first_sample and not first_sample_plotted:
                    plot_first_realigned_multitrack_sample(
                        convolved_tracks=synthesized["convolved_tracks"],
                        speaker_activity_masks=synthesized["speaker_activity_masks"],
                        speaker_metadata=synthesized["speaker_metadata"],
                        fs=fs,
                        output_path=os.path.join(
                            output_folder,
                            f"first_realigned_multitrack_sample_task{task_id:04d}.png",
                        ),
                        max_seconds=first_sample_plot_max_seconds,
                    )
                    first_sample_plotted = True

                # Use the first available RIR filename.
                rir_name = next(
                    meta["rir_file"]
                    for meta in synthesized["speaker_metadata"]
                    if meta["rir_file"] is not None
                )

                rir_stem = os.path.splitext(os.path.basename(rir_name))[0]

                # Example: R0_P0_S0_IR -> R0_P0
                rir_stem = re.sub(r"_S\d+(?:_IR)?$", "", rir_stem)

                output_wav_name = rir_stem + ".wav"

                output_wav_path = os.path.join(
                    wav_folder,
                    output_wav_name,
                )

                if write_wavs:
                    wavfile.write(
                        output_wav_path,
                        fs,
                        synthesized["ambisonics"].T.astype(np.float32),
                    )

                sample_records, csv_rows = save_hopiv_samples_for_mixture(
                    ambisonics=synthesized["ambisonics"],
                    speaker_activity_masks=synthesized["speaker_activity_masks"],
                    speaker_directions=synthesized["speaker_directions"],
                    speaker_metadata=synthesized["speaker_metadata"],
                    room=room,
                    position=position_idx,
                    fs=fs,
                    theta=synthesized["theta"],
                    output_wav_name=output_wav_name,
                    pt_batch_name=batch_name,
                    hop_length=hop_length,
                    frame_length=vad_frame_length,
                    n_fft=n_fft,
                    n_mels=n_mels,
                    overlap_frame_counts=overlap_frame_counts,
                )
                room_sample_records.extend(sample_records)
                room_csv_rows.extend(csv_rows)

            if not room_sample_records:
                print(f"Skipping room R{room}: no valid samples")
                continue

            batch_room_records.append(
                {
                    "room": int(room),
                    "num_samples": int(len(room_sample_records)),
                    "samples": room_sample_records,
                }
            )
            batch_csv_rows.extend(room_csv_rows)

        save_batch(batch_rooms, batch_room_records, batch_csv_rows)

    total_overlap_frames = int(np.sum(overlap_frame_counts))
    print("\nSource-overlap distribution for this task:")
    if total_overlap_frames == 0:
        print("No generated frames were available.")
    else:
        for overlap_count, frame_count in enumerate(overlap_frame_counts):
            percentage = 100.0 * int(frame_count) / total_overlap_frames
            print(
                f"  {overlap_count} active source(s): "
                f"{int(frame_count)} frames ({percentage:.2f}%)"
            )
        print(f"  Total: {total_overlap_frames} frames (100.00%)")

    print(f"PT output folder: {pt_batch_folder}")
    print(f"CSV output folder: {room_csv_folder}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parallel Slurm dataset generation for HOA/HO-PIV data."
    )

    parser.add_argument(
        "--order",
        type=int,
        default=3,
        help="Target HOA order.",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        required=True,
        help="Base directory containing the RIR CSV and RIRs folder.",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        required=True,
        help="Directory for generated WAV, PT, CSV, and plot files.",
    )
    parser.add_argument(
        "--nrooms",
        type=int,
        default=None,
        help="Maximum number of rooms to process.",
    )

    parser.add_argument(
        "--random-rotation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply a random rotation around the z-axis.",
    )
    parser.add_argument(
        "--write-wavs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write synthesized ambisonic WAV files.",
    )
    parser.add_argument(
        "--plot-first-sample",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Plot the first synthesized sample assigned to this task.",
    )

    return parser.parse_args()


def get_slurm_task_info():
    """
    Return the zero-based Slurm array task ID and number of array tasks.

    Outside a Slurm array, this falls back to task_id=0 and num_tasks=1.
    """
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))

    if "SLURM_ARRAY_TASK_COUNT" in os.environ:
        num_tasks = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
    elif (
        "SLURM_ARRAY_TASK_MIN" in os.environ
        and "SLURM_ARRAY_TASK_MAX" in os.environ
    ):
        task_min = int(os.environ["SLURM_ARRAY_TASK_MIN"])
        task_max = int(os.environ["SLURM_ARRAY_TASK_MAX"])
        task_step = int(os.environ.get("SLURM_ARRAY_TASK_STEP", "1"))

        num_tasks = ((task_max - task_min) // task_step) + 1

        # Convert a possibly non-zero/non-unit Slurm array index to a
        # contiguous zero-based task index.
        task_id = (task_id - task_min) // task_step
    else:
        num_tasks = 1

    return task_id, num_tasks


if __name__ == "__main__":
    args = parse_args()
    task_id, num_tasks = get_slurm_task_info()

    base_dir = os.path.abspath(os.path.expanduser(args.base_dir))
    output_folder = os.path.abspath(
        os.path.expanduser(args.output_folder)
    )

    csv_file = next(
        (
            path
            for path in (
                os.path.join(base_dir, "Generated_HOA_SRIR_data.csv"),
                os.path.join(base_dir, "ivan_directions_rt60.csv"),
                os.path.join(base_dir, "arni_directions_rt60.csv"),
            )
            if os.path.isfile(path)
        ),
        None,
    )

    rir_folder = next(
        (
            path
            for path in (
                os.path.join(base_dir, "RIRs_truncated"),
                os.path.join(base_dir, "RIRs"),
            )
            if os.path.isdir(path)
        ),
        os.path.join(base_dir, "RIRs"),
    )

    speech_folder = (
        "/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/"
        "staubf1/Projects/Master_Thesis/data/DARPA_TIMIT/data/TRAIN"
    )

    print(
        f"Slurm task assignment: task_id={task_id}, "
        f"num_tasks={num_tasks}"
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
        max_pause_seconds=1.5,
        random_rotation=args.random_rotation,
        n_fft=1024,
        n_mels=128,
        order=args.order,
        nrooms=args.nrooms,
        write_wavs=args.write_wavs,
        rooms_per_pt=25,
        correct_vad_for_rir=True,
        rir_delay_threshold_db=-40.0,
        rt60_extension_fraction=1.0 / 6.0,
        plot_first_sample=args.plot_first_sample,
        task_id=task_id,
        num_tasks=num_tasks,
    )