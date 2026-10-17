import os
import numpy as np
import pandas as pd
from scipy.io import wavfile
import random
from scipy.signal import fftconvolve
import csv
import sys
import resource
import time
from tqdm import tqdm
import librosa
from concurrent.futures import ProcessPoolExecutor
from scipy.signal import resample_poly
from scipy.stats import linregress
from scipy.signal import butter, sosfiltfilt
from scipy.special import spherical_jn, spherical_yn


def detect_active_regions_relative(signal, fs, frame_len=512, hop_len=256, rel_threshold=0.01):
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

def random_segment_with_activity(sig, segment_len_samples, active_regions, min_active_ratio=0.8, max_trials=1000):
    if len(active_regions) == 0:
        return 0  # fallback

    for _ in range(max_trials):
        candidates = [r for r in active_regions if (r[1]-r[0]) >= segment_len_samples * min_active_ratio]
        if candidates:
            start, end = random.choice(candidates)
            max_start = max(start, end - segment_len_samples)
            start_sample = random.randint(start, max_start)
        else:
            start, end = max(active_regions, key=lambda r: r[1]-r[0])
            start_sample = start

        active_count = sum(
            max(0, min(r_end, start_sample + segment_len_samples) - max(r_start, start_sample))
            for r_start, r_end in active_regions
        )

        if active_count / segment_len_samples >= min_active_ratio:
            return start_sample

    return start_sample


def compute_scm(signal_bands):
    n_bands, n_channels, n_samples = signal_bands.time.shape
    cov_all_bands = np.zeros((n_bands, n_channels, n_channels), dtype=np.float32)

    for b in range(n_bands):
        X = signal_bands.time[b]
        cov_all_bands[b] = X @ X.T
        max_val = np.max(np.abs(cov_all_bands[b]))
        if max_val > 0:
            cov_all_bands[b] = cov_all_bands[b] / max_val

    # Normalize globally
    #min_val = cov_all_bands.min()
    #max_val = cov_all_bands.max()
    #cov_all_bands = (cov_all_bands - min_val) / (max_val - min_val + 1e-8)

    # Normalize locally
    #for b in range(cov_all_bands.shape[0]):
    #    max_val = cov_all_bands[b].max()
    #    cov_all_bands[b] = (cov_all_bands[b] - min_val) / (max_val - min_val + 1e-8)

    return cov_all_bands


def compute_scm_avg(signal, fs, freq_range, num_scms):
    """
    REPAIR-style SCM computation using GLOBAL FFT (no STFT).

    Differences from original:
    - keeps complex Hermitian SCMs
    - no magnitude operation
    - energy-consistent normalization (trace-based)
    - frequency-bin grouping like REPAIR
    """

    signal = np.asarray(signal)

    if signal.ndim != 2:
        raise ValueError("signal must have shape (n_channels, n_samples)")
    
    # DC offset removal
    signal = signal - np.mean(signal, axis=1, keepdims=True)

    n_channels, n_samples = signal.shape

    # -----------------------------
    # 1. GLOBAL FFT (REPAIR simplification)
    # -----------------------------
    X_f = np.fft.rfft(signal, axis=1)  # (n_channels, n_freqs)

    freqs = np.fft.rfftfreq(n_samples, d=1 / fs)

    # Frequency selection
    f_min, f_max = freq_range
    valid_idx = np.where((freqs >= f_min) & (freqs <= f_max))[0]

    if len(valid_idx) == 0:
        raise ValueError("No frequency bins found in frequency_range")

    X_f = X_f[:, valid_idx]

    # -----------------------------
    # 2. Frequency grouping (REPAIR-style)
    # -----------------------------
    freq_groups = np.array_split(np.arange(X_f.shape[1]), num_scms)

    cov_all_bands = np.zeros(
        (num_scms, n_channels, n_channels),
        dtype=np.complex64
    )

    # -----------------------------
    # 3. SCM computation per group
    # -----------------------------
    for b, group in enumerate(freq_groups):

        scm_accum = np.zeros((n_channels, n_channels), dtype=np.complex64)

        for k in group:
            x = X_f[:, k:k+1]  # (n_channels, 1)

            # Hermitian SCM (REPAIR core operation)
            scm_accum += x @ x.conj().T

        # Average over frequency bins
        scm = scm_accum / len(group)
        scm = np.real(scm)  # discard imaginary part (should be negligible)

        # Normalization
        cov_all_bands[b] = scm / np.max(np.abs(scm)) #(np.trace(scm) + 1e-8)
        cov_all_bands[b] = (cov_all_bands[b] + 1.0) / 2.0  # map from [-1, 1] to [0, 1]
    
    return np.real(cov_all_bands)


def compute_scm_avg_log(signal, fs, freq_range, num_scms):
    """
    REPAIR-style SCM computation using GLOBAL FFT with logarithmic frequency grouping.

    Changes:
    - Log-spaced frequency bands instead of linear splits
    - Hermitian SCM computation preserved
    - Energy-consistent normalization kept
    """

    signal = np.asarray(signal)

    if signal.ndim != 2:
        raise ValueError("signal must have shape (n_channels, n_samples)")

    # DC offset removal
    signal = signal - np.mean(signal, axis=1, keepdims=True)

    n_channels, n_samples = signal.shape

    # -----------------------------
    # 1. GLOBAL FFT
    # -----------------------------
    X_f = np.fft.rfft(signal, axis=1)  # (n_channels, n_freqs)
    freqs = np.fft.rfftfreq(n_samples, d=1 / fs)

    # Frequency selection
    f_min, f_max = freq_range
    valid_idx = np.where((freqs >= f_min) & (freqs <= f_max))[0]

    if len(valid_idx) == 0:
        raise ValueError("No frequency bins found in frequency_range")

    X_f = X_f[:, valid_idx]
    freqs = freqs[valid_idx]

    # -----------------------------
    # 2. Logarithmic frequency grouping
    # -----------------------------
    f_min_safe = max(f_min, 1e-6)

    log_edges = np.logspace(
        np.log10(f_min_safe),
        np.log10(f_max),
        num_scms + 1
    )

    freq_groups = []
    for i in range(num_scms):
        f_low, f_high = log_edges[i], log_edges[i + 1]

        group = np.where((freqs >= f_low) & (freqs < f_high))[0]
        if len(group) > 0:
            freq_groups.append(group)

    # Update number of valid bands
    num_scms = len(freq_groups)

    if num_scms == 0:
        raise ValueError("All log-frequency bands are empty")

    # -----------------------------
    # 3. SCM computation per group
    # -----------------------------
    cov_all_bands = np.zeros(
        (num_scms, n_channels, n_channels),
        dtype=np.float32
    )

    for b, group in enumerate(freq_groups):

        scm_accum = np.zeros((n_channels, n_channels), dtype=np.complex64)

        for k in group:
            x = X_f[:, k:k+1]  # (n_channels, 1)

            # Hermitian SCM
            scm_accum += x @ x.conj().T

        scm = scm_accum / len(group)

        # Keep real part (imag should be negligible)
        scm = np.real(scm)

        # Normalization (robust scaling)
        denom = np.max(np.abs(scm))
        if denom < 1e-8:
            denom = 1e-8

        scm = scm / denom

        # Map from [-1, 1] to [0, 1]
        scm = (scm + 1.0) / 2.0

        cov_all_bands[b] = scm.astype(np.float32)

    return cov_all_bands


def compute_scm_avg_log_stft(
    signal,
    fs,
    freq_range,
    num_scms,
    n_fft=256,
    hop_length=None,
    window="hann",
):
    """
    REPAIR-style SCM computation using STFT windows with logarithmic frequency grouping.

    Returns
    -------
    cov_all_windows : np.ndarray
        Shape:
            (n_windows, n_bands, n_channels, n_channels)

        Each entry cov_all_windows[t, b] is the SCM for time window t
        and logarithmic frequency band b.

    Notes
    -----
    - Computes a multichannel STFT.
    - For each STFT time window, computes SCMs over log-spaced frequency bands.
    - Hermitian SCM computation is preserved.
    - Per-window, per-band robust normalization is applied.
    """

    signal = np.asarray(signal)

    if signal.ndim != 2:
        raise ValueError("signal must have shape (n_channels, n_samples)")

    n_channels, n_samples = signal.shape

    if hop_length is None:
        hop_length = n_fft // 2

    if n_fft > n_samples:
        raise ValueError("n_fft cannot be larger than the number of samples")

    # -----------------------------
    # 1. DC offset removal
    # -----------------------------
    signal = signal - np.mean(signal, axis=1, keepdims=True)

    # -----------------------------
    # 2. Window creation
    # -----------------------------
    if window == "hann":
        win = np.hanning(n_fft)
    elif window == "hamming":
        win = np.hamming(n_fft)
    elif window == "rect":
        win = np.ones(n_fft)
    else:
        raise ValueError("window must be one of: 'hann', 'hamming', 'rect'")

    win = win.astype(signal.dtype)

    # -----------------------------
    # 3. Frame signal into STFT windows
    # -----------------------------
    n_windows = 1 + (n_samples - n_fft) // hop_length

    if n_windows <= 0:
        raise ValueError("No STFT windows could be created")

    frames = np.zeros(
        (n_windows, n_channels, n_fft),
        dtype=signal.dtype
    )

    for t in range(n_windows):
        start = t * hop_length
        end = start + n_fft
        frames[t] = signal[:, start:end] * win

    # -----------------------------
    # 4. STFT over each window
    # -----------------------------
    X = np.fft.rfft(frames, axis=2)
    # Shape: (n_windows, n_channels, n_freqs)

    freqs = np.fft.rfftfreq(n_fft, d=1 / fs)

    # -----------------------------
    # 5. Frequency selection
    # -----------------------------
    f_min, f_max = freq_range

    valid_idx = np.where((freqs >= f_min) & (freqs <= f_max))[0]

    if len(valid_idx) == 0:
        raise ValueError("No frequency bins found in frequency_range")

    X = X[:, :, valid_idx]
    freqs = freqs[valid_idx]

    # -----------------------------
    # 6. Logarithmic frequency grouping
    # -----------------------------
    if f_max <= 0:
        raise ValueError("f_max must be positive for logarithmic grouping")

    f_min_safe = max(f_min, 1e-6)

    log_edges = np.logspace(
        np.log10(f_min_safe),
        np.log10(f_max),
        num_scms + 1
    )

    freq_groups = []

    for i in range(num_scms):
        f_low, f_high = log_edges[i], log_edges[i + 1]

        if i == num_scms - 1:
            group = np.where((freqs >= f_low) & (freqs <= f_high))[0]
        else:
            group = np.where((freqs >= f_low) & (freqs < f_high))[0]

        if len(group) > 0:
            freq_groups.append(group)

    n_bands = len(freq_groups)

    if n_bands == 0:
        raise ValueError("All log-frequency bands are empty")

    # -----------------------------
    # 7. SCM computation per time window and frequency group
    # -----------------------------
    cov_all_windows = np.zeros(
        (n_windows, n_bands, n_channels, n_channels),
        dtype=np.float32
    )

    for t in range(n_windows):
        for b, group in enumerate(freq_groups):

            scm_accum = np.zeros(
                (n_channels, n_channels),
                dtype=np.complex64
            )

            for k in group:
                x = X[t, :, k:k + 1]
                # Shape: (n_channels, 1)

                # Hermitian SCM
                scm_accum += x @ x.conj().T

            scm = scm_accum / len(group)

            # Keep real part
            scm = np.real(scm)

            # Robust normalization
            denom = np.max(np.abs(scm))
            if denom < 1e-8:
                denom = 1e-8

            scm = scm / denom

            # Map from [-1, 1] to [0, 1]
            scm = (scm + 1.0) / 2.0

            cov_all_windows[t, b] = scm.astype(np.float32)

    return cov_all_windows


def compute_polynomial_scm(x, max_tau):
    """
    Compute polynomial space-time covariance matrices R[tau]
    for a multichannel signal.

    Parameters
    ----------
    x : ndarray, shape (M, N)
        Multichannel time series (M sensors/channels, N samples)
    max_tau : int
        Maximum lag (order of polynomial SCM)

    Returns
    -------
    R : ndarray, shape (max_tau+1, M, M)
        R[tau] = covariance matrix at lag tau
    """

    M, N = x.shape
    R = np.zeros((max_tau + 1, M, M), dtype=np.complex128)

    for tau in range(max_tau + 1):
        if N - tau <= 0:
            break

        # time-shifted data matrices
        x1 = x[:, tau:]      # x[n]
        x2 = x[:, :N - tau]  # x[n - tau]

        # lagged covariance estimate
        R_tau = (x1 @ x2.conj().T) / (N - tau)

        R[tau] = R_tau

    return R


def compute_scm_avg_1(signal, fs, freq_range, num_scms):
    """
    REPAIR-style SCM computation using GLOBAL FFT (no STFT).

    Differences from original:
    - keeps complex Hermitian SCMs
    - no magnitude operation
    - energy-consistent normalization (trace-based)
    - frequency-bin grouping like REPAIR
    """

    signal = np.asarray(signal)

    if signal.ndim != 2:
        raise ValueError("signal must have shape (n_channels, n_samples)")
    # --------------------------------
    # DC offset removal
    # --------------------------------
    signal = signal - np.mean(signal, axis=1, keepdims=True)

    n_channels, n_samples = signal.shape

    # -----------------------------
    # 1. GLOBAL FFT (REPAIR simplification)
    # -----------------------------
    X_f = np.fft.rfft(signal, axis=1)  # (n_channels, n_freqs)

    freqs = np.fft.rfftfreq(n_samples, d=1 / fs)

    # Frequency selection
    f_min, f_max = freq_range
    valid_idx = np.where((freqs >= f_min) & (freqs <= f_max))[0]

    if len(valid_idx) == 0:
        raise ValueError("No frequency bins found in frequency_range")

    X_f = X_f[:, valid_idx]

    # -----------------------------
    # 2. Frequency grouping (REPAIR-style)
    # -----------------------------
    freq_groups = np.array_split(np.arange(X_f.shape[1]), num_scms)

    cov_all_bands = np.zeros(
        (num_scms, n_channels, n_channels),
        dtype=np.complex64
    )

    # -----------------------------
    # 3. SCM computation per group
    # -----------------------------
    for b, group in enumerate(freq_groups):

        scm_accum = np.zeros((n_channels, n_channels), dtype=np.complex64)

        for k in group:
            x = X_f[:, k:k+1]  # (n_channels, 1)

            # Hermitian SCM (REPAIR core operation)
            scm_accum += x @ x.conj().T

        # Average over frequency bins
        scm = scm_accum / len(group)

        # -----------------------------
        # 4. REPAIR-style normalization (energy preserving)
        # -----------------------------
        cov_all_bands[b] = scm / (np.trace(scm).real + 1e-8)

    # -----------------------------
    # 5. Final normalization (0 to 1 per SCM)
    # -----------------------------
    cov_all_bands = np.real(cov_all_bands)
    
    for b in range(num_scms):
        scm = cov_all_bands[b]

        scm_min = scm.min()
        scm_max = scm.max()

        # Avoid division by zero
        if scm_max > scm_min:
            cov_all_bands[b] = (scm - scm_min) / (scm_max - scm_min)
        else:
            cov_all_bands[b] = np.zeros_like(scm)
    return cov_all_bands


def compute_scm_avg_2(signal, fs, freq_range, num_scms):
    """
    REPAIR-style SCM computation using GLOBAL FFT (no STFT).
    FIXED: ensures Hermitian + PSD covariance matrices
    """

    signal = np.asarray(signal)

    if signal.ndim != 2:
        raise ValueError("signal must have shape (n_channels, n_samples)")

    # DC offset removal
    signal = signal - np.mean(signal, axis=1, keepdims=True)

    n_channels, n_samples = signal.shape

    # -----------------------------
    # 1. GLOBAL FFT
    # -----------------------------
    X_f = np.fft.rfft(signal, axis=1)
    freqs = np.fft.rfftfreq(n_samples, d=1 / fs)

    # Frequency selection
    f_min, f_max = freq_range
    valid_idx = np.where((freqs >= f_min) & (freqs <= f_max))[0]

    if len(valid_idx) == 0:
        raise ValueError("No frequency bins found in frequency_range")

    X_f = X_f[:, valid_idx]

    # -----------------------------
    # 2. Frequency grouping
    # -----------------------------
    freq_groups = np.array_split(np.arange(X_f.shape[1]), num_scms)

    cov_all_bands = np.zeros(
        (num_scms, n_channels, n_channels),
        dtype=np.complex128  # keep precision
    )

    # -----------------------------
    # 3. SCM computation per band
    # -----------------------------
    for b, group in enumerate(freq_groups):

        if len(group) == 0:
            continue

        scm_accum = np.zeros((n_channels, n_channels), dtype=np.complex128)

        for k in group:
            x = X_f[:, k:k+1]
            scm_accum += x @ x.conj().T

        # average over frequency bins
        scm = scm_accum / len(group)

        # -----------------------------
        # FIX 1: enforce Hermitian structure
        # -----------------------------
        scm = (scm + scm.conj().T) / 2

        # -----------------------------
        # FIX 2: proper normalization (trace-based ONLY)
        # -----------------------------
        scm = scm / (np.trace(scm) + 1e-12)

        cov_all_bands[b] = scm

    return cov_all_bands

'''
# --- Convolve RIRs robust und in float32 ---
def convolve_rirs(rirs, sigs, rir_folder, fs):
    convolved = []
    for rir_row, sig in zip(rirs.itertuples(), sigs):
        rir_path = os.path.join(rir_folder, rir_row.Name)
        sr, rir = wavfile.read(rir_path)

        # --- Robust conversion ---
        if rir.dtype != np.float32:
            # int16 / int32 → float32 im Bereich [-1,1]
            if np.issubdtype(rir.dtype, np.integer):
                max_val = np.iinfo(rir.dtype).max
                rir = rir.astype(np.float32) / max_val
            else:
                rir = rir.astype(np.float32)

        # Sicherstellen: 2D (channels, samples)
        if rir.ndim == 1:
            rir = rir[:, None]

        if sr != fs:
            rir = resample_poly(rir, fs, sr, axis=0)

        # --- NORMALISIERUNG (global, erhält relative Peaks) ---
        #peak = np.max(np.abs(rir))
        #rir = rir / peak

        rir = rir.T  # shape: (channels, samples)

        sig = sig.astype(np.float32)  # Quelle ebenfalls float32

        # FFT-Convolve für jeden Kanal
        out = np.stack([fftconvolve(sig, rir_ch, mode='full') for rir_ch in rir], axis=0)
        convolved.append(out.astype(np.float32))
    return convolved
'''

def convolve_rirs(
    rirs,
    sigs,
    rir_folder,
    fs,
    rir_noise_db_range=None,
    rng=None,
    noisy_rir_output_folder=None,
    noisy_rir_prefix="noisy",
    fadeout_rel_threshold=1e-3,
    noise_fade_samples=100,
):
    if rng is None:
        rng = np.random.default_rng()

    loaded_rirs = []
    rir_names = []

    # =========================
    # Load RIRs with robust conversion
    # =========================
    for rir_row in rirs.itertuples():
        rir_path = os.path.join(rir_folder, rir_row.Name)
        sr, rir = wavfile.read(rir_path)

        rir_names.append(rir_row.Name)

        # --- Robust conversion ---
        if rir.dtype != np.float32:
            # int16 / int32 → float32 im Bereich [-1,1]
            if np.issubdtype(rir.dtype, np.integer):
                max_val = np.iinfo(rir.dtype).max
                rir = rir.astype(np.float32) / max_val
            else:
                rir = rir.astype(np.float32)

        # Sicherstellen: 2D (samples, channels)
        if rir.ndim == 1:
            rir = rir[:, None]

        if sr != fs:
            rir = resample_poly(rir, fs, sr, axis=0)

        # shape: (channels, samples)
        rir = rir.T.astype(np.float32)

        loaded_rirs.append(rir)

    # =========================
    # Add RIR noise before convolution
    # =========================
    if rir_noise_db_range is not None and len(loaded_rirs) > 0:
        peaks = [np.max(np.abs(rir)) for rir in loaded_rirs]
        quietest_peak = min(peaks)

        if quietest_peak > 0:
            db_low, db_high = rir_noise_db_range

            # Threshold below which the RIR is considered faded out.
            # Example: 1e-5 means -100 dB relative to the quietest selected RIR peak.
            fadeout_threshold = quietest_peak * fadeout_rel_threshold

            for i, rir in enumerate(loaded_rirs):
                # RIR shape: (channels, samples)
                if rir.ndim != 2:
                    raise ValueError(f"Expected 2D RIR, got shape {rir.shape}")

                # Active wherever at least one channel is above threshold.
                # This does NOT break on zero samples in the middle.
                active = np.any(np.abs(rir) > fadeout_threshold, axis=0)

                if not np.any(active):
                    continue

                # Last meaningful sample before fadeout
                last_active_idx = np.where(active)[0][-1] + 1

                noise_db = rng.uniform(db_low, db_high)

                # Noise is x-y dB quieter than the quietest selected RIR peak
                noise_amp = quietest_peak * 10.0 ** (-noise_db / 20.0)

                noise = rng.normal(
                    loc=0.0,
                    scale=noise_amp,
                    size=rir[:, :last_active_idx].shape,
                ).astype(np.float32)

                # =========================
                # Fade in / fade out added noise
                # =========================
                fade_len = min(noise_fade_samples, noise.shape[1])

                if fade_len > 1:
                    fade_in = np.linspace(
                        0.0,
                        1.0,
                        fade_len,
                        dtype=np.float32,
                    )

                    fade_out = np.linspace(
                        1.0,
                        0.0,
                        fade_len,
                        dtype=np.float32,
                    )

                    noise[:, :fade_len] *= fade_in[None, :]
                    noise[:, -fade_len:] *= fade_out[None, :]

                loaded_rirs[i][:, :last_active_idx] += noise

    # =========================
    # Export noisy RIRs
    # =========================
    if noisy_rir_output_folder is not None:
        os.makedirs(noisy_rir_output_folder, exist_ok=True)

        for rir_name, rir in zip(rir_names, loaded_rirs):
            base_name = os.path.splitext(os.path.basename(rir_name))[0]
            out_name = f"{noisy_rir_prefix}_{base_name}.wav"
            out_path = os.path.join(noisy_rir_output_folder, out_name)

            # wavfile.write expects shape (samples, channels)
            rir_to_write = rir.T.astype(np.float32)

            wavfile.write(out_path, fs, rir_to_write)

    # =========================
    # Convolution
    # =========================
    convolved = []

    for rir, sig in zip(loaded_rirs, sigs):
        sig = sig.astype(np.float32)

        out = np.stack(
            [fftconvolve(sig, rir_ch, mode="full") for rir_ch in rir],
            axis=0,
        )

        convolved.append(out.astype(np.float32))

    return convolved

def apply_order_weighting(cov_all_bands):
    n_bands, n_channels, _ = cov_all_bands.shape

    order_weights = {
        0: 1.0,
        1: 1.0,
        2: 0.75,
        3: 0.5,
        4: 0.25,
    }

    # Build channel-wise weight vector
    weights = np.ones(n_channels, dtype=np.float32)

    n_max = int(np.sqrt(n_channels) - 1)

    for n in range(n_max + 1):
        start = n**2
        end = (n+1)**2
        w = order_weights.get(n, 1.0)
        weights[start:end] = w

    # ---- process each band ----
    for b in range(n_bands):
        C = cov_all_bands[b].astype(np.float32)

        # 1) normalize to [-1, 1]
        c_min, c_max = C.min(), C.max()
        if c_max > c_min:
            C = 2.0 * (C - c_min) / (c_max - c_min) - 1.0
        else:
            C = np.zeros_like(C)

        # 2) apply weighting (W C W)
        C = weights[:, None] * C * weights[None, :]

        # 3) normalize to [0, 1]
        c_min, c_max = C.min(), C.max()
        if c_max > c_min:
            C = (C - c_min) / (c_max - c_min)
        else:
            C = np.zeros_like(C)

        cov_all_bands[b] = C

    return cov_all_bands



def estimate_rt60(rir, fs):
    rir = np.asarray(rir)
    rir = rir / np.max(np.abs(rir))
    rir = rir + np.random.randn(*rir.shape)*1e-9

    # --- select first channel if multi-channel ---
    if rir.ndim > 1:  # assume shape is (n_samples, n_channels)
        rir = rir[:, 0]

    # --- normalize (peak normalization) ---
    eps = 1e-12
    rir = rir / (np.max(np.abs(rir)) + eps)

    # --- energy decay curve ---
    energy = rir ** 2
    edc = np.cumsum(energy[::-1])[::-1]
    edc /= (np.max(edc) + eps)

    edc_db = 10 * np.log10(edc + eps)

    mask = (edc_db <= -5) & (edc_db >= -35)

    if np.sum(mask) < 10:
        return np.nan

    t = np.arange(len(rir)) / fs
    slope, _, _, _, _ = linregress(t[mask], edc_db[mask])

    return -60 / slope



def apply_order_highpass_filters(signal, fs, cutoff_freqs, slopes_db_per_oct):
    signal = signal.T
    signal = np.asarray(signal)
    dtype_in = signal.dtype
    signal = signal.astype(np.float64)

    if signal.ndim != 2:
        raise ValueError("signal must have shape (samples, channels).")

    n_samples, n_channels = signal.shape

    order_sht = int(np.sqrt(n_channels) - 1)

    if (order_sht + 1) ** 2 != n_channels:
        raise ValueError("Number of channels must be a square Ambisonics channel count.")

    cutoff_freqs = np.asarray(cutoff_freqs, dtype=float)
    slopes_db_per_oct = np.asarray(slopes_db_per_oct, dtype=float)

    if len(slopes_db_per_oct) != order_sht:
        raise ValueError("slopes_db_per_oct must have length Ambisonic order.")

    if len(cutoff_freqs) == order_sht:
        cutoff_freqs = np.r_[0.0, cutoff_freqs]
    elif len(cutoff_freqs) != order_sht + 1:
        raise ValueError("cutoff_freqs must have length order_sht or order_sht + 1.")

    filtered = signal.copy()

    for n in range(1, order_sht + 1):
        fc = cutoff_freqs[n]
        slope = slopes_db_per_oct[n - 1]

        if fc <= 0 or fc >= fs / 2:
            raise ValueError(f"Invalid cutoff frequency for order {n}: {fc}")

        # sosfiltfilt doubles the effective slope, so divide by 12 instead of 6
        hp_order = max(1, int(round(slope / 12)))

        sos = butter(
            hp_order,
            fc,
            btype="highpass",
            fs=fs,
            output="sos"
        )

        ch_start = n * n
        ch_end = (n + 1) * (n + 1)

        filtered[:, ch_start:ch_end] = sosfiltfilt(
            sos,
            filtered[:, ch_start:ch_end],
            axis=0
        )

    return filtered.T.astype(dtype_in)



def rigid_sphere_modal_coeffs(max_order, x):
    b = np.zeros((len(x), max_order + 1), dtype=complex)

    for n in range(max_order + 1):
        jn = spherical_jn(n, x)
        jn_der = spherical_jn(n, x, derivative=True)

        yn = spherical_yn(n, x)
        yn_der = spherical_yn(n, x, derivative=True)

        hn2 = jn - 1j * yn
        hn2_der = jn_der - 1j * yn_der

        b[:, n] = 4 * np.pi * (1j ** n) * (
            jn - (jn_der / hn2_der) * hn2
        )

    return b

def apply_tikhonov_difference_filters(
    signal,
    fs,
    R=0.042,
    Nmic=32,
    amp_threshold=15,
    c=343.0
):
    
    """
    Apply order-dependent Moreau/Daniel difference filters to Ambisonic signals.

    Parameters
    ----------
    signal : ndarray
        Ambisonic signal with shape (channels, samples).
    fs : float
        Sampling rate in Hz.
    R : float
        Microphone array radius in meters.
    Nmic : int
        Number of microphones.
    amp_threshold : float
        Regularization threshold in dB.
    c : float
        Speed of sound in m/s.

    Returns
    -------
    filtered : ndarray
        Filtered Ambisonic signal with shape (channels, samples).
    """

    signal = signal.T
    signal = np.asarray(signal)
    dtype_in = signal.dtype
    signal = signal.astype(np.float64)

    if signal.ndim != 2:
        raise ValueError("signal must have shape (channels, samples).")

    n_samples, n_channels = signal.shape

    order_sht = int(np.sqrt(n_channels) - 1)

    if (order_sht + 1) ** 2 != n_channels:
        raise ValueError("Number of channels must be a square Ambisonics channel count.")

    # -----------------------------
    # Frequency axis for FFT bins
    # -----------------------------
    f = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    kR = 2 * np.pi * f * R / c

    kR_safe = kR.copy()
    kR_safe[0] = 1e-12

    # -----------------------------
    # Rigid-sphere modal coefficients
    # -----------------------------
    bN = rigid_sphere_modal_coeffs(order_sht, kR_safe) / (4 * np.pi)

    # DC handling
    bN[0, 1:] = 0.0

    # -----------------------------
    # Moreau/Daniel regularization
    # -----------------------------
    alpha = np.sqrt(Nmic) * 10 ** (amp_threshold / 20)

    if alpha <= 1:
        raise ValueError("amp_threshold is too low: alpha must be greater than 1.")

    beta = np.sqrt(
        (1 - np.sqrt(1 - 1 / alpha**2)) /
        (1 + np.sqrt(1 - 1 / alpha**2))
    )

    # -----------------------------
    # Difference filters
    #
    # H_reg = conj(bN) / (abs(bN)^2 + beta^2)
    # H_opt = 1 / bN
    #
    # Difference filter:
    # H_diff = H_reg / H_opt
    #        = H_reg * bN
    #        = abs(bN)^2 / (abs(bN)^2 + beta^2)
    # -----------------------------
    H_diff = np.abs(bN) ** 2 / (np.abs(bN) ** 2 + beta**2)

    # Avoid filtering order 0 at DC incorrectly
    H_diff[0, 0] = 1.0

    # -----------------------------
    # Apply filters in frequency domain
    # -----------------------------
    X = np.fft.rfft(signal, axis=0)

    for n in range(order_sht + 1):
        ch_start = n * n
        ch_end = (n + 1) * (n + 1)

        X[:, ch_start:ch_end] *= H_diff[:, n, None]

    filtered = np.fft.irfft(X, n=n_samples, axis=0)

    return filtered.T.astype(dtype_in)




def add_noise_to_rirs_until_fadeout(
    rirs,
    noise_db_range=(40, 60),
    rng=None,
    fadeout_rel_threshold=1e-5,
):
    """
    Add white noise to selected RIRs before convolution.

    Noise level is chosen independently per RIR and is x-y dB quieter than
    the peak of the quietest selected RIR.

    Noise is only added until the RIR has faded out, defined as the last sample
    where at least one channel is larger than a small threshold.

    Parameters
    ----------
    rirs : list[np.ndarray]
        List of RIR arrays with shape (channels, samples).
    noise_db_range : tuple[float, float]
        Range (x, y) in dB. Noise will be x-y dB quieter than the quietest RIR peak.
    rng : np.random.Generator or None
        Random generator for deterministic behavior.
    fadeout_rel_threshold : float
        Relative threshold for detecting RIR activity.
        Example: 1e-5 means -100 dB relative to quietest selected RIR peak.

    Returns
    -------
    noisy_rirs : list[np.ndarray]
        RIRs with added noise.
    """
    if rng is None:
        rng = np.random.default_rng()

    if len(rirs) == 0:
        return rirs

    peaks = [np.max(np.abs(rir)) for rir in rirs]
    quietest_peak = min(peaks)

    if quietest_peak <= 0:
        return rirs

    db_low, db_high = noise_db_range
    fadeout_threshold = quietest_peak * fadeout_rel_threshold

    noisy_rirs = []

    for rir in rirs:
        rir_noisy = rir.astype(np.float32, copy=True)

        if rir_noisy.ndim != 2:
            raise ValueError(f"Expected 2D RIR, got shape {rir_noisy.shape}")

        # Active until at least one channel is above threshold
        active = np.any(np.abs(rir_noisy) > fadeout_threshold, axis=0)

        if not np.any(active):
            noisy_rirs.append(rir_noisy)
            continue

        last_active_idx = np.where(active)[0][-1] + 1

        noise_db = rng.uniform(db_low, db_high)
        noise_amp = quietest_peak * 10.0 ** (-noise_db / 20.0)

        noise = rng.normal(
            loc=0.0,
            scale=noise_amp,
            size=rir_noisy[:, :last_active_idx].shape,
        ).astype(np.float32)

        rir_noisy[:, :last_active_idx] += noise
        noisy_rirs.append(rir_noisy)

    return noisy_rirs