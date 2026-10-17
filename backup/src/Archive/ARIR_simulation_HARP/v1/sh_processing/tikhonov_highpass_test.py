import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.io import wavfile

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

    return filtered.astype(dtype_in)

infile = "/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/harp/test/R0_P0_S0_IR.wav"
outfile = "/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/harp/test/R0_P0_S0_IR_hpfiltered.wav"

fs, x = wavfile.read(infile)
x = x.T

x_hp = apply_order_highpass_filters(
    signal=x,
    fs=fs,
    cutoff_freqs=(40, 490, 1200, 2200),
    slopes_db_per_oct=(5, 15, 20, 24)
)

wavfile.write(outfile, fs, x_hp)