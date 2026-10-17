import numpy as np
import soundfile as sf
from scipy.special import spherical_jn, spherical_yn
from scipy.interpolate import interp1d

# -----------------------------
# Input / output
# -----------------------------
infile = "/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/harp/test/R0_P0_S0_IR.wav"
outfile = "/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/harp/test/R0_P0_S0_IR_hpfiltered.wav"

# -----------------------------
# Parameters
# -----------------------------
R = 0.042
Nmic = 32
order_sht = 4
Lfilt = 4096
fs_expected = 24000
amp_threshold = 15
c = 343.0

# Individual highpass slopes for orders 1..order_sht
hp_slopes_db_per_oct = np.array([5, 15, 20, 24])

if len(hp_slopes_db_per_oct) != order_sht:
    raise ValueError("hp_slopes_db_per_oct must have length order_sht.")

# -----------------------------
# Read audio
# -----------------------------
x, fs = sf.read(infile, always_2d=True)

if fs != fs_expected:
    raise ValueError(f"Expected fs={fs_expected}, got fs={fs}")

n_samples, n_channels = x.shape
expected_channels = (order_sht + 1) ** 2

if n_channels < expected_channels:
    raise ValueError(
        f"Expected at least {expected_channels} Ambisonics channels, got {n_channels}"
    )

old_peak = np.max(np.abs(x))

# -----------------------------
# Frequency axis
# -----------------------------
f_filt = np.linspace(0, fs / 2, Lfilt // 2 + 1)
kR = 2 * np.pi * f_filt * R / c
kR_safe = kR.copy()
kR_safe[0] = 1e-12

# -----------------------------
# Rigid-sphere modal coefficients
# Used only to find cutoff frequencies
# -----------------------------
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


bN = rigid_sphere_modal_coeffs(order_sht, kR_safe) / (4 * np.pi)

# DC handling
bN[0, 1:] = 0.0

# -----------------------------
# Moreau/Daniel regularization
# Used only to find peak frequencies
# -----------------------------
alpha = np.sqrt(Nmic) * 10 ** (amp_threshold / 20)

if alpha <= 1:
    raise ValueError("amp_threshold is too low: alpha must be greater than 1.")

beta = np.sqrt(
    (1 - np.sqrt(1 - 1 / alpha**2)) /
    (1 + np.sqrt(1 - 1 / alpha**2))
)

H_filt = np.conj(bN) / (np.abs(bN) ** 2 + beta**2)
H_mag_db = 20 * np.log10(np.maximum(np.abs(H_filt), 1e-12))

# -----------------------------
# Build highpass magnitude filters only
# Order 0 is left unchanged
# Orders 1..order_sht get individual HP filters
# -----------------------------
HP = np.ones((len(f_filt), order_sht + 1), dtype=float)
cutoff_freqs = np.zeros(order_sht + 1)

f_min = 20.0
search_idx = f_filt >= f_min

for n in range(1, order_sht + 1):
    f_peak = f_filt[search_idx][np.argmax(H_mag_db[search_idx, n])]
    cutoff_freqs[n] = f_peak

    hp_slope = hp_slopes_db_per_oct[n - 1]
    hp_order = hp_slope / 6.0

    ratio = f_filt / f_peak

    HP[:, n] = ratio ** hp_order / np.sqrt(
        1.0 + ratio ** (2.0 * hp_order)
    )

# DC handling
HP[0, 1:] = 0.0

# -----------------------------
# Apply highpass magnitude filters by Ambisonic order
# Assumes ACN channel ordering:
# order 0: ch 0
# order 1: ch 1-3
# order 2: ch 4-8
# order 3: ch 9-15
# order 4: ch 16-24
# -----------------------------
X = np.fft.rfft(x, axis=0)
f_audio = np.fft.rfftfreq(n_samples, 1 / fs)

Y = X.copy()

for n in range(order_sht + 1):
    interp_hp = interp1d(
        f_filt,
        HP[:, n],
        kind="linear",
        bounds_error=False,
        fill_value=(HP[0, n], HP[-1, n]),
    )

    Hn = interp_hp(f_audio)

    ch_start = n * n
    ch_end = (n + 1) * (n + 1)

    Y[:, ch_start:ch_end] *= Hn[:, None]

y = np.fft.irfft(Y, n=n_samples, axis=0)

# -----------------------------
# Match original peak
# -----------------------------
new_peak = np.max(np.abs(y))

if old_peak > 0 and new_peak > 0:
    y *= old_peak / new_peak

# -----------------------------
# Save
# -----------------------------
sf.write(outfile, y, fs)

print(f"Saved: {outfile}")
print(f"Original peak: {old_peak:.8f}")
print(f"New peak:      {np.max(np.abs(y)):.8f}")

print("Highpass cutoff frequencies:")
for n in range(1, order_sht + 1):
    print(
        f"Order {n}: {cutoff_freqs[n]:.1f} Hz, "
        f"slope = {hp_slopes_db_per_oct[n - 1]:.1f} dB/oct"
    )