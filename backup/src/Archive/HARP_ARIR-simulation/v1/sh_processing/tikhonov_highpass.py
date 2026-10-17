import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spherical_jn, spherical_yn

# -----------------------------
# Parameters
# -----------------------------
R = 0.042
Nmic = 32
order_sht = 4
Lfilt = 4096
fs = 48000
amp_threshold = 15  # dB

# Individual highpass slopes for orders 1..order_sht
hp_slopes_db_per_oct = np.array([5, 15, 20, 24])  # dB/oct

c = 343.0

if len(hp_slopes_db_per_oct) != order_sht:
    raise ValueError("hp_slopes_db_per_oct must have length order_sht.")

# -----------------------------
# Frequency axis
# -----------------------------
f = np.linspace(0, fs / 2, Lfilt // 2 + 1)
kR = 2 * np.pi * f * R / c

kR_safe = kR.copy()
kR_safe[0] = 1e-12

# -----------------------------
# Rigid-sphere modal coefficients
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
# Highpass filters
# Cutoff frequency = frequency of maximum radial inverse response
# Individual slope per highpass filter
# -----------------------------
HP = np.ones_like(H_filt, dtype=float)
cutoff_freqs = np.zeros(order_sht + 1)

f_min = 20
search_idx = f >= f_min

for n in range(1, order_sht + 1):
    f_peak = f[search_idx][np.argmax(H_mag_db[search_idx, n])]
    cutoff_freqs[n] = f_peak

    hp_slope = hp_slopes_db_per_oct[n - 1]
    hp_order = hp_slope / 6  # 6 dB/oct per Butterworth order

    HP[:, n] = (f / f_peak) ** hp_order / np.sqrt(
        1 + (f / f_peak) ** (2 * hp_order)
    )

# DC handling for highpass
HP[0, 1:] = 0.0

HP_mag_db = 20 * np.log10(np.maximum(HP, 1e-12))

# Apply highpass filters to radial inverse filters
H_filt_hp = H_filt * HP
H_filt_hp_mag_db = 20 * np.log10(np.maximum(np.abs(H_filt_hp), 1e-12))

# -----------------------------
# Print cutoff frequencies
# -----------------------------
print("Highpass cutoff frequencies:")
for n in range(1, order_sht + 1):
    print(
        f"Order {n}: {cutoff_freqs[n]:.1f} Hz, "
        f"slope = {hp_slopes_db_per_oct[n - 1]:.1f} dB/oct"
    )

# -----------------------------
# Plot radial inverse filters and highpass filters together
# -----------------------------
idx = f >= f_min

plt.figure(figsize=(10, 6))

for n in range(order_sht + 1):
    plt.semilogx(
        f[idx],
        H_mag_db[idx, n],
        label=f"Radial Inv Order {n}"
    )

for n in range(1, order_sht + 1):
    plt.semilogx(
        f[idx],
        HP_mag_db[idx, n],
        "--",
        label=(
            f"HP Order {n}, "
            f"fc={cutoff_freqs[n]:.0f} Hz, "
            f"slope={hp_slopes_db_per_oct[n - 1]:.0f} dB/oct"
        )
    )

plt.xlim(f_min, 20000)
plt.ylim(-80, 40)
plt.grid(True, which="both", alpha=0.3)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude [dB]")
plt.title("Radial Inverse Filters and Individual Highpass Filters")
plt.legend()
plt.tight_layout()
plt.savefig("radial_filters_and_highpass_filters.png", dpi=300)
plt.show()