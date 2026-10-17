import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spherical_jn, spherical_yn

# -----------------------------
# Parameters
# -----------------------------
R = 0.042          # sphere radius in meters, adjust as needed
Nmic = 32          # number of microphones
order_sht = 4
Lfilt = 4096
fs = 48000
amp_threshold = 15 # dB

c = 343.0

# -----------------------------
# Frequency axis
# -----------------------------
f = np.linspace(0, fs / 2, Lfilt // 2 + 1)
kR = 2 * np.pi * f * R / c

# Avoid exact zero for stable Hankel derivative evaluation
kR_safe = kR.copy()
kR_safe[0] = 1e-12

# -----------------------------
# Rigid-sphere modal coefficients
# b_n(kR)
#
# Convention:
# b_n = 4*pi*i^n * [ j_n(kR)
#        - j_n'(kR) / h_n^(2)'(kR) * h_n^(2)(kR) ]
#
# MATLAB code divides sphModalCoeffs(...) by 4*pi,
# so we do the same here.
# -----------------------------
def rigid_sphere_modal_coeffs(max_order, x):
    b = np.zeros((len(x), max_order + 1), dtype=complex)

    for n in range(max_order + 1):
        jn = spherical_jn(n, x)
        jn_der = spherical_jn(n, x, derivative=True)

        yn = spherical_yn(n, x)
        yn_der = spherical_yn(n, x, derivative=True)

        # Spherical Hankel function of the second kind
        hn2 = jn - 1j * yn
        hn2_der = jn_der - 1j * yn_der

        b[:, n] = 4 * np.pi * (1j ** n) * (
            jn - (jn_der / hn2_der) * hn2
        )

    return b


bN = rigid_sphere_modal_coeffs(order_sht, kR_safe) / (4 * np.pi)

# Fix DC where numerical Hankel terms can be awkward
# Only order 0 has meaningful low-frequency response.
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

# Regularized radial inverse filters
H_filt = np.conj(bN) / (np.abs(bN) ** 2 + beta**2)

# Magnitude in dB
H_mag_db = 20 * np.log10(np.maximum(np.abs(H_filt), 1e-12))

# -----------------------------
# Plot 0-20 kHz
# -----------------------------
# -----------------------------
# Plot 20 Hz-20 kHz, logarithmic x axis
# -----------------------------
plt.figure(figsize=(10, 6))

f_min = 20
idx = f >= f_min

for n in range(order_sht + 1):
    plt.semilogx(f[idx], H_mag_db[idx, n], label=f"Order {n}")

plt.xlim(f_min, 20000)
plt.ylim(-40, 40)
plt.grid(True, which="both", alpha=0.3)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude [dB]")
plt.title("Regularized Radial Inverse Filters, Rigid Sphere")
plt.legend()
plt.tight_layout()
plt.savefig("radial_filters.png")