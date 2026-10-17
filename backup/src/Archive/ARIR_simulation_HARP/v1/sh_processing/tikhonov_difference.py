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

c = 343.0

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

# -----------------------------
# Radial inverse filters
# -----------------------------

# Regularized radial filters
H_reg = np.conj(bN) / (np.abs(bN) ** 2 + beta**2)

# Optimal radial filters, no regularization
eps = 1e-12
H_opt = 1.0 / np.where(np.abs(bN) < eps, eps, bN)

# Magnitudes in dB
H_reg_db = 20 * np.log10(np.maximum(np.abs(H_reg), eps))
H_opt_db = 20 * np.log10(np.maximum(np.abs(H_opt), eps))

# Difference in dB
H_diff_db = H_reg_db - H_opt_db

# -----------------------------
# Plot 20 Hz - 20 kHz
# -----------------------------
f_min = 20
f_max = 20000
idx = (f >= f_min) & (f <= f_max)

fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)

for n in range(order_sht + 1):
    axes[0].semilogx(f[idx], H_reg_db[idx, n], label=f"Order {n}")
    axes[1].semilogx(f[idx], H_opt_db[idx, n], label=f"Order {n}")
    axes[2].semilogx(f[idx], H_diff_db[idx, n], label=f"Order {n}")

axes[0].set_title("Regularized Radial Inverse Filters")
axes[0].set_ylabel("Magnitude [dB]")
axes[0].set_ylim(-40, 40)

axes[1].set_title("Optimal Radial Inverse Filters, No Regularization")
axes[1].set_ylabel("Magnitude [dB]")
axes[1].set_ylim(-40, 80)

axes[2].set_title("Difference: Regularized - Optimal")
axes[2].set_xlabel("Frequency [Hz]")
axes[2].set_ylabel("Difference [dB]")
axes[2].set_ylim(-80, 20)

for ax in axes:
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")

axes[2].set_xlim(f_min, f_max)

plt.tight_layout()
plt.savefig("radial_filters_comparison.png", dpi=300)
plt.close()