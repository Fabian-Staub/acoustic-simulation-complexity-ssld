import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

import pyfar as pf
import pyrato as ra
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from utils import estimate_rt60

# -----------------------------
# Load RIR
# -----------------------------
fs, rir = wavfile.read(
    "/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/arni_em32/sh_wavs/RIRs_truncated/dry_P1_S1.wav"
)


# Ensure correct shape and dtype
rir = rir.T.astype(np.float64)
rir = rir[0]

print("shape:", rir.shape)
print("min:", np.min(rir))
print("max:", np.max(rir))
print("finite:", np.all(np.isfinite(rir)))

# Convert to dB safely
rir_db = 10 * np.log10(np.abs(rir)**2 + 1e-12)

# Convert to pyfar signal
rir_sig = pf.Signal(rir, fs)

# Lundeby intersection
inter_time, rt60, noise_power = ra.edc.intersection_time_lundeby(rir_sig)

# -----------------------------
# ONE FIGURE, MULTIPLE SUBPLOTS
# -----------------------------
fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=False)

# --- 1. Linear RIR ---
axs[0].plot(rir)
axs[0].set_title("RIR (linear)")
axs[0].set_ylabel("Amplitude")
axs[0].grid(True)

# --- 2. RIR in dB ---
axs[1].plot(rir_db)
axs[1].set_title("RIR (energy in dB)")
axs[1].set_ylabel("dB")
axs[1].grid(True)

# --- 3. Pyfar + EDC intersection ---
pf.plot.time(rir_sig, dB=True, ax=axs[2])

axs[2].axvline(inter_time, c='k', linestyle='--', label='$T_i$')
axs[2].axhline(
    10 * np.log10(noise_power),
    c='k',
    linestyle=':',
    label='Noise floor'
)

axs[2].set_title("RIR with Lundeby intersection")
axs[2].legend()
axs[2].grid(True)

# -----------------------------
# Save ONE combined image
# -----------------------------
plt.tight_layout()
plt.savefig("rir_all_plots.png", dpi=300, bbox_inches="tight")
plt.close()



