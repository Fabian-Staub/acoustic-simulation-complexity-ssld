import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from utils import compute_scm_avg_log, apply_tikhonov_difference_filters

fs, signal = wavfile.read("/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/harp/R2000_P10_S5_Order4_24kHz_shoeBox_rtProfilesMIT_Direct/RIRs/R0_P0_S0_IR.wav")
signal = signal.T
signal = apply_tikhonov_difference_filters(signal=signal, fs=fs)

scm = compute_scm_avg_log(signal, fs, (2000, 8000), 10)

n = scm.shape[0]

fig, axes = plt.subplots(1, n, figsize=(5*n, 4))

if n == 1:
    axes = [axes]

for i in range(n):
    im = axes[i].imshow(scm[i], aspect='auto', origin='lower', cmap='viridis')
    axes[i].set_title(f"SCM {i}")
    axes[i].set_xlabel("Time")
    axes[i].set_ylabel("Freq bin")
    plt.colorbar(im, ax=axes[i])

plt.tight_layout()
plt.savefig("all_scms.png", dpi=300)
plt.show()