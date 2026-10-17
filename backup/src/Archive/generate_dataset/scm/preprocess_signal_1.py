import os
import numpy as np
import pyfar as pf
import matplotlib.pyplot as plt
from scipy.io import wavfile
import spaudiopy as spa
import torch

# ==============================
# Parameters
# ==============================
order = 4
numSH = (order + 1) ** 2  # channels
FS = 480000

OUTPUT_PLOTS_DIR = "../data/plots"
WAV_FILE = "../data/output/ambisonic_output.wav"
os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)



# ==============================
# Generate Test Noise Signal from front
# ==============================

TARGET_ORDER = 4
azi = 0
elev = 0
polar = np.pi / 2 - elev
Y_src = spa.sph.sh_matrix(TARGET_ORDER, azi, polar, sh_type="real")
num_samples = 48000  # 1 second at 48 kHz
sig = np.random.randn(num_samples)
ambi = np.outer(Y_src, sig)
print(ambi)
'''


# ==============================
# Load ambisonics signal from file
# ==============================
fs, ambi = wavfile.read(WAV_FILE)
ambi = ambi[:, :numSH].T  # select first numSH channels and transpose
print("Loaded WAV with shape:", ambi.shape)
'''

# Create signal object for pyfar
signal_full = pf.Signal(ambi, FS)
print("Signal object shape:", signal_full.time.shape)


# ==============================
# Filter signal with octave filter bank
# ==============================
signal_bands = pf.dsp.filter.fractional_octave_bands(
    signal_full,
    num_fractions=1,          # 1/1 octave
    frequency_range=(100, 8000),
    order=15
)
print("Filtered signal shape:", signal_bands.time.shape)


'''
# ==============================
# FFT AND AVERAGE MAGNITUDE PER BAND/CHANNEL
# ==============================
N = signal_bands.time.shape[-1]

fft_vals = np.fft.rfft(signal_bands.time, axis=-1)
amp = np.abs(fft_vals)

avg_amp = amp.mean(axis=-1)  # shape: (n_bands, n_channels)
print("Average Amplitude shape:", avg_amp.shape)

'''

# ==============================
# Amplitude average per bands for every channel
# ==============================
# signal_bands.time shape: (n_bands, n_channels, n_samples)

# Compute instantaneous power
amp = (signal_bands.time ** 2) ** 0.5

# Average power over time
avg_amp = amp.mean(axis=-1)  # shape: (n_bands, n_channels)

print("Average Amplitude shape:", avg_amp.shape)


# ==============================
# Save dataset as torch tensors for later use in training
# ==============================

n_bands, n_channels, n_samples = signal_bands.time.shape

cov_all_bands = np.zeros((n_bands, n_channels, n_channels))
for b in range(n_bands):
    cov_all_bands[b] = np.outer(avg_amp[b], avg_amp[b])
cov_all_bands = cov_all_bands / np.max(np.abs(cov_all_bands))


# Example: 3 sources
azi = np.array([0.0, np.pi/4, -np.pi/2])      # azimuths in radians
el  = np.array([0.0, np.pi/6, -np.pi/6])     # elevations in radians
active = np.array([1, 0, 1])                 # 1=active, 0=inactive

def azel_to_accdoa(azi, el, active):
    """
    Convert azimuth/elevation to ACCDOA vectors.
    Length = 1 if active, 0 if inactive.
    """
    x = np.cos(el) * np.cos(azi)
    y = np.cos(el) * np.sin(azi)
    z = np.sin(el)
    vec = np.stack([x, y, z], axis=-1)
    
    # Normalize (unit vector) just in case
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    vec /= norm + 1e-8
    
    # Apply activity (length=1 or 0)
    vec *= active[:, None]
    
    return vec

accdoa_targets = azel_to_accdoa(azi, el, active)
print("ACC-DOA vectors:\n", accdoa_targets)

torch.save({
    "X": torch.tensor(cov_all_bands, dtype=torch.float32),
    "Y": torch.tensor(accdoa_targets, dtype=torch.float32)
}, "dataset.pt")

data = torch.load("dataset.pt")
print(data.shape)


# ==============================
# Plot Spacial Covariance Matrix per frequency band
# ==============================

for b in range(n_bands):
    cov_matrix = np.outer(avg_amp[b, :], avg_amp[b, :])
    
    plt.figure(figsize=(6,5))
    plt.imshow(cov_matrix, cmap='viridis', interpolation='none')
    plt.colorbar(label='Covariance-like')
    plt.gca().invert_yaxis()
    plt.title(f'Covariance Matrix - Band {b+1}')
    plt.xlabel('Channel')
    plt.ylabel('Channel')
    
    plt.savefig(f"{OUTPUT_PLOTS_DIR}/cov_band_{b+1}.png", dpi=300)
    plt.close()


print(f"Saved {n_bands} covariance plots to {OUTPUT_PLOTS_DIR}")
