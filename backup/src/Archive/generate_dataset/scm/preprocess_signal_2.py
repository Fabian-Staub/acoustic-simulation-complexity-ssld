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
azi  = np.array([0.0, np.pi/2])
el   = np.array([0.0, 0.0])
num_sources = len(azi)
#azi  = 0
#elev = 0
polar = np.pi / 2 - el
Y_src = spa.sph.sh_matrix(TARGET_ORDER, azi, polar, sh_type="real")
num_samples = 48000  # 1 second at 48 kHz
sig = np.random.randn(num_sources, num_samples)
ambi = Y_src.T @ sig
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
print("Covariance-like matrices shape:", cov_all_bands.shape)


# Example: 3 sources
#azi = np.array([0.0, np.pi/4])      # azimuths in radians
#el  = np.array([0.0, np.pi/6])     # elevations in radians

def azel_to_accdoa(azi_list, el_list, max_sources=3):
    """
    Convert lists of az/el to fixed-size ACCDOA array.
    Automatically fills remaining slots with [0,0,0].
    
    Parameters:
    - azi_list: azimuths in radians, shape (n_sources,)
    - el_list: elevations in radians, shape (n_sources,)
    - max_sources: number of sources to always encode
    
    Returns:
    - accdoa: shape (max_sources, 3)
    """
    accdoa = np.zeros((max_sources, 3), dtype=np.float32)  # default all zeros
    
    n = min(len(azi_list), max_sources)
    
    for i in range(n):
        x = np.cos(el_list[i]) * np.cos(azi_list[i])
        y = np.cos(el_list[i]) * np.sin(azi_list[i])
        z = np.sin(el_list[i])
        vec = np.array([x, y, z], dtype=np.float32)
        vec /= np.linalg.norm(vec) + 1e-8  # normalize to length 1
        accdoa[i] = vec  # fill into fixed array
    
    return accdoa

accdoa_targets = azel_to_accdoa(azi, el)
print("ACC-DOA vectors:\n", accdoa_targets)

# Suppose cov_all_bands is a single sample of shape (7,25,25)
X = torch.tensor(cov_all_bands, dtype=torch.float32).unsqueeze(0)  # shape: (1,7,25,25)
Y = torch.tensor(accdoa_targets, dtype=torch.float32).unsqueeze(0)  # shape: (1,7,25,25) or whatever

# Duplicate along batch dimension
X_double = torch.cat([X, X], dim=0)  # now shape (2,7,25,25)
Y_double = torch.cat([Y, Y], dim=0)  # shape (2,...)


# Save to .pt
torch.save({
    "X": X_double,
    "Y": Y_double
}, "dataset.pt")

data = torch.load("dataset.pt")
X_loaded = data["X"]
Y_loaded = data["Y"]
print("Loaded X shape:", X_loaded.shape)  # (2, n_bands, n_channels, n_channels)
print("Loaded Y shape:", Y_loaded.shape)  # (2, max_sources,


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
