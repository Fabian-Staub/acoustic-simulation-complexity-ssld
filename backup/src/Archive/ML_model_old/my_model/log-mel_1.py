import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.io import wavfile

# ==============================
# Parameters
# ==============================
order = 4
numSH = (order + 1) ** 2  # 25 channels
FS = 24000
duration_sec = 1  # Analyze first 0.1 seconds

OUTPUT_PLOTS_DIR = "../../data/plots"
WAV_FILE = "../../data/harp/Generated_HOA_IRs_ISM/R1_P0_S0_IR.wav"
os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)

# ==============================
# Load ambisonics signal
# ==============================
fs, ambi = wavfile.read(WAV_FILE)
assert fs == FS, f"Expected {FS} Hz but got {fs} Hz"

# Select first 25 channels and transpose -> (channels, samples)
ambi = ambi[:int(duration_sec * FS), :numSH].T
print("Loaded WAV with shape:", ambi.shape)

# ==============================
# STFT / Mel parameters
# ==============================
win_length = int(0.040 * FS)   # 40 ms
hop_length = int(0.020 * FS)   # 20 ms
n_fft = 2048                   # >= win_length (safe choice)
n_mels = 64

# ==============================
# Compute and plot log-mel spectrograms
# ==============================
fig, axes = plt.subplots(5, 5, figsize=(20, 18))
axes = axes.flatten()

for ch in range(numSH):
    signal = ambi[ch].astype(np.float32)

    mel_spec = librosa.feature.melspectrogram(
        y=signal,
        sr=FS,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window="hann",
        n_mels=n_mels,
        power=2.0
    )

    log_mel = librosa.power_to_db(mel_spec, ref=np.max)

    ax = axes[ch]
    
    # Convert mel bins to Hz for axis scaling
    mel_frequencies = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=FS/2)

    duration = signal.shape[0] / FS

    ax.imshow(
        log_mel,
        aspect='auto',
        origin='lower',
        interpolation='nearest',
        extent=[0, duration, mel_frequencies[0], mel_frequencies[-1]]
    )

    ax.set_xlabel("Time (s)")

    ax.set_ylabel("Frequency (Hz)")

    ax.set_title(f'Channel {ch}')
    ax.label_outer()

plt.suptitle("Log-Mel Spectrograms (25 Ambisonics Channels)", fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.96])

output_path = os.path.join(OUTPUT_PLOTS_DIR, "logmel_25ch.png")
plt.savefig(output_path, dpi=300)
plt.close(fig)

print("Saved plot to:", output_path)