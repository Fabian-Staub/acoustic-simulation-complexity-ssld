import torch
import matplotlib.pyplot as plt
import os
import numpy as np

sample_idx = 0  # which sample to inspect
data = torch.load("../../../data/harp/R800_P10_S5_Order4_24kHz_shoeBox_ISM/harp_dataset.pt", weights_only=False)
X = data["X"]  # shape: (num_samples, n_channels, n_mels, n_frames)
Y = data["Y"]  # shape: (num_samples, 5, metadata_len)

print("X shape:", X.shape)
print("Y shape:", Y.shape)

# Print Y metadata for this sample
print("\nY metadata for sample", sample_idx)
print(Y[sample_idx])

# Create output folder
OUTPUT_PLOTS_DIR = "../../../data/harp/R800_P10_S5_Order4_24kHz_shoeBox_ISM/logmel_plots"
os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)

sample_X = X[sample_idx]  # shape: (n_channels, n_mels, n_frames)
n_channels, n_mels, n_frames = sample_X.shape

plt.figure(figsize=(15, 3*n_channels))

for ch in range(n_channels):
    plt.subplot(n_channels, 1, ch+1)
    # Convert to numpy
    S_db = sample_X[ch].numpy() if isinstance(sample_X[ch], torch.Tensor) else sample_X[ch]
    # Display log-mel spectrogram
    plt.imshow(S_db, aspect='auto', origin='lower', cmap='magma')
    plt.colorbar(label='magnitude (log scale)')
    plt.ylabel(f'Ch {ch+1}')
    if ch == 0:
        plt.title(f'Log-Mel Spectrograms for Sample {sample_idx}')
    if ch == n_channels-1:
        plt.xlabel('Frames')

plt.tight_layout()
plt.savefig(f"{OUTPUT_PLOTS_DIR}/sample{sample_idx}_logmel.png", dpi=300)
plt.close()

print(f"Saved log-mel spectrogram plot to {OUTPUT_PLOTS_DIR}/sample{sample_idx}_logmel.png")