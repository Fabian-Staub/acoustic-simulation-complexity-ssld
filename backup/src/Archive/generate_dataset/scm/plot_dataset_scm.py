import torch
import matplotlib.pyplot as plt
import os

# =========================
# SETTINGS
# =========================
#DATA_PATH = "../../../data/arni_em32/sh_wavs/arni_dataset_2-7kHz_10-SCM_multiSource.pt"
#OUTPUT_PLOTS_DIR = "../../../data/arni_em32/sh_wavs/covariance_plots/2-7kHz_10-SCM_multiSource"
DATA_PATH = "/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/harp/R2000_P10_S5_Order4_24kHz_shoeBox_rtProfilesMIT_ISM+Late/harp_dataset_2-8kHz_highpass_10-SCM_randRot_10-repeats_noisy.pt"
OUTPUT_PLOTS_DIR = "/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/harp/R2000_P10_S5_Order4_24kHz_shoeBox_rtProfilesMIT_ISM+Late/covariance_plots/2-8kHz_10-SCM_noisy"

start_index = 0         # <-- NEW: where to start
num_samples_to_plot = 10      # how many samples to plot

# =========================
# LOAD DATA
# =========================
data = torch.load(DATA_PATH, weights_only=False)
X = data["X"]
Y = data["Y"]

print("X shape:", X.shape)
print("Y shape:", Y.shape)

# Create output folder
os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)

# =========================
# PLOTTING MULTIPLE SAMPLES
# =========================
total_samples = X.shape[0]

start_index = max(0, min(start_index, total_samples - 1))
end_index = min(start_index + num_samples_to_plot, total_samples)

print(f"Plotting samples from {start_index} to {end_index - 1}")

for sample_idx in range(start_index, end_index):
    print(f"Processing sample {sample_idx}...")

    sample_X = X[sample_idx]
    sample_Y = Y[sample_idx]
    print("Sample Y:", sample_Y)
    n_bands, n_channels, _ = sample_X.shape

    # Create one figure containing all bands
    fig, axes = plt.subplots(
        1, n_bands,
        figsize=(5 * n_bands, 5),
        squeeze=False
    )

    axes = axes[0]  # flatten row

    for b in range(n_bands):
        cov_matrix = sample_X[b]

        im = axes[b].imshow(
            cov_matrix,
            cmap='viridis',
            interpolation='none'
        )

        axes[b].set_title(f'Band {b+1}')
        axes[b].set_xlabel('Channel')

        if b == 0:
            axes[b].set_ylabel('Channel')

        plt.colorbar(im, ax=axes[b], fraction=0.046, pad=0.04)

    fig.suptitle(f'Sample {sample_idx} | Y = {sample_Y.tolist()}', fontsize=14)
    plt.tight_layout()

    filename = f"sample{sample_idx}.png"
    save_path = os.path.join(OUTPUT_PLOTS_DIR, filename)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

print(f"\nSaved plots for {num_samples_to_plot} samples to:\n{OUTPUT_PLOTS_DIR}")