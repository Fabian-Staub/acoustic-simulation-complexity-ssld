import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# =========================
# SETTINGS
# =========================
DATA_PATH = "../../../data/arni_em32/sh_wavs/arni_dataset_intensity_time_narrowband_singleSource.pt"
OUTPUT_PLOTS_DIR = "../../../data/arni_em32/sh_wavs/intensity_plots"
#DATA_PATH = "../../../data/harp/R2000_P10_S5_Order4_24kHz_shoeBox_ISM/harp_dataset_energy_time_randRot_narrowband_singleSource_task_0.pt"
#OUTPUT_PLOTS_DIR = "../../../data/harp/R2000_P10_S5_Order4_24kHz_shoeBox_ISM/energy_plots"

start_index = 0
num_samples_to_plot = 5

# =========================
# LOAD DATA
# =========================
data = torch.load(DATA_PATH, weights_only=False)
X = data["X"]   # shape: (num_samples, channels, time_steps)
Y = data["Y"]

print("X shape:", X.shape)
print("Y shape:", Y.shape)

os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)

# =========================
# GLOBAL COLOR SCALE (optional but recommended)
# =========================
vmin = X.min().item()
vmax = X.max().item()

# =========================
# PLOTTING
# =========================
total_samples = X.shape[0]

start_index = max(0, min(start_index, total_samples - 1))
end_index = min(start_index + num_samples_to_plot, total_samples)

print(f"Plotting samples from {start_index} to {end_index - 1}")

for sample_idx in range(start_index, end_index):
    print(f"\nProcessing sample {sample_idx}...")

    sample_X = X[sample_idx].numpy()  # (channels, time_steps)

    plt.figure(figsize=(10, 6))

    plt.imshow(
        sample_X,
        aspect='auto',
        origin='lower',
        cmap='viridis',
        vmin=vmin,
        vmax=vmax
    )

    plt.colorbar(label='Energy')
    plt.title(f'Sample {sample_idx} - Channel Energy Over Time')
    plt.xlabel('Time Frame')
    plt.ylabel('Channel')

    # Optional: nicer ticks
    plt.yticks(np.arange(sample_X.shape[0]))
    plt.xticks(np.arange(sample_X.shape[1]))

    filename = f"sample{sample_idx}_energy.png"
    save_path = os.path.join(OUTPUT_PLOTS_DIR, filename)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

print(f"\nSaved plots to:\n{OUTPUT_PLOTS_DIR}")