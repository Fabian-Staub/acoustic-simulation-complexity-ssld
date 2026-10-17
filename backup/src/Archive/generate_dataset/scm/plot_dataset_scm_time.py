import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# =========================
# SETTINGS
# =========================
#DATA_PATH = "../../../data/harp/R800_P10_S5_Order4_24kHz_shoeBox_ISM/harp_dataset_scm_time.pt"
#OUTPUT_PLOTS_DIR = "../../../data/harp/R800_P10_S5_Order4_24kHz_shoeBox_ISM/covariance_plots"
DATA_PATH = "../../../data/arni_em32/sh_wavs/arni_dataset_scm_time.pt"
OUTPUT_PLOTS_DIR = "../../../data/arni_em32/sh_wavs/covariance_plots"

start_index = 0
num_samples_to_plot = 5

# =========================
# LOAD DATA
# =========================
data = torch.load(DATA_PATH, weights_only=False)
X = data["X"]   # shape: (num_samples, num_scm_channels, num_time_steps)
Y = data["Y"]

print("X shape:", X.shape)
print("Y shape:", Y.shape)

os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)

# =========================
# HELPER: reconstruct SCM
# =========================
def reconstruct_scm(vec):
    num_scm_channels = vec.shape[0]

    # Recover number of channels
    n_channels = int((np.sqrt(8 * num_scm_channels + 1) - 1) / 2)

    scm = np.zeros((n_channels, n_channels), dtype=np.float32)

    tri_idx = np.triu_indices(n_channels)
    scm[tri_idx] = vec

    # Mirror to lower triangle
    scm = scm + np.triu(scm, 1).T

    return scm

# =========================
# PLOTTING
# =========================
total_samples = X.shape[0]

start_index = max(0, min(start_index, total_samples - 1))
end_index = min(start_index + num_samples_to_plot, total_samples)

print(f"Plotting samples from {start_index} to {end_index - 1}")

for sample_idx in range(start_index, end_index):
    print(f"\nProcessing sample {sample_idx}...")

    sample_X = X[sample_idx].numpy()  
    num_scm_channels, num_time_steps = sample_X.shape

    for t in range(num_time_steps):
        vec = sample_X[:, t]

        scm_matrix = reconstruct_scm(vec)

        plt.figure(figsize=(6, 5))
        plt.imshow(scm_matrix, cmap='viridis', interpolation='none')
        plt.colorbar(label='Covariance (normalized 0-1)')
        plt.title(f'Sample {sample_idx} - Time step {t}')
        plt.xlabel('Channel')
        plt.ylabel('Channel')

        filename = f"sample{sample_idx}_t{t}.png"
        save_path = os.path.join(OUTPUT_PLOTS_DIR, filename)
        plt.savefig(save_path, dpi=300)
        plt.close()

print(f"\nSaved plots to:\n{OUTPUT_PLOTS_DIR}")