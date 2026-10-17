import torch
import numpy as np
import matplotlib.pyplot as plt

# ---- Load dataset ----
path = "../../data/harp/R2000_P10_S5_Order4_24kHz_shoeBox_anisotropic_source-constraints_direct/harp_dataset_2-6kHz_8-SCM_randRot_multiSource_normalized.pt"
data = torch.load(path, map_location="cpu")

# ---- Extract X ----
X = data["X"]  # expected shape: [N, 8, 25, 25]
X = 2.0 * X - 1.0

print("X shape:", X.shape)

# ---- First sample ----
sample0 = X[1]  # shape: [8, 25, 25]

# ---- First frequency band SCM ----
cov = sample0[0].numpy()  # shape: [25, 25]

# ---- Ensure Hermitian symmetry (optional but good practice) ----
cov = (cov + cov.conj().T) / 2

# ---- Correlation coefficient matrix ----
diag = np.diag(cov).real
std = np.sqrt(np.abs(diag) + 1e-12)

D_inv = np.diag(1.0 / std)
corr = D_inv @ cov @ D_inv

# ---- Plot covariance + correlation ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

im0 = axes[0].imshow(np.real(cov), cmap="viridis")
axes[0].set_title("Covariance (SCM) - Band 0")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(np.real(corr), cmap="viridis", vmin=-1, vmax=1)
axes[1].set_title("Correlation Coefficient Matrix")
plt.colorbar(im1, ax=axes[1])

plt.tight_layout()
plt.savefig("correlation_matrix.png", dpi=300)
plt.show()

# ---- Eigenvalues of correlation matrix ----
corr_real = np.real(corr)

eigvals = np.linalg.eigvalsh(corr_real)  # Hermitian-safe
eigvals = np.sort(eigvals)[::-1]  # descending

print("Eigenvalues shape:", eigvals.shape)
print("Top 10 eigenvalues:", eigvals[:10])

# ---- Plot eigenvalues (linear scale) ----
plt.figure(figsize=(6, 4))
plt.plot(eigvals, marker='o', markersize=3, linewidth=1)
plt.title("Eigenvalues of Correlation Coefficient Matrix")
plt.xlabel("Index (sorted)")
plt.ylabel("Eigenvalue")
plt.grid(True)

plt.tight_layout()
plt.savefig("correlation_eigenvalues.png", dpi=300)
plt.show()

# ---- Plot eigenvalues (log scale) ----
plt.figure(figsize=(6, 4))
plt.semilogy(eigvals + 1e-12, marker='o', markersize=3, linewidth=1)
plt.title("Eigenvalues (Log Scale)")
plt.xlabel("Index (sorted)")
plt.ylabel("Eigenvalue (log scale)")
plt.grid(True)

plt.tight_layout()
plt.savefig("correlation_eigenvalues_log.png", dpi=300)
plt.show()