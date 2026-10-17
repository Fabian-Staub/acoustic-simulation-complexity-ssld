import torch
import numpy as np
import matplotlib.pyplot as plt

# ---- Load dataset ----
path = "../../data/harp/R2000_P10_S5_Order4_24kHz_shoeBox_anisotropic_source-constraints_direct/harp_dataset_2-6kHz_8-SCM_randRot_multiSource_normalized.pt"
data = torch.load(path, map_location="cpu")

# ---- Extract X ----
X = data["X"]
X = 2.0 * X - 1.0

sample0 = X[0]
scm = sample0[0].numpy()
scm = (scm + scm.conj().T) / 2

# ---- Correlation matrix ----
diag = np.diag(scm).real
std = np.sqrt(np.abs(diag) + 1e-12)
D_inv = np.diag(1.0 / std)
corr = D_inv @ scm @ D_inv

corr_real = np.real(corr)

# ---- Eigenvalues ----
eig_scm = np.linalg.eigvalsh(np.real(scm))
eig_scm = np.sort(eig_scm)[::-1]

eig_corr = np.linalg.eigvalsh(corr_real)
eig_corr = np.sort(eig_corr)[::-1]

# ---- Safe log transform (avoid negatives/zeros) ----
eps = 1e-12
eig_scm_log = np.maximum(eig_scm, eps)
eig_corr_log = np.maximum(eig_corr, eps)

# ---- Multipanel plot ----
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# --- SCM ---
im0 = axes[0, 0].imshow(np.real(scm), cmap="viridis")
axes[0, 0].set_title("SCM (Covariance)")
plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

# --- Correlation ---
im1 = axes[0, 1].imshow(corr_real, cmap="viridis", vmin=-1, vmax=1)
axes[0, 1].set_title("Correlation Matrix")
plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

# --- SCM eigenvalues (log scale) ---
axes[1, 0].semilogy(eig_scm_log, marker='o', markersize=3, linewidth=1)
axes[1, 0].set_title("SCM Eigenvalues (Log Scale)")
axes[1, 0].set_xlabel("Index")
axes[1, 0].set_ylabel("Eigenvalue (log)")
axes[1, 0].grid(True, which="both")

# --- Correlation eigenvalues (log scale) ---
axes[1, 1].semilogy(eig_corr_log, marker='o', markersize=3, linewidth=1)
axes[1, 1].set_title("Correlation Eigenvalues (Log Scale)")
axes[1, 1].set_xlabel("Index")
axes[1, 1].set_ylabel("Eigenvalue (log)")
axes[1, 1].grid(True, which="both")

plt.tight_layout()
plt.savefig("scm_correlation_eigs_multipanel_log.png", dpi=300)
plt.show()