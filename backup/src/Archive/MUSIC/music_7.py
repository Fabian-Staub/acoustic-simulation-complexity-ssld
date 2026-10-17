import numpy as np
import torch
import matplotlib.pyplot as plt
import spaudiopy as spa

# =========================
# Dataset loading
# =========================
def load_dataset(pt_path):
    data = torch.load(pt_path)

    x = data["X"].numpy()  # [N, F, C, C]
    y = data["Y"].numpy()  # [N, S, 3]

    return x, y


# =========================
# Eigen decomposition
# =========================
def eig_decompose(scm):
    evals, evecs = np.linalg.eigh(scm)
    idx = np.argsort(evals)[::-1]
    return evals[idx].real, evecs[:, idx]


# =========================
# Subspaces
# =========================
def estimate_subspaces(evecs, num_sources):
    Qn = evecs[:, num_sources:]
    return Qn


# =========================
# MUSIC spectrum
# =========================
def compute_music_spectrum(Qn, order, az_grid, zen_grid):

    AZ, ZEN = np.meshgrid(az_grid, zen_grid)

    az_flat = AZ.ravel()
    zen_flat = ZEN.ravel()

    Y = spa.sph.sh_matrix(
        order,
        az_flat,
        zen_flat,
        sh_type="real"
    ).T

    Pn = Qn @ Qn.conj().T

    denom = np.sum(
        (Y.conj() * (Pn @ Y)),
        axis=0
    ).real

    P = 1.0 / (denom + 1e-12)

    return AZ, ZEN, P.reshape(AZ.shape)


# =========================
# Peak extraction
# =========================
def estimate_doa_topk(AZ, ZEN, P, K, min_sep_deg=15):

    flat = P.ravel()
    idx_sorted = np.argsort(flat)[::-1]

    az_all = np.rad2deg(AZ.ravel())
    el_all = 90 - np.rad2deg(ZEN.ravel())

    selected = []

    def dist(a1, e1, a2, e2):
        return np.sqrt((a1 - a2)**2 + (e1 - e2)**2)

    for idx in idx_sorted:

        cand = (az_all[idx], el_all[idx])

        if all(dist(cand[0], cand[1], s[0], s[1]) >= min_sep_deg for s in selected):
            selected.append(cand)

        if len(selected) >= K:
            break

    az = np.array([s[0] for s in selected])
    el = np.array([s[1] for s in selected])

    return az, el


# =========================
# Cartesian → spherical
# =========================
def cartesian_to_azel(v):

    x, y, z = v

    az = np.rad2deg(np.arctan2(y, x))
    el = np.rad2deg(np.arcsin(z / np.linalg.norm(v)))

    return az, el


# =========================
# Plot
# =========================
def plot_music(AZ, ZEN, P, az_est, el_est, az_gt, el_gt):

    P_db = 10 * np.log10(P / np.max(P) + 1e-12)

    plt.figure(figsize=(8, 4))

    plt.tricontourf(
        np.rad2deg(AZ.ravel()),
        90 - np.rad2deg(ZEN.ravel()),
        P_db.ravel(),
        levels=50
    )

    plt.colorbar(label="dB")

    # Estimated
    plt.scatter(
        az_est,
        el_est,
        c="red",
        s=120,
        marker="x",
        label="Estimated"
    )

    # Ground truth
    plt.scatter(
        az_gt,
        el_gt,
        c="white",
        s=60,
        marker="+",
        linewidths=2,
        label="Ground Truth"
    )

    plt.xlabel("Azimuth (deg)")
    plt.ylabel("Elevation (deg)")
    plt.title("MUSIC (Averaged over Bands)")

    plt.legend()
    plt.gca().invert_xaxis()

    plt.tight_layout()
    plt.savefig("music_map_avg.png", dpi=300)
    plt.close()


# =========================
# MAIN
# =========================
def main():

    pt_path = (
        "/scratch/elec/t412-techpsychoacoustics/"
        "msc-thesis/2026_fstaub/staubf1/Projects/"
        "Master_Thesis/data/harp/"
        "R2000_P10_S5_Order4_24kHz_shoeBox_anisotropic_"
        "source-constraints_direct/"
        "harp_dataset_2-6kHz_8-SCM_randRot_multiSource_normalized.pt"
    )

    order = 4
    num_sources = 10

    # =========================
    # Load dataset
    # =========================
    x, y = load_dataset(pt_path)

    N = x.shape[0]
    start_idx = int(0.99999 * N)

    print(f"Total samples: {N}")
    print(f"Using: {start_idx} → {N}")

    # =========================
    # grid
    # =========================
    vecs = spa.grids.load_t_design(degree=21)
    az_grid = np.arctan2(vecs[:, 1], vecs[:, 0])
    zen_grid = np.arccos(vecs[:, 2])

    # =========================
    # loop over samples
    # =========================
    for sample_idx in range(start_idx, N):

        scm_bands = x[sample_idx]
        doa_gt = y[sample_idx]

        spectra = []

        for band_idx in range(scm_bands.shape[0]):

            scm = scm_bands[band_idx]

            evals, evecs = eig_decompose(scm)
            Qn = estimate_subspaces(evecs, num_sources)

            AZ, ZEN, P = compute_music_spectrum(
                Qn,
                order,
                az_grid,
                zen_grid
            )

            spectra.append(P)

        # average over frequency bands
        P_avg = np.mean(spectra, axis=0)

        # DOA estimation
        az_est, el_est = estimate_doa_topk(
            AZ,
            ZEN,
            P_avg,
            num_sources
        )

        # GT conversion
        az_gt, el_gt = [], []

        for v in doa_gt:
            az, el = cartesian_to_azel(v)
            az_gt.append(az)
            el_gt.append(el)

        az_gt = np.array(az_gt)
        el_gt = np.array(el_gt)

        # plot per sample (optional overwrite or save indexed)
        plot_music(
            AZ,
            ZEN,
            P_avg,
            az_est,
            el_est,
            az_gt,
            el_gt
        )

        print(f"\nSample {sample_idx}")
        print("Estimated:", list(zip(az_est, el_est)))
        print("GT:", list(zip(az_gt, el_gt)))


if __name__ == "__main__":
    main()