import torch
import numpy as np
import spaudiopy as spa
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../ML_model/scm_model")))
from evaluate_5 import match_tracks, init_metrics, update_metrics, compute_seld, print_metrics

from scipy.optimize import linear_sum_assignment

# =========================================================
# DATA LOADING
# =========================================================
def load_dataset(pt_path):
    data = torch.load(pt_path)
    x = data["X"].numpy()
    y = data["Y"].numpy()
    return x, y


# =========================================================
# ANGULAR METRIC (same as SELD)
# =========================================================
def angular_distance(a, b):
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    dot = np.clip(np.dot(a, b), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


# =========================================================
# MUSIC CORE FUNCTIONS
# =========================================================
def eig_decompose(scm):
    evals, evecs = np.linalg.eigh(scm)
    idx = np.argsort(evals)[::-1]
    return evals[idx].real, evecs[:, idx]


def estimate_subspaces(evecs, num_sources):
    return evecs[:, num_sources:]


def compute_music_spectrum(Qn, order, az_grid, zen_grid):

    AZ, ZEN = np.meshgrid(az_grid, zen_grid)

    az_flat = AZ.ravel()
    zen_flat = ZEN.ravel()

    Y = spa.sph.sh_matrix(order, az_flat, zen_flat, sh_type="real").T

    Pn = Qn @ Qn.conj().T

    denom = np.sum((Y.conj() * (Pn @ Y)), axis=0).real

    P = 1.0 / (denom + 1e-12)

    return AZ, ZEN, P.reshape(AZ.shape)


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


def azel_to_cartesian(az_deg, el_deg):

    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)

    x = np.cos(el) * np.cos(az)
    y = np.cos(el) * np.sin(az)
    z = np.sin(el)

    return np.stack([x, y, z], axis=-1)


# =========================================================
# FULL MUSIC EVALUATION
# =========================================================
def evaluate_music(
    x,
    y,
    order,
    num_sources,
    az_grid,
    zen_grid,
    doa_threshold=20,
    verbose=True):

    metrics_all = init_metrics()
    metrics_1 = init_metrics()
    metrics_2 = init_metrics()
    metrics_3 = init_metrics()

    N = x.shape[0]

    for i in range(N):

        scm_bands = x[i]
        doa_gt = y[i]

        spectra = []

        for b in range(scm_bands.shape[0]):

            scm = scm_bands[b]

            evals, evecs = eig_decompose(scm)
            Qn = estimate_subspaces(evecs, num_sources)

            AZ, ZEN, P = compute_music_spectrum(
                Qn, order, az_grid, zen_grid
            )

            spectra.append(P)

        P_avg = np.mean(spectra, axis=0)

        az_est, el_est = estimate_doa_topk(
            AZ, ZEN, P_avg, num_sources
        )

        # =====================================================
        # GT processing
        # =====================================================
        gt_active = doa_gt[np.linalg.norm(doa_gt, axis=1) > 0]

        # =====================================================
        # Prediction (NO padding)
        # =====================================================
        pred_xyz_list = azel_to_cartesian(az_est, el_est)
        valid_pred = pred_xyz_list if len(pred_xyz_list) > 0 else np.zeros((0, 3))

        # =====================================================
        # Matching (use only valid GT / pred for metrics)
        # =====================================================
        dists = match_tracks(gt_active, valid_pred)

        # =====================================================
        # PRINT PER SAMPLE
        # =====================================================
        if verbose:

            print("\n" + "=" * 60)
            print(f"Sample {i}")
            print("=" * 60)

            print("\nGT (xyz):")
            if len(gt_active) == 0:
                print("  None")
            else:
                for g in gt_active:
                    print(" ", np.round(g, 4))

            print("\nPred (xyz):")
            if len(valid_pred) == 0:
                print("  None")
            else:
                for p in valid_pred:
                    print(" ", np.round(p, 4))

            print("\nMatched distances (deg):")
            if len(dists) == 0:
                print("  None")
            else:
                print(" ", np.round(dists, 2))

        # =====================================================
        # Update metrics
        # =====================================================
        update_metrics(
            metrics_all,
            gt_active,
            valid_pred,
            dists,
            doa_threshold
        )

        # -----------------------------------------
        # Metrics by number of GT sources
        # -----------------------------------------
        n_sources = len(gt_active)

        if n_sources == 1:

            update_metrics(
                metrics_1,
                gt_active,
                valid_pred,
                dists,
                doa_threshold
            )

        elif n_sources == 2:

            update_metrics(
                metrics_2,
                gt_active,
                valid_pred,
                dists,
                doa_threshold
            )

        elif n_sources == 3:

            update_metrics(
                metrics_3,
                gt_active,
                valid_pred,
                dists,
                doa_threshold
            )

    if verbose:
        print_metrics(
            "Evaluation Results TOTAL",
            metrics_all
        )

        print_metrics(
            "Evaluation Results 1 Active Source",
            metrics_1
        )

        print_metrics(
            "Evaluation Results 2 Active Sources",
            metrics_2
        )

        print_metrics(
            "Evaluation Results 3 Active Sources",
            metrics_3
        )

    return compute_seld(metrics_all), compute_seld(metrics_1), compute_seld(metrics_2), compute_seld(metrics_3)


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    pt_path = "../../data/harp/R2000_P10_S5_Order4_24kHz_shoeBox_anisotropic_source-constraints_direct/harp_dataset_2-6kHz_8-SCM_randRot_multiSource_normalized.pt"

    x, y = load_dataset(pt_path)
    # USE ONLY LAST 20% OF DATA
    start_idx = int(0.9999 * x.shape[0])  # last 20%
    x = x[start_idx:]
    y = y[start_idx:]
    print(f"Using samples from index {start_idx} to {x.shape[0] + start_idx - 1}")
    print(f"Total samples used: {x.shape[0]} / {start_idx + x.shape[0]} ({0.2 * 100:.0f}%)")

    order = 4
    num_sources = 3

    # GRID (same as your MUSIC pipeline)
    vecs = spa.grids.load_t_design(degree=21)
    az_grid = np.arctan2(vecs[:, 1], vecs[:, 0])
    zen_grid = np.arccos(vecs[:, 2])

    evaluate_music(
        x,
        y,
        order,
        num_sources,
        az_grid,
        zen_grid,
        doa_threshold=20
    )