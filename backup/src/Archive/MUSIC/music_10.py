import torch
import numpy as np
import spaudiopy as spa
import os
import sys
import time
import resource
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../ML_model/scm_model")))
from evaluate_5 import match_tracks, init_metrics, update_metrics, compute_seld, print_metrics


# =========================================================
# DATA LOADING
# =========================================================
def load_dataset(pt_path):
    data = torch.load(pt_path)
    return data["X"].numpy(), data["Y"].numpy()


# =========================================================
# ANGULAR DISTANCE
# =========================================================
def angular_distance(a, b):
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    dot = np.clip(np.dot(a, b), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


# =========================================================
# EIG / SUBSPACE
# =========================================================
def eig_decompose(scm):
    evals, evecs = np.linalg.eigh(scm)
    idx = np.argsort(evals)[::-1]
    return evals[idx].real, evecs[:, idx]


def estimate_subspaces(evecs, num_sources):
    return evecs[:, num_sources:]


# =========================================================
# DOA HELPERS
# =========================================================
def estimate_doa_topk(AZ, ZEN, P, K, min_sep_deg=15):
    flat = P.ravel()
    idx_sorted = np.argsort(flat)[::-1]

    az_all = np.rad2deg(AZ.ravel())
    el_all = 90 - np.rad2deg(ZEN.ravel())

    selected = []

    def dist(a1, e1, a2, e2):
        return np.sqrt((a1 - a2) ** 2 + (e1 - e2) ** 2)

    for idx in idx_sorted:
        cand = (az_all[idx], el_all[idx])

        if all(dist(cand[0], cand[1], s[0], s[1]) >= min_sep_deg for s in selected):
            selected.append(cand)

        if len(selected) >= K:
            break

    if len(selected) == 0:
        return np.array([]), np.array([])

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
# PRECOMPUTE GRID
# =========================================================
def precompute_music_grid(order, az_grid, zen_grid):
    AZ, ZEN = np.meshgrid(az_grid, zen_grid)

    Y = spa.sph.sh_matrix(
        order,
        AZ.ravel(),
        ZEN.ravel(),
        sh_type="real"
    ).T

    return AZ, ZEN, Y, AZ.shape


# =========================================================
# ================== WORKER FUNCTION ======================
# =========================================================
def process_sample(
    i,
    x_i,
    y_i,
    Y,
    AZ,
    ZEN,
    grid_shape,
    num_sources,
    doa_threshold
):
    scm_bands = x_i
    doa_gt = y_i

    gt_active = doa_gt[np.linalg.norm(doa_gt, axis=1) > 0]

    spectra = np.zeros(grid_shape, dtype=np.float64)

    Yc = Y.conj()

    for b in range(scm_bands.shape[0]):
        scm = scm_bands[b]

        _, evecs = eig_decompose(scm)
        Qn = estimate_subspaces(evecs, num_sources)
        Pn = Qn @ Qn.conj().T

        denom = np.sum((Yc * (Pn @ Y)), axis=0).real
        spectra += (1.0 / (denom + 1e-12)).reshape(grid_shape)

    P_avg = spectra / scm_bands.shape[0]

    az_est, el_est = estimate_doa_topk(AZ, ZEN, P_avg, num_sources)
    pred_xyz = azel_to_cartesian(az_est, el_est)

    valid_pred = pred_xyz if len(pred_xyz) > 0 else np.zeros((0, 3))

    dists = match_tracks(gt_active, valid_pred)

    n_sources = len(gt_active)

    return gt_active, valid_pred, dists, n_sources


# =========================================================
# ================= PARALLEL EVALUATION ===================
# =========================================================
def evaluate_music_parallel(
    x,
    y,
    Y,
    AZ,
    ZEN,
    grid_shape,
    num_sources,
    doa_threshold=20,
):

    metrics_all = init_metrics()
    metrics_1 = init_metrics()
    metrics_2 = init_metrics()
    metrics_3 = init_metrics()

    N = x.shape[0]

    results = []

    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                process_sample,
                i,
                x[i],
                y[i],
                Y,
                AZ,
                ZEN,
                grid_shape,
                num_sources,
                doa_threshold
            )
            for i in range(N)
        ]

        for f in as_completed(futures):
            results.append(f.result())

    # ================= REDUCTION =================
    for sample_idx, (gt_active, valid_pred, dists, n_sources) in enumerate(results):

        print("\n" + "=" * 60)
        print(f"Sample {sample_idx}")
        print("=" * 60)

        print("GT DOA(s):")
        if len(gt_active) == 0:
            print("  None")
        else:
            for k, gt in enumerate(gt_active):
                print(f"  GT {k}: {gt}")

        print("\nPredicted DOA(s):")
        if len(valid_pred) == 0:
            print("  None")
        else:
            for k, pred in enumerate(valid_pred):
                print(f"  Pred {k}: {pred}")

        print("\nAngular distances:")
        print(dists)

        update_metrics(metrics_all, gt_active, valid_pred, dists, doa_threshold)

        if n_sources == 1:
            update_metrics(metrics_1, gt_active, valid_pred, dists, doa_threshold)
        elif n_sources == 2:
            update_metrics(metrics_2, gt_active, valid_pred, dists, doa_threshold)
        elif n_sources == 3:
            update_metrics(metrics_3, gt_active, valid_pred, dists, doa_threshold)

    print_metrics("TOTAL", metrics_all)
    print_metrics("1 source", metrics_1)
    print_metrics("2 sources", metrics_2)
    print_metrics("3 sources", metrics_3)

    return (
        compute_seld(metrics_all),
        compute_seld(metrics_1),
        compute_seld(metrics_2),
        compute_seld(metrics_3)
    )


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    start = time.time()

    pt_path = "../../data/harp/R2000_P10_S5_Order4_24kHz_shoeBox_anisotropic_source-constraints_direct/harp_dataset_2-6kHz_8-SCM_randRot_multiSource_normalized.pt"

    x, y = load_dataset(pt_path)

    start_idx = int(0.999 * x.shape[0])
    x = x[start_idx:]
    y = y[start_idx:]

    print(f"Using samples {start_idx} → {x.shape[0] + start_idx}")

    order = 4
    num_sources = 3

    vecs = spa.grids.load_t_design(degree=21)
    az_grid = np.arctan2(vecs[:, 1], vecs[:, 0])
    zen_grid = np.arccos(vecs[:, 2])

    AZ, ZEN, Y, grid_shape = precompute_music_grid(
        order,
        az_grid,
        zen_grid
    )

    results = evaluate_music_parallel(
        x,
        y,
        Y,
        AZ,
        ZEN,
        grid_shape,
        num_sources,
        doa_threshold=20,
    )

    # Performance stats
    rusage_s = resource.getrusage(resource.RUSAGE_SELF)
    rusage_c = resource.getrusage(resource.RUSAGE_CHILDREN)
    print(f'Walltime {time.time()-start:.2f}s', file=sys.stderr)
    print(f'User time {rusage_s.ru_utime+rusage_c.ru_utime:.2f}s', file=sys.stderr)
    print(f'System time {rusage_s.ru_stime+rusage_c.ru_stime:.2f}s', file=sys.stderr)
    print(f'MaxRSS {(rusage_s.ru_maxrss+rusage_c.ru_maxrss)/2**20:.3f} GiB', file=sys.stderr)