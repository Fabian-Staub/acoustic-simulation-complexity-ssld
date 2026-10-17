import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import spaudiopy as spa
from scipy.signal import butter, sosfiltfilt
import pyfar as pf
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../generate_dataset/scm")))
from utils import compute_scm, compute_scm_avg_2


# =========================
# I/O
# =========================
def load_wav(path):
    x, sr = sf.read(path)
    if x.ndim == 1:
        x = x[:, None]
    return x, sr

def bandpass_filter(x, sr, f_low=1900, f_high=2100, order=4):
    """
    Zero-phase bandpass filter.
    """
    nyq = 0.5 * sr
    low = f_low / nyq
    high = f_high / nyq

    sos = butter(order, [low, high], btype='bandpass', output='sos')

    # apply per channel
    y = np.zeros_like(x)
    for ch in range(x.shape[1]):
        y[:, ch] = sosfiltfilt(sos, x[:, ch])

    return y


# =========================
# SCM
# =========================
def compute_scm_simple(x, remove_mean=False):
    if remove_mean:
        x = x - np.mean(x, axis=0, keepdims=True)

    X = x.T.astype(np.complex128)
    return (X @ X.conj().T) / X.shape[1]


# =========================
# Eigen decomposition
# =========================
def eig_decompose(scm):
    evals, evecs = np.linalg.eigh(scm)
    idx = np.argsort(evals)[::-1]
    return evals[idx].real, evecs[:, idx]


# =========================
# MDL source estimation
# =========================
def estimate_num_sources_sorte(evals, max_sources=3):
    """
    SORTE (Second-Order Relative Eigenvalue Threshold) estimator.

    Based on curvature of eigenvalue spectrum:
    lambda_i - 2*lambda_{i+1} + lambda_{i+2}
    """

    evals = np.array(evals, dtype=float)

    # safety: remove negative numerical noise
    evals = np.maximum(evals, 0)

    # limit search space (important for stability)
    if max_sources is None:
        max_sources = len(evals) - 2
    else:
        max_sources = min(max_sources, len(evals) - 2)

    if max_sources < 2:
        return 1

    # normalize eigenvalues (important in practice)
    evals = evals / (np.max(evals) + 1e-12)

    # first differences
    diff = np.diff(evals)

    # second-order differences (curvature)
    curvature = diff[:-1] - diff[1:]

    # avoid division issues
    curvature = np.abs(curvature)

    # normalize
    curvature = curvature / (np.max(curvature) + 1e-12)

    # SORTE decision: strongest curvature transition
    k_opt = np.argmax(curvature) + 1

    # clamp to valid range
    k_opt = int(np.clip(k_opt, 1, max_sources))

    return k_opt


def mdl_number_of_sources(eigvals, N):
    """
    eigvals: eigenvalues of SCM (1D array)
    N: number of snapshots/samples
    returns: estimated number of sources K
    """

    M = len(eigvals)
    mdl_values = []

    for k in range(M):  # k = number of sources hypothesis

        noise_eigs = eigvals[k:]
        Mk = M - k

        if Mk <= 0:
            break

        # arithmetic mean
        arith = np.mean(noise_eigs)

        # geometric mean (log-stable)
        geom = np.exp(np.mean(np.log(noise_eigs + 1e-12)))  # small epsilon for safety

        # likelihood term
        if geom <= 0 or arith <= 0:
            mdl_values.append(np.inf)
            continue

        term1 = -N * Mk * np.log(geom / arith)

        # penalty term (as in your paper)
        term2 = k * (2 * M - k) * np.log(N)

        mdl = term1 + term2
        mdl_values.append(mdl)

    K_hat = int(np.argmin(mdl_values))
    return K_hat, np.array(mdl_values)


import numpy as np
import matplotlib.pyplot as plt


def mdl_source_number(X, eps=1e-12, plot=True, use_aic=True):
    """
    Robust MDL (and optional AIC) source number estimation.

    Parameters
    ----------
    X : (M, N) ndarray
        Sensor data matrix (M sensors, N snapshots)

    eps : float
        Numerical floor for eigenvalues (stability)

    plot : bool
        Whether to plot eigenvalues and criteria

    use_aic : bool
        Also compute AIC for cross-check

    Returns
    -------
    k_mdl : int
        Estimated number of sources via MDL

    k_aic : int or None
        Estimated number of sources via AIC (if enabled)

    mdl_values : ndarray
    aic_values : ndarray or None
    eigvals : ndarray
    """
    
    M, N = X.shape

    # 1. Sample covariance matrix
    R = (X @ X.conj().T) / N

    # 2. Eigenvalues (sorted descending)
    eigvals = np.linalg.eigvalsh(R)
    eigvals = np.flip(np.sort(eigvals.real))

    # Numerical stabilization
    eigvals = np.clip(eigvals, eps, None)

    print("Eigenvalues:\n", eigvals)

    mdl_values = []
    aic_values = [] if use_aic else None

    for k in range(M - 1):  # important: avoid empty noise subspace
        noise_eigs = eigvals[k:]
        m_k = M - k

        # geometric + arithmetic mean (stable)
        geom_mean = np.exp(np.mean(np.log(noise_eigs)))
        arith_mean = np.mean(noise_eigs)

        if arith_mean <= 0 or geom_mean <= 0:
            mdl_values.append(np.inf)
            if use_aic:
                aic_values.append(np.inf)
            continue

        ratio = geom_mean / arith_mean

        # prevent log(0)
        ratio = np.clip(ratio, eps, 1.0)

        # MDL criterion
        mdl = -(m_k * N) * np.log(ratio) + 0.5 * k * (2 * M - k) * np.log(N)
        mdl_values.append(mdl)

        # AIC criterion (optional)
        if use_aic:
            aic = -(m_k * N) * np.log(ratio) + k * (2 * M - k)
            aic_values.append(aic)

    mdl_values = np.array(mdl_values)
    k_mdl = int(np.argmin(mdl_values))

    k_aic = None
    if use_aic:
        aic_values = np.array(aic_values)
        k_aic = int(np.argmin(aic_values))

    # ---------------- PLOTTING ----------------
    if plot:
        plt.figure()
        plt.plot(eigvals, marker='o')
        #plt.yscale('log')
        plt.xlabel("Eigenvalue Index")
        plt.ylabel("Eigenvalue")
        plt.title("Eigenvalues of SCM")
        plt.savefig("eigenvalues.png", dpi=300)
        plt.close()


        plt.figure()
        plt.plot(mdl_values, label="MDL")
        if use_aic:
            plt.plot(aic_values, label="AIC")
        plt.title("Model Order Selection Criteria")
        plt.xlabel("Number of sources k")
        plt.ylabel("Cost")
        plt.legend()
        plt.grid()
        plt.savefig("mdl_aic.png", dpi=300)
        plt.close()

    return k_mdl, k_aic, mdl_values, aic_values, eigvals


# =========================
# Subspaces
# =========================
def estimate_subspaces(evals, evecs, num_sources):
    Qs = evecs[:, :num_sources]
    Qn = evecs[:, num_sources:]
    return Qs, Qn


# =========================
# MUSIC spectrum
# =========================
def compute_music_spectrum(Qn, order, az, zen):
    AZ, ZEN = np.meshgrid(az, zen)

    az_flat = AZ.ravel()
    zen_flat = ZEN.ravel()

    Y = spa.sph.sh_matrix(order, az_flat, zen_flat, sh_type="real").T

    Pn = Qn @ Qn.conj().T

    denom = np.sum((Y.conj() * (Pn @ Y)), axis=0).real
    P = 1.0 / (denom + 1e-12)

    return AZ, ZEN, P.reshape(AZ.shape)


# =========================
# DOA extraction
# =========================
def estimate_doa_topk(AZ, ZEN, P, K, min_sep_deg=20):
    """
    Select DOAs with angular separation constraint.
    """

    flat = P.ravel()
    idx_sorted = np.argsort(flat)[::-1]

    az_all = np.rad2deg(AZ.ravel())
    el_all = np.rad2deg((np.pi / 2 - ZEN.ravel()))

    selected = []

    def angular_distance(a1, e1, a2, e2):
        da = a1 - a2
        de = e1 - e2
        return np.sqrt(da**2 + de**2)

    for idx in idx_sorted:
        cand = (az_all[idx], el_all[idx])

        # check distance to already selected sources
        too_close = False
        for s in selected:
            if angular_distance(cand[0], cand[1], s[0], s[1]) < min_sep_deg:
                too_close = True
                break

        if not too_close:
            selected.append(cand)

        if len(selected) >= K:
            break

    az = np.array([s[0] for s in selected])
    el = np.array([s[1] for s in selected])

    return az, el


# =========================
# Plotting
# =========================
def plot_music(AZ, ZEN, P, az_est=None, el_est=None):
    P_db = 10 * np.log10(P / np.max(P) + 1e-12)

    az_deg = np.rad2deg(AZ)
    el_deg = 90 - np.rad2deg(ZEN)  # convert colatitude → elevation

    plt.figure(figsize=(8, 4))
    plt.tricontourf(
        np.rad2deg(AZ.ravel()),
        90 - np.rad2deg(ZEN.ravel()),
        P_db.ravel(),
        levels=50
    )
    plt.colorbar(label="dB")
    
    if az_est is not None and el_est is not None:
        plt.scatter(
            az_est,
            el_est,
            c='red',
            s=100,
            marker='x',
            label='Estimated DOA'
        )
    
    plt.legend()
    plt.xlabel("Azimuth (deg)")
    plt.ylabel("Elevation (deg)")
    plt.title("MUSIC Spectrum")

    plt.gca().invert_xaxis()  # invert azimuth axis

    plt.tight_layout()
    plt.savefig("music_map.png", dpi=300)
    plt.close()


# =========================
# Main
# =========================
def main():
    wav_path = "/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/generate_dataset/scm/output_ambisonics/ambisonics_r0_p0_rep0.wav"
    order = 4

    vecs = spa.grids.load_t_design(degree=21)
    az = np.arctan2(vecs[:, 1], vecs[:, 0])
    zen = np.arccos(vecs[:, 2]) 

    # Load
    x, fs = load_wav(wav_path)
    print(f"Loaded: {x.shape}, SR={fs}")

    # SCM
    #x = bandpass_filter(x, fs, f_low=3000, f_high=5000)
    #scm = compute_scm(x)
    '''
    signal_obj = pf.Signal(x.T, fs)
    signal_bands = pf.dsp.filter.fractional_octave_bands(
        signal_obj,
        num_fractions=1,
        frequency_range=(2000, 8000),
        order=15
    )
    cov_all_bands = compute_scm(signal_bands)
    scm = cov_all_bands[1]  # extract single SCM from group

    #scm = compute_scm_avg_2(x.T, fs, (2000, 7000), num_scms=8)
    #scm = scm[2]  # extract single SCM from group

    print(f"SCM shape: {scm.shape}")
    plt.figure(figsize=(6, 5))
    plt.imshow(np.real(scm), aspect='auto')
    plt.title("SCM Heatmap")
    plt.colorbar(label="Value")
    plt.xlabel("Channels")
    plt.ylabel("Channels")
    plt.tight_layout()
    plt.savefig("scm_heatmap.png", dpi=300)
    plt.close()

    # Eigen decomposition
    evals, evecs = eig_decompose(scm)
    print("Eigenvalues:", evals)

    plt.plot(evals, marker='o')
    #plt.yscale('log')
    plt.xlabel("Eigenvalue Index")
    plt.ylabel("Eigenvalue")
    plt.title("Eigenvalues of SCM")
    plt.savefig("eigenvalues.png", dpi=300)
    plt.close()

    num_sources_est, mdl_values = mdl_number_of_sources(evals, N=24000)
    print(f"Estimated number of sources (MDL): {num_sources_est}")
    print("MDL values:", mdl_values)
    '''

    k_mdl, k_aic, mdl_vals, aic_vals, evals = mdl_source_number(x.T)
    print(f"Estimated number of sources (MDL): {k_mdl}")
    print(f"Estimated number of sources (AIC): {k_aic}")
    print("MDL values:", mdl_vals)
    print("AIC values:", aic_vals)

    num_sources_est_sorte = estimate_num_sources_sorte(evals, max_sources=3)
    print(f"Estimated number of sources (SORTE): {num_sources_est_sorte}")

    # Estimate number of sources (MDL)
    N = x.shape[0]
    num_sources = 3 #estimate_num_sources_sorte(evals, max_sources=3)

    print(f"\nEstimated number of sources: {num_sources}")

    # Subspaces
    Qs, Qn = estimate_subspaces(evals, evecs, num_sources)

    # MUSIC
    AZ, ZEN, P = compute_music_spectrum(Qn, order, az, zen)

    # DOA extraction
    az_est, el_est = estimate_doa_topk(AZ, ZEN, P, num_sources, min_sep_deg=20)

    plot_music(AZ, ZEN, P, az_est, el_est)

    print("\nEstimated DOAs:")
    for i, (az, el) in enumerate(zip(az_est, el_est)):
        print(f"Source {i+1}: Azimuth = {az:.2f}°, Elevation = {el:.2f}°")

    print("\nSaved: music_map.png")


if __name__ == "__main__":
    main()