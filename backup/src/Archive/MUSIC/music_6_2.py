import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import spaudiopy as spa
from scipy.signal import butter, sosfiltfilt


# =========================
# I/O
# =========================
def load_wav(path):
    x, sr = sf.read(path)
    if x.ndim == 1:
        x = x[:, None]
    return x, sr


def bandpass_filter(x, sr, f_low=1900, f_high=2100, order=4):
    nyq = 0.5 * sr
    low = f_low / nyq
    high = f_high / nyq

    sos = butter(order, [low, high], btype='bandpass', output='sos')

    y = np.zeros_like(x)
    for ch in range(x.shape[1]):
        y[:, ch] = sosfiltfilt(sos, x[:, ch])

    return y


# =========================
# SCM
# =========================
def compute_scm(x, remove_mean=True):
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
# DOA peak picking
# =========================
def estimate_doa_topk(AZ, ZEN, P, K, min_sep_deg=20):
    flat = P.ravel()
    idx_sorted = np.argsort(flat)[::-1]

    az_all = np.rad2deg(AZ.ravel())
    el_all = np.rad2deg((np.pi / 2 - ZEN.ravel()))

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
# Steering matrix (SH domain)
# =========================
def build_steering_matrix(az_deg, el_deg, order):
    az = np.deg2rad(az_deg)
    el = np.deg2rad(90 - el_deg)

    Y = spa.sph.sh_matrix(order, az, el, sh_type="real")
    return Y.T


# =========================
# PAPER METHOD CORE
# =========================
def estimate_sources_paper_method(X, A_est, eps=1e-12):
    """
    Paper method:
    S_hat = (A^H A)^-1 A^H X
    Then estimate source number from power structure.
    """

    X = X.T.astype(np.complex128)  # (M x N)
    A = A_est.astype(np.complex128)

    # reconstruction
    S_hat = np.linalg.pinv(A.conj().T @ A + eps * np.eye(A.shape[1])) @ A.conj().T @ X

    # power per component
    power = np.mean(np.abs(S_hat)**2, axis=1)

    # normalize
    power = power / (np.max(power) + eps)

    # paper logic: strong vs weak components
    threshold = 0.25
    K_est = int(np.sum(power > threshold))

    return K_est, power


# =========================
# Plot MUSIC
# =========================
def plot_music(AZ, ZEN, P, az_est=None, el_est=None):
    P_db = 10 * np.log10(P / np.max(P) + 1e-12)

    plt.figure(figsize=(8, 4))
    plt.tricontourf(
        np.rad2deg(AZ.ravel()),
        90 - np.rad2deg(ZEN.ravel()),
        P_db.ravel(),
        levels=50
    )
    plt.colorbar(label="dB")

    if az_est is not None:
        plt.scatter(az_est, el_est, c='red', marker='x', s=100)

    plt.xlabel("Azimuth (deg)")
    plt.ylabel("Elevation (deg)")
    plt.title("MUSIC Spectrum")
    plt.tight_layout()
    plt.savefig("music_map.png", dpi=300)
    plt.close()


# =========================
# MAIN
# =========================
def main():
    wav_path = "/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/generate_dataset/scm/output_ambisonics/ambisonics_r0_p0_rep3.wav"
    order = 4

    vecs = spa.grids.load_t_design(degree=21)
    az = np.arctan2(vecs[:, 1], vecs[:, 0])
    zen = np.arccos(vecs[:, 2])

    # Load audio
    x, sr = load_wav(wav_path)
    print(f"Loaded: {x.shape}, SR={sr}")

    x = bandpass_filter(x, sr, 3000, 5000)

    # SCM
    scm = compute_scm(x)

    # Eigen decomposition (not used for K directly anymore)
    evals, evecs = eig_decompose(scm)

    plt.plot(evals, marker='o')
    plt.title("Eigenvalues")
    plt.savefig("eigenvalues.png", dpi=300)
    plt.close()

    # =========================
    # PAPER STEP 1: OVERCOMPLETE MODEL
    # =========================
    M = x.shape[1]
    k_over = M - 1
    print(f"\nOverestimated sources: {k_over}")

    # MUSIC noise subspace
    Qn = evecs[:, k_over:]

    AZ, ZEN, P = compute_music_spectrum(Qn, order, az, zen)

    # extract candidate DOAs
    az_full, el_full = estimate_doa_topk(AZ, ZEN, P, K=k_over, min_sep_deg=10)

    # steering matrix from candidates
    A_est = build_steering_matrix(az_full, el_full, order)

    # =========================
    # PAPER STEP 2: SOURCE NUMBER ESTIMATION
    # =========================
    K_est, power = estimate_sources_paper_method(x, A_est)

    print(f"\n[Paper Method] Estimated source number: {K_est}")

    plt.figure()
    plt.plot(power, marker='o')
    plt.title("Reconstructed Source Powers")
    plt.savefig("source_power.png", dpi=300)
    plt.close()

    # =========================
    # FINAL DOA SELECTION
    # =========================
    az_est, el_est = estimate_doa_topk(AZ, ZEN, P, K=K_est, min_sep_deg=20)

    plot_music(AZ, ZEN, P, az_est, el_est)

    print("\nEstimated DOAs:")
    for i, (a, e) in enumerate(zip(az_est, el_est)):
        print(f"Source {i+1}: Azimuth={a:.2f}, Elevation={e:.2f}")

    print("\nSaved: music_map.png + source_power.png")


if __name__ == "__main__":
    main()