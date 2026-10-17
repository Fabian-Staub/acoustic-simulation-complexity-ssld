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
def plot_music(AZ, ZEN, P):
    P_db = 10 * np.log10(P / np.max(P) + 1e-12)

    az_deg = np.rad2deg(AZ)
    el_deg = 90 - np.rad2deg(ZEN)  # convert colatitude → elevation

    plt.figure(figsize=(8, 4))
    plt.pcolormesh(az_deg, el_deg, P_db, shading="auto")
    plt.colorbar(label="dB")
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
    wav_path = "/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/generate_dataset/scm/output_ambisonics/ambisonics_r0_p9_rep1.wav"
    order = 4

    az = np.linspace(-np.pi, np.pi, 360)
    zen = np.linspace(0, np.pi, 180)

    # Load
    x, sr = load_wav(wav_path)
    print(f"Loaded: {x.shape}, SR={sr}")
    x = bandpass_filter(x, sr, f_low=200, f_high=300)

    # SCM
    scm = compute_scm(x)

    # Eigen decomposition
    evals, evecs = eig_decompose(scm)

    print("Eigenvalues:", evals)
    plt.plot(evals, marker='o')
    plt.xlabel("Eigenvalue Index")
    plt.ylabel("Eigenvalue")
    plt.title("Eigenvalues of SCM")
    plt.savefig("eigenvalues.png", dpi=300)
    plt.close()

    # Estimate number of sources (MDL)
    N = x.shape[0]
    num_sources = estimate_num_sources_sorte(evals, max_sources=3)

    print(f"\nEstimated number of sources: {num_sources}")

    # Subspaces
    Qs, Qn = estimate_subspaces(evals, evecs, num_sources)

    # MUSIC
    AZ, ZEN, P = compute_music_spectrum(Qn, order, az, zen)

    plot_music(AZ, ZEN, P)

    # DOA extraction
    az_est, el_est = estimate_doa_topk(AZ, ZEN, P, num_sources, min_sep_deg=20)

    print("\nEstimated DOAs:")
    for i, (az, el) in enumerate(zip(az_est, el_est)):
        print(f"Source {i+1}: Azimuth = {az:.2f}°, Elevation = {el:.2f}°")

    print("\nSaved: music_map.png")


if __name__ == "__main__":
    main()