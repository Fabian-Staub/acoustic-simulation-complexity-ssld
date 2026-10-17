import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import spaudiopy as spa


# =========================
# I/O
# =========================
def load_wav(path):
    x, sr = sf.read(path)
    if x.ndim == 1:
        x = x[:, None]
    return x, sr


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
def estimate_num_sources_robust(evals, max_sources=3):
    """
    Robust eigenvalue-gap based source estimation.
    Works better than MDL in low snapshot / SH-MUSIC cases.
    """

    evals = np.array(evals)

    # remove tiny numerical negatives
    evals = np.maximum(evals, 0)

    # optional limit search range
    if max_sources is None:
        max_sources = len(evals) // 2

    # only look at leading part (important!)
    evals = evals[:max_sources]

    # eigenvalue gaps
    gaps = np.diff(evals)

    # normalize gaps (important for stability)
    gaps = gaps / (np.max(gaps) + 1e-12)

    # strongest drop = signal/noise boundary
    k_opt = np.argmax(gaps) + 1

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
def estimate_doa_topk(AZ, ZEN, P, K):
    flat = P.ravel()
    idx = np.argsort(flat)[::-1][:K]

    az = np.rad2deg(AZ.ravel()[idx])
    el = np.rad2deg((np.pi / 2 - ZEN.ravel()[idx]))  # colatitude → elevation

    return az, el


# =========================
# Plotting
# =========================
def plot_music(AZ, ZEN, P):
    P_db = 10 * np.log10(P / np.max(P) + 1e-12)

    plt.figure(figsize=(8, 4))
    plt.pcolormesh(np.rad2deg(AZ), np.rad2deg(ZEN), P_db, shading="auto")
    plt.colorbar(label="dB")
    plt.xlabel("Azimuth (deg)")
    plt.ylabel("Colatitude (deg)")
    plt.title("MUSIC Spectrum")
    plt.tight_layout()
    plt.savefig("music_map.png", dpi=300)
    plt.close()


# =========================
# Main
# =========================
def main():
    wav_path = "sine_sh_30_-51_-132_14.wav"
    order = 4

    az = np.linspace(-np.pi, np.pi, 360)
    zen = np.linspace(0, np.pi, 180)

    # Load
    x, sr = load_wav(wav_path)
    print(f"Loaded: {x.shape}, SR={sr}")

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
    num_sources = estimate_num_sources_robust(evals, max_sources=3)

    print(f"\nEstimated number of sources: {num_sources}")

    # Subspaces
    Qs, Qn = estimate_subspaces(evals, evecs, num_sources)

    # MUSIC
    AZ, ZEN, P = compute_music_spectrum(Qn, order, az, zen)

    plot_music(AZ, ZEN, P)

    # DOA extraction
    az_est, el_est = estimate_doa_topk(AZ, ZEN, P, num_sources)

    print("\nEstimated DOAs:")
    for i, (az, el) in enumerate(zip(az_est, el_est)):
        print(f"Source {i+1}: Azimuth = {az:.2f}°, Elevation = {el:.2f}°")

    print("\nSaved: music_map.png")


if __name__ == "__main__":
    main()