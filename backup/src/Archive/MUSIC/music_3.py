import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import spaudiopy as spa

def load_ambisonics_wav(path):
    x, sr = sf.read(path)
    if x.ndim == 1:
        x = x[:, None]
    return x, sr

def compute_scm(x, normalize=True, remove_mean=False):
    if remove_mean:
        x = x - np.mean(x, axis=0, keepdims=True)
    
    x = np.asarray(x, dtype=np.complex128)
    X = x.T  # (channels, samples)
    
    scm = X @ X.conj().T
    
    if normalize:
        scm /= X.shape[1]
    
    return scm

def plot_and_save_scm(scm, output_path="scm.png"):
    # Use magnitude for visualization
    scm_mag = np.abs(scm)
    
    plt.figure()
    plt.imshow(scm_mag)
    plt.colorbar()
    plt.title("Spatial Covariance Matrix (Magnitude)")
    plt.xlabel("Channel")
    plt.ylabel("Channel")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def main():
    wav_path = "sine_sh_67_37.wav"
    output_img = "scm.png"
    
    x, sr = load_ambisonics_wav(wav_path)
    print(f"Loaded: {x.shape[0]} samples, {x.shape[1]} channels")
    
    scm = np.cov(x, rowvar=False)
    print("SCM computed:", scm.shape)
    plot_and_save_scm(scm, output_img)
    print(f"SCM plot saved to: {output_img}")

    [evalues, evectors] = np.linalg.eig(scm)
    print("Eigenvalues:", np.real(evalues))
    plt.plot(np.real(evalues), marker='o')
    plt.savefig("eigenvalues.png", dpi=300)

    thresh=0.5
    n_signals=np.count_nonzero(np.real(evalues) > thresh)
    n_noise=np.count_nonzero(np.real(evalues) <= thresh)
    print(n_signals, n_noise)

    evectors = evectors[:, np.argsort(evalues)]
    
    Qn = evectors[:, :n_noise]
    #print("Noise subspace eigenvectors:", np.real(Qn))

    # --- Grid ---
    az = np.linspace(-np.pi, np.pi, 360)
    zen = np.linspace(0, np.pi, 180)   # zenith = colatitude

    AZ, ZEN = np.meshgrid(az, zen)

    order = 4

    # --- MUSIC map ---
    P = np.zeros_like(AZ, dtype=float)

    # Precompute noise projector (faster + cleaner)
    Pn = Qn @ Qn.conj().T

    for i in range(AZ.shape[0]):
        for j in range(AZ.shape[1]):
            a = AZ[i, j]
            z = ZEN[i, j]

            # steering vector (azimuth, colatitude=zenith)
            y = spa.sph.sh_matrix(order, a, z, sh_type="real")
            y = np.atleast_2d(y).T   # (C,1)

            denom = (y.conj().T @ Pn @ y).real
            P[i, j] = 1.0 / (denom + 1e-12)

    # --- Convert to dB ---
    P_db = 10 * np.log10(P / np.max(P) + 1e-12)

    # --- Plot ---
    plt.figure(figsize=(8, 4))
    plt.pcolormesh(
        np.rad2deg(AZ),
        np.rad2deg(ZEN),
        P_db,
        shading="auto"
    )
    plt.colorbar(label="dB")
    plt.xlabel("Azimuth (deg)")
    plt.ylabel("Colatitude (deg)")
    plt.title("MUSIC Spectrum")
    plt.tight_layout()
    plt.savefig("music_map.png", dpi=300)
    plt.close()

    print("Saved MUSIC map: music_map.png")


if __name__ == "__main__":
    main()