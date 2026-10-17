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

    az = np.linspace(-np.pi, np.pi, 360)
    colat = np.linspace(0, np.pi, 180)
    print("Azimuth:", az)
    print("Colatitude:", colat)
    order = 4
    answer=[]

    for a, c in zip(az, colat):
        steer = spa.sph.sh_matrix(order, a, c, sh_type="real")
        theta_music = 1/(steer.conj().T @ Qn @ Qn.conj().T @ steer)
        theta_music = 10*np.log10(np.abs(theta_music))
        answer.append(theta_music)


if __name__ == "__main__":
    main()