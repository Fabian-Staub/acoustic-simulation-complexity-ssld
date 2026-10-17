import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

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
    
    scm = compute_scm(x, normalize=True, remove_mean=True)
    
    print("SCM computed:", scm.shape)
    
    plot_and_save_scm(scm, output_img)
    print(f"SCM plot saved to: {output_img}")

if __name__ == "__main__":
    main()