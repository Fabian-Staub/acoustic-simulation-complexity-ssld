import os
import numpy as np
import soundfile as sf
import spaudiopy as spa
import matplotlib.pyplot as plt
from scipy.signal import resample_poly


# ============================================================
# USER SETTINGS
# ============================================================

DATA_FOLDER     = "../data"
INPUT_FOLDER    = "output"
TARGET_ORDER    = 4
TARGET_FS       = 16000
TARGET_SUBTYPE  = "PCM_24"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def list_wav_files(folder):
    """
    Return a sorted list of all .wav files in the folder.
    """
    files = [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if f.lower().endswith(".wav")
    ]

    if not files:
        raise FileNotFoundError(f"No wav files found in {folder}")

    return files


def preprocess_wav(wav_file, target_order, target_fs, target_subtype):
    """
    Resample and convert Ambisonics WAV to target order,
    sample rate, and bit depth.
    """
    sig, fs = sf.read(wav_file, dtype="float32")

    nSH = sig.shape[1] if sig.ndim > 1 else 1
    current_order = int(np.sqrt(nSH) - 1)

    # --------------------------------------------------------
    # Resample (if needed)
    # --------------------------------------------------------
    if fs != target_fs:
        gcd = np.gcd(fs, target_fs)
        sig = resample_poly(
            sig,
            target_fs // gcd,
            fs // gcd,
            axis=0
        )
        fs = target_fs

    # --------------------------------------------------------
    # Convert Ambisonics order (if needed)
    # --------------------------------------------------------
    if current_order != target_order:
        nSH_target = (target_order + 1) ** 2
        sig_new = np.zeros((sig.shape[0], nSH_target), dtype=sig.dtype)

        minSH = min(nSH, nSH_target)
        sig_new[:, :minSH] = sig[:, :minSH]  # truncate or zero-pad

        sig = sig_new
        nSH = nSH_target

    # --------------------------------------------------------
    # Save as target subtype (if needed)
    # --------------------------------------------------------
    info = sf.info(wav_file)
    if info.subtype != target_subtype or info.samplerate != target_fs:
        sf.write(wav_file, sig, fs, subtype=target_subtype)

    return sig, fs


# ============================================================
# GRID & STEERING SETUP
# ============================================================

grid = spa.grids.load_t_design(TARGET_ORDER * 2)

x, y, z = grid[:, 0], grid[:, 1], grid[:, 2]

azi = np.arctan2(y, x)   # azimuth (theta)
zen = np.arccos(z)       # zenith  (phi)

stVec = spa.sph.sh_matrix(
    TARGET_ORDER,
    azi,
    zen,
    sh_type="real"
)


# ============================================================
# MAIN PROCESSING
# ============================================================

input_path = os.path.join(DATA_FOLDER, INPUT_FOLDER)
wav_files  = list_wav_files(input_path)

plot_folder = os.path.join(DATA_FOLDER, "plots")
os.makedirs(plot_folder, exist_ok=True)


for wav_file in wav_files:

    # --------------------------------------------------------
    # Preprocess WAV
    # --------------------------------------------------------
    ambisonic_sig, fs = preprocess_wav(
        wav_file,
        TARGET_ORDER,
        TARGET_FS,
        TARGET_SUBTYPE
    )

    if ambisonic_sig.ndim != 2:
        print(f"Skipping {wav_file}: not a multichannel Ambisonics file")
        continue

    nSH   = ambisonic_sig.shape[1]
    order = int(np.sqrt(nSH) - 1)

    print(f"\nLoaded {wav_file}")
    print(f"Detected order: {order}  ({nSH} SH channels)")

    # --------------------------------------------------------
    # Project to steering directions
    # --------------------------------------------------------
    sig_dir = ambisonic_sig @ stVec.T
    print("Shapes:", ambisonic_sig.shape, stVec.shape, sig_dir.shape)

    # Directional power
    P_dir = np.mean(sig_dir ** 2, axis=0)

    # Transform back to SH domain
    R_dir = stVec.T @ np.diag(P_dir) @ stVec


    # ========================================================
    # PLOT 1 — Spatial Covariance Matrix (SCM)
    # ========================================================
    scm_file = os.path.join(
        plot_folder,
        os.path.basename(wav_file).replace(".wav", "_scm.png")
    )

    plt.figure(figsize=(6, 5))
    plt.imshow(R_dir, origin="lower", cmap="magma")
    plt.colorbar(label="Magnitude (dB)")
    plt.title(f"Directional SCM: {os.path.basename(wav_file)}")
    plt.xlabel("Steering direction index")
    plt.ylabel("Steering direction index")
    plt.tight_layout()
    plt.savefig(scm_file, dpi=200)
    plt.close()

    print(f"Saved SCM plot: {scm_file}")


    # ========================================================
    # PLOT 2 — Directional Power
    # ========================================================
    power_file = os.path.join(
        plot_folder,
        os.path.basename(wav_file).replace(".wav", "_power.png")
    )

    plt.figure(figsize=(6, 4))
    plt.stem(P_dir)
    plt.xlabel("Steering direction index")
    plt.ylabel("Normalized power")
    plt.title(f"Spatial Power: {os.path.basename(wav_file)}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(power_file, dpi=200)
    plt.close()

    print(f"Saved power plot: {power_file}")


    # ========================================================
    # PLOT 3 — 3D Steering Directions
    # ========================================================
    steer3d_file = os.path.join(
        plot_folder,
        f"steering_directions.png"
    )

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Hollow points with index labels
    for i in range(len(x)):
        ax.scatter(
            x[i], y[i], z[i],
            facecolors="none",
            edgecolors="blue",
            s=100,
            linewidths=1
        )

        ax.text(
            x[i], y[i], z[i],
            str(i),
            color="red",
            fontsize=8,
            horizontalalignment="center",
            verticalalignment="center"
        )

    # Vectors from origin
    for i in range(len(x)):
        ax.plot(
            [0, x[i]],
            [0, y[i]],
            [0, z[i]],
            color="gray",
            linewidth=0.7
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"3D Steering Directions (Order {TARGET_ORDER})")
    ax.set_box_aspect([1, 1, 1])

    plt.tight_layout()
    plt.savefig(steer3d_file, dpi=200)
    plt.close()

    print(f"Saved 3D steering plot: {steer3d_file}")


print("\nDone processing all WAV files.")