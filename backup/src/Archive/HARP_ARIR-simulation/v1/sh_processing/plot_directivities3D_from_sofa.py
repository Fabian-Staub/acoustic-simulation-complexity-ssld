import sofar as sf
import numpy as np
import matplotlib.pyplot as plt

sofa = sf.read_sofa("eigenmike32_simulated.sofa", verify=False)

rir = sofa.Data_IR
fs = sofa.Data_SamplingRate

# ----------------------------
# FFT + Frequenzband um 4 kHz
# ----------------------------
n_fft = rir.shape[-1]
freqs = np.fft.rfftfreq(n_fft, 1/fs)

band = (freqs >= 4000) & (freqs <= 5000)

rir_fft = np.fft.rfft(rir, axis=2)

# (480, 32) Magnitude pro Richtung & Mic
mag = np.mean(np.abs(rir_fft[:, :, band]), axis=2)

# ----------------------------
# Source directions
# ----------------------------
src = sofa.SourcePosition
az = np.deg2rad(src[:, 0])
el = np.deg2rad(src[:, 1])

x = np.cos(el) * np.cos(az)
y = np.cos(el) * np.sin(az)
z = np.sin(el)

from scipy.spatial import Delaunay

fig = plt.figure(figsize=(20, 16))

# triangulation in angular space (IMPORTANT: done once)
tri = Delaunay(np.column_stack([az, el]))

for mic in range(32):

    ax = fig.add_subplot(8, 4, mic + 1, projection='3d')

    gain = mag[:, mic]

    # dB normalize
    gain_db = 20 * np.log10(gain + 1e-12)
    gain_db -= np.max(gain_db)

    # radius mapping
    r = 1 + gain_db / 40
    r = np.clip(r, 0, 1)

    # spherical coordinates → surface points
    X = x * r
    Y = y * r
    Z = z * r

    # surface plot (CONNECTED)
    ax.plot_trisurf(
        X, Y, Z,
        triangles=tri.simplices,
        cmap='viridis',
        linewidth=0.2,
        antialiased=True,
        alpha=0.5
    )

    ax.set_title(f"Mic {mic+1}", fontsize=8)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

plt.suptitle("EM32 Directivity (Connected Sphere Surfaces)", fontsize=16)
plt.tight_layout()

plt.savefig("em32_directivity_connected_spheres.png", dpi=300)
plt.show()