import sofar as sf
import numpy as np
import matplotlib.pyplot as plt

sofa = sf.read_sofa("eigenmike32_analytic_rigid_sphere.sofa", verify=False)

rir = sofa.Data_IR
fs = sofa.Data_SamplingRate
src = sofa.SourcePosition

# ----------------------------
# FFT
# ----------------------------
n_fft = rir.shape[-1]
freqs = np.fft.rfftfreq(n_fft, 1/fs)
rir_fft = np.fft.rfft(rir, axis=2)

# ----------------------------
# Mic selection (2nd mic)
# ----------------------------
mic = 1

# ----------------------------
# Equatorial slice (middle circle)
# ----------------------------
az = np.deg2rad(src[:, 0])
el = np.deg2rad(src[:, 1])

equator_mask = np.abs(el) < np.deg2rad(5)  # ±5° around equator

az_eq = az[equator_mask]
rir_eq = rir_fft[:, mic, :][equator_mask, :]

# ----------------------------
# Log frequency bands
# ----------------------------
f_min = 100
f_max = 16000
n_bands = 20

edges = np.logspace(np.log10(f_min), np.log10(f_max), n_bands + 1)

colors = plt.cm.viridis(np.linspace(0, 1, n_bands))

# ----------------------------
# Plot
# ----------------------------
fig = plt.figure(figsize=(10, 8))
ax = plt.subplot(111, projection="polar")

for i in range(n_bands):

    f1, f2 = edges[i], edges[i + 1]
    band = (freqs >= f1) & (freqs < f2)

    if not np.any(band):
        continue

    # magnitude per direction
    mag = np.mean(np.abs(rir_eq[:, band]), axis=1)

    # dB normalization
    mag_db = 20 * np.log10(mag + 1e-12)
    mag_db = mag_db - np.max(mag_db)
    mag_db = np.clip(mag_db, -40, 0)

    # sort azimuth
    order = np.argsort(az_eq)

    ax.plot(
        az_eq[order],
        mag_db[order],
        color=colors[i],
        linewidth=1.5,
        label=f"{int(f1)}–{int(f2)} Hz"
    )

# ----------------------------
# Styling (TRUE dB SCALE)
# ----------------------------
ax.set_title("Mic 2 Directivity (Equator Slice, Log Bands, dB Scale)")

ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)

# dB radial axis
ax.set_ylim(-40, 0)
ax.set_yticks([-40, -30, -20, -10, 0])
ax.set_yticklabels(["-40 dB", "-30 dB", "-20 dB", "-10 dB", "0 dB"])

ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))

plt.tight_layout()

# Save output
plt.savefig("analytic_mic2_directivity_equator_logbands_db.png", dpi=300)

plt.show()