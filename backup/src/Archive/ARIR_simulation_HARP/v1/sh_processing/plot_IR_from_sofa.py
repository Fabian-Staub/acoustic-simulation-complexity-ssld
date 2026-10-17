import sofar as sf
import numpy as np
import matplotlib.pyplot as plt

# =========================
# LOAD SOFA FILE
# =========================
sofa = sf.read_sofa("eigenmike32_analytic_rigid_sphere_causal_unwindowed.sofa", verify=False)

rir = sofa.Data_IR  # (M sources, 32 mics, N samples)
fs = float(sofa.Data_SamplingRate)

# =========================
# SELECT ONE SOURCE POSITION
# =========================
m = 0
ir = rir[m]  # (32, N)

# =========================
# TIME AXIS
# =========================
n_samples = ir.shape[1]
t = np.arange(n_samples) / fs * 1000  # ms

# =========================
# PLOT MULTICHANNEL RIR
# =========================
plt.figure(figsize=(12, 6))

for ch in range(ir.shape[0]):
    plt.plot(t, ir[ch], linewidth=0.8, alpha=0.8)

plt.title("Multichannel Impulse Responses (Time Domain)")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()
plt.savefig("analytic_RIR_multichannel.png", dpi=300)
plt.show()

print("Saved: analytic_RIR_multichannel.png")