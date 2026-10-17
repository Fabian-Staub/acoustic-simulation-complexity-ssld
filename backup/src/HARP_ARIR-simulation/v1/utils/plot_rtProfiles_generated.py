import re
import numpy as np
import matplotlib.pyplot as plt

# Read output file
with open("job_generate_shoebox_array_0.out", "r") as f:
    text = f.read()

# Extract all RT60 arrays
rt60_matches = re.findall(
    r"RT60:\s*\[(.*?)\]",
    text,
    re.DOTALL
)

# Convert to numpy array
rtProfilesMit = np.array([
    np.fromstring(match, sep=' ')
    for match in rt60_matches
])

print(f"Loaded {len(rtProfilesMit)} RT60 profiles")
print(f"Shape: {rtProfilesMit.shape}")

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Histogram for each band
for band_idx in range(rtProfilesMit.shape[1]):
    ax1.hist(
        rtProfilesMit[:, band_idx],
        bins=100,
        alpha=0.3,
        label=f"Band {band_idx + 1}"
    )

ax1.set_xlabel("RT60 (s)")
ax1.set_ylabel("Count")
ax1.set_title("RT60 Distribution by Band")
ax1.legend()

# RT60 profiles (one line per room)
bands = np.arange(rtProfilesMit.shape[1])

for rt_profile in rtProfilesMit:
    ax2.plot(bands, rt_profile, alpha=0.2, linewidth=0.8)

ax2.set_xlabel("Band Index")
ax2.set_ylabel("RT60 (s)")
ax2.set_title("RT60 per Band")

plt.tight_layout()

# Save figure
plt.savefig("rt60_profiles_generated_histogram.png", dpi=300)
plt.close()