import pandas as pd
import matplotlib.pyplot as plt

# Path to your CSV file
csv_path = "/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/motus/motus_directions_music_rt60.csv"

# Load the CSV
df = pd.read_csv(csv_path)

# Validate the required column
if "rt60" not in df.columns:
    raise ValueError("The CSV file must contain an 'rt60' column.")

# Remove missing or invalid values
rt60 = pd.to_numeric(df["rt60"], errors="coerce").dropna()

# Plot histogram
plt.figure(figsize=(6, 4))
plt.hist(rt60, bins=100)

plt.xlabel("RT60 (seconds)")
plt.ylabel("Count")
plt.title("Motus")
plt.tight_layout()

# Save as PNG
output_path = "rt60_histogram_motus.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Plot saved to: {output_path}")