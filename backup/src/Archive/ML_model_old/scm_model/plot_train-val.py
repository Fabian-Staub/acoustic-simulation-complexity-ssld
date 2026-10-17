import re
import matplotlib.pyplot as plt

log_file = "train_model_18300191.out"  # change to your file

epochs = []
train_loss = []
val_loss = []
seld = []

# Read and parse log file
with open(log_file, "r") as f:
    for line in f:
        match = re.search(
            r"Epoch (\d+)/\d+,\s*Train Loss:\s*([0-9.]+),\s*Val Loss:\s*([0-9.]+),\s*SELD:\s*([0-9.]+)",
            line
        )
        if match:
            epochs.append(int(match.group(1)))
            train_loss.append(float(match.group(2)))
            val_loss.append(float(match.group(3)))
            seld.append(float(match.group(4)))

# Plot
fig, ax1 = plt.subplots(figsize=(10, 5))

# Loss curves
ax1.plot(epochs, train_loss, label="Train Loss", color="tab:blue")
ax1.plot(epochs, val_loss, label="Val Loss", color="tab:orange")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.grid(True)

# SELD on second axis
ax2 = ax1.twinx()
ax2.plot(epochs, seld, label="SELD", color="tab:green")
ax2.set_ylabel("SELD")

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

plt.title("Training / Validation Loss and SELD")
plt.tight_layout()

plt.savefig("train-val-seld.png", dpi=300)