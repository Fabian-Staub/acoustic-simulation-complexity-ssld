import torch

data_path = "../../../data/harp/R2000_P10_S5_Order4_24kHz_shoeBox_ISM/harp_dataset_scm_time_rot_task_6.pt"

# Load data
data = torch.load(data_path, weights_only=False)

X = data["X"]
Y = data["Y"]

# Print shapes (optional)
print("X shape:", X.shape)
print("Y shape:", Y.shape)

# Count number of non-zero rows for each sample
distribution = {1: 0, 2: 0, 3: 0}

for y in Y:
    # Compute which rows are non-zero along the last dimension
    non_zero_rows = torch.any(y != 0, dim=1)  # shape: (3,) boolean
    count = non_zero_rows.sum().item()        # number of non-zero rows
    
    if count in distribution:
        distribution[count] += 1

print("\nDistribution of samples by non-zero rows in Y:")
for k in sorted(distribution):
    print(f"{k} non-zero rows: {distribution[k]} samples")