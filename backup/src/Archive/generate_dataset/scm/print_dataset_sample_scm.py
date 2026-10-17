import torch

data_path = "../../../data/harp/R2000_P10_S5_Order4_24kHz_shoeBox_anisotropic_ISM/harp_dataset_2-6kHz_multiSCM_singleSource_task_0.pt"  # <-- adjust this if needed

# Load data
data = torch.load(data_path, weights_only=False)

X = data["X"]
Y = data["Y"]

# Print shapes (optional but useful)
print("X shape:", X.shape)
print("Y shape:", Y.shape)

# Print first entries
print("\nFirst entry of X:")
print(X[0])   # shape: (n_bands, n_channels, n_channels)

print("\nFirst entry of Y:")
print(Y[0])   # shape