from pathlib import Path

import torch
import matplotlib.pyplot as plt


# =========================================================
# CONFIG: edit these directly
# =========================================================
data_path = "/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/harp/R2000_P10_S5_Order4_24kHz_shoeBox_rtProfilesMIT_ISM/hopiv_timit_1s_random_window_pt_debug"   # directory of .pt files or one .pt file
sample_index = 0
save_png = "first_time_frame.png"
# =========================================================


def load_sample(data_path, sample_index=0):
    data_path = Path(data_path)

    if data_path.is_dir():
        pt_files = sorted(data_path.rglob("*.pt"))

        if not pt_files:
            raise FileNotFoundError(f"No .pt files found in {data_path}")

        sample_path = pt_files[sample_index]
        sample = torch.load(sample_path, map_location="cpu")

        print(f"Loaded file: {sample_path}")

    elif data_path.is_file():
        obj = torch.load(data_path, map_location="cpu")

        if isinstance(obj, dict) and "X" in obj and "Y" in obj:
            sample = obj
            print(f"Loaded single-sample file: {data_path}")

        elif isinstance(obj, (list, tuple)):
            sample = obj[sample_index]
            print(f"Loaded sample {sample_index} from container file: {data_path}")

        else:
            raise TypeError(
                "Expected .pt file to contain either a dict with keys 'X' and 'Y', "
                "or a list/tuple of such dicts."
            )

    else:
        raise FileNotFoundError(f"Path does not exist: {data_path}")

    if not isinstance(sample, dict):
        raise TypeError("Loaded sample is not a dict.")

    return sample


def print_dimensions(sample):
    print("\n===== SAMPLE KEYS =====")
    print(list(sample.keys()))

    print("\n===== DIMENSIONS =====")

    for key, value in sample.items():
        if torch.is_tensor(value):
            print(f"{key}: shape={tuple(value.shape)}, dtype={value.dtype}")
        elif isinstance(value, dict):
            print(f"{key}: dict with keys={list(value.keys())}")
        else:
            print(f"{key}: type={type(value).__name__}")

    X = sample["X"]
    Y = sample["Y"]

    print("\n===== EXPECTED INTERPRETATION =====")
    print(f"X shape: {tuple(X.shape)}")

    if X.ndim == 3:
        T, F, C = X.shape
        print(f"  T/time frames   = {T}")
        print(f"  F/freq bins     = {F}")
        print(f"  C/channels      = {C}")
    else:
        print("  WARNING: expected X to have shape (T, F, C)")

    print(f"\nY shape: {tuple(Y.shape)}")

    if Y.ndim == 3:
        T_y, S, D = Y.shape
        print(f"  T/time frames   = {T_y}")
        print(f"  S/source tracks = {S}")
        print(f"  D/coordinates   = {D}")
    else:
        print("  WARNING: expected Y to have shape (T, tracks, 3)")


def plot_first_time_frame(sample, save_png):
    X = sample["X"]

    if not torch.is_tensor(X):
        X = torch.as_tensor(X)

    if X.ndim != 3:
        raise ValueError(f"Expected X shape (T,F,C), got {tuple(X.shape)}")

    # First time frame: shape (F, C)
    first_frame = X[0].detach().cpu().numpy()

    print("\n===== FIRST TIME FRAME =====")
    print(f"X[0] shape: {first_frame.shape}  # (freq bins, channels)")

    plt.figure(figsize=(12, 6))
    plt.imshow(
        first_frame.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )
    plt.colorbar(label="Feature value")
    plt.xlabel("Frequency bin")
    plt.ylabel("Feature channel")
    plt.title("First time frame of X: X[0] with shape (F, C)")
    plt.tight_layout()
    plt.savefig(save_png, dpi=200)
    plt.close()

    print(f"Saved plot to: {save_png}")


if __name__ == "__main__":
    sample = load_sample(data_path, sample_index=sample_index)
    print_dimensions(sample)
    plot_first_time_frame(sample, save_png)