import re
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})


def parse_training_log(log_file):
    """
    Parse lines like:
    Epoch 001/10000 | train=0.059521 | val=0.044029 | lr=0.0010000
    """

    pattern = re.compile(
        r"Epoch\s+(\d+)/\d+\s+\|\s+"
        r"train=([0-9.eE+-]+)\s+\|\s+"
        r"val=([0-9.eE+-]+)\s+\|\s+"
        r"lr=([0-9.eE+-]+)"
    )

    epochs = []
    train_losses = []
    val_losses = []
    learning_rates = []

    with open(log_file, "r") as f:
        for line in f:
            match = pattern.search(line)

            if match:
                epochs.append(int(match.group(1)))
                train_losses.append(float(match.group(2)))
                val_losses.append(float(match.group(3)))
                learning_rates.append(float(match.group(4)))

    if not epochs:
        raise ValueError(
            f"No epoch information was found in {log_file}. "
            "Check that the log format matches the regex."
        )

    return epochs, train_losses, val_losses, learning_rates


def plot_training_log(log_file, output_file="training_curves.png"):
    epochs, train_losses, val_losses, learning_rates = parse_training_log(
        log_file
    )

    fig, ax_loss = plt.subplots(figsize=(8, 4))

    # Train / validation loss
    ax_loss.plot(
        epochs,
        train_losses,
        label="Train loss",
        linewidth=2,
    )

    ax_loss.plot(
        epochs,
        val_losses,
        label="Validation loss",
        linewidth=2,
    )

    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training / Validation Loss and Learning Rate")
    ax_loss.grid(True, alpha=0.3)

    # Learning rate on second y-axis
    ax_lr = ax_loss.twinx()

    ax_lr.step(
        epochs,
        learning_rates,
        where="post",
        linewidth=2,
        linestyle="--",
        label="Learning rate",
    )

    ax_lr.set_ylabel("Learning rate")
    ax_lr.set_yscale("log")

    # Combined legend
    handles_loss, labels_loss = ax_loss.get_legend_handles_labels()
    handles_lr, labels_lr = ax_lr.get_legend_handles_labels()

    ax_loss.legend(
        handles_loss + handles_lr,
        labels_loss + labels_lr,
        loc="best",
    )

    plt.tight_layout()

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    print(f"Saved plot to: {output_file}")

    plt.show()


if __name__ == "__main__":
    log_file = (
        "/scratch/elec/t412-techpsychoacoustics/msc-thesis/"
        "2026_fstaub/staubf1/Projects/Master_Thesis/src/"
        "poschadel/model/checkpoints/logs/"
        "train_shoeBox_em32_ISMOrder12.out"
    )

    plot_training_log(
        log_file,
        output_file="training_loss_and_lr.pdf",
    )