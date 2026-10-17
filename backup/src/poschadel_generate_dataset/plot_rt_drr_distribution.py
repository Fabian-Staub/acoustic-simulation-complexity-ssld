from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_rt_drr_distributions(
    csv_files,
    column_titles=None,
    rt_col="RT",
    drr_col="drr_db",
    bins=100,
    density=False,
    save_path=None,
):
    """
    Plot RT and DRR distributions for multiple CSV files.

    All RT plots use the same:
        - x-axis range
        - histogram bin edges

    All DRR plots use the same:
        - x-axis range
        - histogram bin edges

    Parameters
    ----------
    csv_files : list[str] or list[Path]
        List of CSV file paths.

    column_titles : list[str] or None
        Custom headings for the columns.
        If None, CSV file stems are used.

    rt_col : str
        Column name containing RT values.

    drr_col : str
        Column name containing DRR values.

    bins : int
        Number of histogram bins.

    density : bool
        If True, plot probability density instead of sample count.

    save_path : str or Path or None
        If given, save the figure to this path.
    """

    csv_files = [Path(f) for f in csv_files]
    n_files = len(csv_files)

    if n_files == 0:
        raise ValueError("csv_files must contain at least one file.")

    # =========================================================
    # Font sizes
    # =========================================================
    TITLE_FONTSIZE = 18
    AXIS_LABEL_FONTSIZE = 16
    TICK_FONTSIZE = 14
    ROW_LABEL_FONTSIZE = 17

    # ---------------------------------------------------------
    # Column headings
    # ---------------------------------------------------------
    if column_titles is None:
        column_titles = [f.stem for f in csv_files]

    if len(column_titles) != n_files:
        raise ValueError(
            "column_titles must have the same length as csv_files."
        )

    # ---------------------------------------------------------
    # Load CSV files
    # ---------------------------------------------------------
    dataframes = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        missing = [
            col
            for col in [rt_col, drr_col]
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{csv_file} is missing required columns: {missing}"
            )

        dataframes.append(df)

    # ---------------------------------------------------------
    # Collect all RT and DRR values
    # ---------------------------------------------------------
    all_rt = np.concatenate([
        df[rt_col].dropna().to_numpy()
        for df in dataframes
    ])

    all_drr = np.concatenate([
        df[drr_col].dropna().to_numpy()
        for df in dataframes
    ])

    if len(all_rt) == 0:
        raise ValueError(
            f"No valid values found in column '{rt_col}'."
        )

    if len(all_drr) == 0:
        raise ValueError(
            f"No valid values found in column '{drr_col}'."
        )

    # ---------------------------------------------------------
    # Global x ranges
    # ---------------------------------------------------------
    rt_min = all_rt.min()
    rt_max = all_rt.max()

    drr_min = all_drr.min()
    drr_max = all_drr.max()

    # ---------------------------------------------------------
    # Common histogram bins
    # ---------------------------------------------------------
    rt_bins = np.linspace(
        rt_min,
        rt_max,
        bins + 1,
    )

    drr_bins = np.linspace(
        drr_min,
        drr_max,
        bins + 1,
    )

    # ---------------------------------------------------------
    # Create figure
    #
    # rows:
    #   0 = RT
    #   1 = DRR
    #
    # columns:
    #   one per CSV
    # ---------------------------------------------------------
    fig, axes = plt.subplots(
        nrows=2,
        ncols=n_files,
        figsize=(5.5 * n_files, 7),
        squeeze=False,
        sharex=False,
        sharey=False,
    )

    # ---------------------------------------------------------
    # Plot each dataset
    # ---------------------------------------------------------
    for col_idx, (df, title) in enumerate(
        zip(dataframes, column_titles)
    ):

        rt = df[rt_col].dropna().to_numpy()
        drr = df[drr_col].dropna().to_numpy()

        # =====================================================
        # RT distribution
        # =====================================================
        ax_rt = axes[0, col_idx]

        ax_rt.hist(
            rt,
            bins=rt_bins,
            density=density,
            alpha=1,
        )

        ax_rt.set_xlim(
            rt_min,
            rt_max,
        )

        ax_rt.set_title(
            title,
            fontsize=TITLE_FONTSIZE,
            fontweight="bold",
            pad=12,
        )

        ax_rt.set_xlabel(
            "RT [s]",
            fontsize=AXIS_LABEL_FONTSIZE,
        )

        if col_idx == 0:
            ax_rt.set_ylabel(
                "Density" if density else "Count",
                fontsize=AXIS_LABEL_FONTSIZE,
            )

        ax_rt.tick_params(
            axis="both",
            labelsize=TICK_FONTSIZE,
        )

        ax_rt.grid(
            True,
            alpha=0.25,
            linestyle="--",
        )

        # =====================================================
        # DRR distribution
        # =====================================================
        ax_drr = axes[1, col_idx]

        ax_drr.hist(
            drr,
            bins=drr_bins,
            density=density,
            alpha=1,
        )

        ax_drr.set_xlim(
            drr_min,
            drr_max,
        )

        ax_drr.set_xlabel(
            "DRR [dB]",
            fontsize=AXIS_LABEL_FONTSIZE,
        )

        if col_idx == 0:
            ax_drr.set_ylabel(
                "Density" if density else "Count",
                fontsize=AXIS_LABEL_FONTSIZE,
            )

        ax_drr.tick_params(
            axis="both",
            labelsize=TICK_FONTSIZE,
        )

        ax_drr.grid(
            True,
            alpha=0.25,
            linestyle="--",
        )

    # ---------------------------------------------------------
    # Row labels
    #
    # Increased x value moves them closer to the plots.
    # ---------------------------------------------------------
    fig.text(
        0.022,
        0.73,
        "RT distribution",
        rotation=90,
        va="center",
        ha="center",
        fontsize=ROW_LABEL_FONTSIZE,
        fontweight="bold",
    )

    fig.text(
        0.022,
        0.27,
        "DRR distribution",
        rotation=90,
        va="center",
        ha="center",
        fontsize=ROW_LABEL_FONTSIZE,
        fontweight="bold",
    )

    # ---------------------------------------------------------
    # Layout
    #
    # Small left margin keeps row labels close to plots.
    # w_pad controls horizontal spacing between columns.
    # h_pad controls vertical spacing between RT and DRR.
    # ---------------------------------------------------------
    fig.tight_layout(
        rect=[0.035, 0.02, 0.995, 0.99],
        w_pad=5,
        h_pad=2.5,
    )

    # ---------------------------------------------------------
    # Save if requested
    # ---------------------------------------------------------
    if save_path is not None:
        save_path = Path(save_path)

        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

        print(f"Saved figure to: {save_path}")

    plt.show()


# =============================================================
# Example
# =============================================================

csv_files = [
    "/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/arni_em32/poschadel/RIRs/02_metadata_T60_DRR.csv",
    "/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/ivan_em32/poschadel/RIRs/02_metadata_T60_DRR.csv",
    "/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/motus/poschadel/RIRs/02_metadata_T60_DRR.csv",
]

column_titles = [
    "Arni",
    "Agrid",
    "Motus",
]

plot_rt_drr_distributions(
    csv_files,
    column_titles=column_titles,
    bins=50,
    density=False,
    save_path="rt_drr_distributions.pdf",
)