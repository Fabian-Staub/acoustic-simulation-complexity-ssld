from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyBboxPatch


OUTPUT_PATH = Path("plot_dataset_names.pdf")


# =============================================================================
# STYLE / LAYOUT VARIABLES
# =============================================================================

# --- Page size ---------------------------------------------------------------

A4_WIDTH_IN = 210 / 25.4

# Automatic figure height
FIGURE_TOP_MARGIN_IN = 0.01
FIGURE_BOTTOM_MARGIN_IN = 0.01
GROUP_TITLE_HEIGHT_IN = 0.1
GROUP_TITLE_TO_FIRST_BOX_IN = 0.05

DATASET_BOX_HEIGHT_IN = 0.22
DATASET_BOX_VERTICAL_GAP_IN = 0.07


# --- Font sizes --------------------------------------------------------------

MAIN_TITLE_FONTSIZE = 12
GROUP_TITLE_FONTSIZE = 8
DATASET_FONTSIZE = 8
FOOTER_FONTSIZE = 8.5


# --- Horizontal box layout ---------------------------------------------------

SIDE_MARGIN = 0
COLUMN_GAP = 0.02

DATASET_BOX_WIDTH = None


# --- Box appearance ----------------------------------------------------------

BOX_FILL_ALPHA = 0.15
BOX_EDGE_ALPHA = 1

BOX_LINEWIDTH = 1
BOX_CORNER_RADIUS = 0.0


# =============================================================================
# COLORS
# =============================================================================

O0_COLORS = ["#006b6b", "#3bc2b9", "#62938f", "#a1d6d1"]
O2_COLORS = ["#8a6a00", "#e6b710", "#a28e4b", "#dfcd88"]
O12_COLORS = ["#951010", "#ec4e4e", "#a96868", "#e7a5a5"]
CORNER_COLORS = ["#0108C8", "#5C61E0", "#A9ABF6"]
RANDOM_COLORS = ["#00B00F", "#54D861", "#8FDC97", "#CCF2D1"]


# =============================================================================
# DATA
# =============================================================================

groups = [
    (
        "ISM order 0",
        [
            ("SH-O0-L0-C0-R0", O0_COLORS[0]),
            ("EM-O0-L0-C0-R0", O0_COLORS[1]),
            ("SH-O0-L1-C0-R0", O0_COLORS[2]),
            ("EM-O0-L1-C0-R0", O0_COLORS[3]),
        ],
    ),
    (
        "ISM order 2",
        [
            ("SH-O2-L0-C0-R0", O2_COLORS[0]),
            ("EM-O2-L0-C0-R0", O2_COLORS[1]),
            ("SH-O2-L1-C0-R0", O2_COLORS[2]),
            ("EM-O2-L1-C0-R0", O2_COLORS[3]),
        ],
    ),
    (
        "ISM order 12",
        [
            ("SH-O12-L0-C0-R0", O12_COLORS[0]),
            ("EM-O12-L0-C0-R0", O12_COLORS[1]),
            ("SH-O12-L1-C0-R0", O12_COLORS[2]),
            ("EM-O12-L1-C0-R0", O12_COLORS[3]),
        ],
    ),
    (
        "Corner displacement",
        [
            ("SH-O12-L0-C0.5-R0", CORNER_COLORS[0]),
            ("SH-O12-L0-C1-R0", CORNER_COLORS[1]),
            ("SH-O12-L0-C3-R0", CORNER_COLORS[2]),
        ],
    ),
    (
        "Random ISM positions",
        [
            ("SH-O12-L0-C0-R0.25", RANDOM_COLORS[0]),
            ("SH-O12-L0-C0-R1", RANDOM_COLORS[1]),
            ("SH-O12-L0-C0-R5", RANDOM_COLORS[2]),
            ("SH-O12-L0-C0-R10", RANDOM_COLORS[3]),
        ],
    ),
]


# =============================================================================
# AUTOMATIC HEIGHT
# =============================================================================

max_rows = max(len(items) for _, items in groups)

content_height_in = (
    GROUP_TITLE_HEIGHT_IN
    + GROUP_TITLE_TO_FIRST_BOX_IN
    + max_rows * DATASET_BOX_HEIGHT_IN
    + max(0, max_rows - 1) * DATASET_BOX_VERTICAL_GAP_IN
)

FIGURE_HEIGHT_IN = (
    FIGURE_TOP_MARGIN_IN
    + content_height_in
    + FIGURE_BOTTOM_MARGIN_IN
)


# Convert physical vertical dimensions to figure-relative coordinates
group_title_y = 1.0 - FIGURE_TOP_MARGIN_IN / FIGURE_HEIGHT_IN

group_title_to_first_box = (
    GROUP_TITLE_TO_FIRST_BOX_IN / FIGURE_HEIGHT_IN
)

dataset_box_h = DATASET_BOX_HEIGHT_IN / FIGURE_HEIGHT_IN
dataset_box_gap = DATASET_BOX_VERTICAL_GAP_IN / FIGURE_HEIGHT_IN

first_box_top = (
    group_title_y
    - GROUP_TITLE_HEIGHT_IN / FIGURE_HEIGHT_IN
    - group_title_to_first_box
)

row_pitch = dataset_box_h + dataset_box_gap


# =============================================================================
# HELPERS
# =============================================================================

def rounded_box(
    ax,
    x,
    y,
    w,
    h,
    color,
    fill_alpha=1.0,
    edge_alpha=1.0,
    linewidth=0.8,
    radius=0.02,
    zorder=1,
):
    """
    Draw a dataset box.
    """
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        facecolor=to_rgba(color, fill_alpha),
        edgecolor=to_rgba(color, edge_alpha),
        linewidth=linewidth,
        transform=ax.transAxes,
        clip_on=False,
        zorder=zorder,
    )

    ax.add_patch(patch)
    return patch


def format_dataset_name(dataset_name):
    """
    Convert a dataset name such as

        SH-O12-L0-C0.5-R0

    into Matplotlib mathtext corresponding approximately to

        SH - O_12 - L_0 - C_0.5 - R_0

    where the microphone name and component letters are
    bold monospaced and the values are monospaced subscripts.
    """
    parts = dataset_name.split("-")

    # First component: microphone type, e.g. SH or EM
    mic = parts[0]

    formatted_parts = [
        rf"\mathbf{{\mathtt{{{mic}}}}}"
    ]

    # Remaining components:
    # O12 -> O_{12}
    # L0  -> L_{0}
    # C0.5 -> C_{0.5}
    # R10 -> R_{10}
    for part in parts[1:]:
        label = part[0]
        value = part[1:]

        formatted_parts.append(
            rf"\mathbf{{\mathtt{{{label}}}}}_{{\mathtt{{{value}}}}}"
        )

    return "$" + r"-".join(formatted_parts) + "$"


# =============================================================================
# PLOT
# =============================================================================

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 14,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "dejavusans",
    }
)

fig = plt.figure(
    figsize=(A4_WIDTH_IN, FIGURE_HEIGHT_IN)
)

ax = fig.add_axes([0, 0, 1, 1])
ax.set_axis_off()


# -----------------------------------------------------------------------------
# Calculate horizontal positions
# -----------------------------------------------------------------------------

n_groups = len(groups)

if DATASET_BOX_WIDTH is None:
    dataset_box_w = (
        1.0
        - 2 * SIDE_MARGIN
        - (n_groups - 1) * COLUMN_GAP
    ) / n_groups
else:
    dataset_box_w = DATASET_BOX_WIDTH


total_columns_width = (
    n_groups * dataset_box_w
    + (n_groups - 1) * COLUMN_GAP
)

left = (1.0 - total_columns_width) / 2


# -----------------------------------------------------------------------------
# Groups
# -----------------------------------------------------------------------------

for i, (group_name, items) in enumerate(groups):

    x = left + i * (dataset_box_w + COLUMN_GAP)

    # Group headline
    ax.text(
        x + dataset_box_w / 2,
        group_title_y,
        group_name,
        ha="center",
        va="top",
        transform=ax.transAxes,
        fontsize=GROUP_TITLE_FONTSIZE,
        fontweight="bold",
        linespacing=1.05,
    )

    # Dataset boxes
    y_cursor = first_box_top

    for dataset_name, color in items:

        y = y_cursor - dataset_box_h

        rounded_box(
            ax,
            x,
            y,
            dataset_box_w,
            dataset_box_h,
            color=color,
            fill_alpha=BOX_FILL_ALPHA,
            edge_alpha=BOX_EDGE_ALPHA,
            linewidth=BOX_LINEWIDTH,
            radius=BOX_CORNER_RADIUS,
        )

        ax.text(
            x + dataset_box_w / 2,
            y + dataset_box_h / 2,
            format_dataset_name(dataset_name),
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=DATASET_FONTSIZE,
        )

        y_cursor -= row_pitch


# =============================================================================
# SAVE
# =============================================================================

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

fig.savefig(
    OUTPUT_PATH,
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.01,
)

plt.close(fig)

print(f"Saved: {OUTPUT_PATH.resolve()}")