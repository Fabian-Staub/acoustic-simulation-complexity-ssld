import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.special import spherical_jn, spherical_yn


def plot_tikhonov_difference_filters(
    fs,
    rigid_sphere_modal_coeffs,
    R=0.042,
    Nmic=32,
    amp_threshold=15.0,
    c=343.0,
    max_order=4,
    n_fft=32768,
    f_min=20.0,
    f_max=None,
    ylim=(-80.0, 80.0),
):
    """
    Plot optimal, regularized, and difference radial filters on one axis.

    Each Ambisonic order uses the same color across all three filter types:

    - Optimal filter: dashed, faint, and drawn in the background.
    - Regularized filter: solid and moderately transparent.
    - Difference filter: solid, opaque, thicker, and drawn in the foreground.

    Parameters
    ----------
    fs : float
        Sampling rate in Hz.

    rigid_sphere_modal_coeffs : callable
        Function with signature:

            bN = rigid_sphere_modal_coeffs(max_order, kR)

        It must return an array with shape:

            (number_of_frequency_bins, max_order + 1)

        An array with the transposed shape is also accepted.

    R : float, default=0.042
        Rigid-sphere radius in metres.

    Nmic : int, default=32
        Number of microphones.

    amp_threshold : float, default=15.0
        Moreau/Daniel regularization threshold in dB.

    c : float, default=343.0
        Speed of sound in metres per second.

    max_order : int, default=4
        Highest Ambisonic order to plot.

    n_fft : int, default=32768
        FFT size used to construct the frequency grid.

    f_min : float, default=20.0
        Lowest plotted frequency in Hz.

    f_max : float or None, default=None
        Highest plotted frequency in Hz. If None, the Nyquist frequency
        is used.

    ylim : tuple of float, default=(-80.0, 80.0)
        Y-axis limits in dB.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Created figure.

    ax : matplotlib.axes.Axes
        Axis containing the combined plot.

    filters : dict
        Linear and dB filter responses.
    """

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if fs <= 0:
        raise ValueError("fs must be positive.")

    if R <= 0:
        raise ValueError("R must be positive.")

    if Nmic <= 0:
        raise ValueError("Nmic must be positive.")

    if max_order < 0:
        raise ValueError("max_order must be non-negative.")

    if n_fft < 2:
        raise ValueError("n_fft must be at least 2.")

    if c <= 0:
        raise ValueError("c must be positive.")

    if len(ylim) != 2 or ylim[0] >= ylim[1]:
        raise ValueError("ylim must contain two increasing values.")

    if f_max is None:
        f_max = fs / 2.0

    f_max = min(float(f_max), fs / 2.0)

    if not 0.0 < f_min < f_max:
        raise ValueError("Require 0 < f_min < f_max <= fs / 2.")

    # ------------------------------------------------------------------
    # Frequency grid
    # ------------------------------------------------------------------
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    kR = 2.0 * np.pi * frequencies * R / c

    # Avoid evaluating the spherical functions exactly at kR = 0.
    kR_safe = kR.copy()
    kR_safe[0] = 1e-12

    # ------------------------------------------------------------------
    # Rigid-sphere modal coefficients
    # ------------------------------------------------------------------
    bN = np.asarray(
        rigid_sphere_modal_coeffs(max_order, kR_safe),
        dtype=np.complex128,
    )

    expected_shape = (frequencies.size, max_order + 1)

    # Accommodate implementations returning (orders, frequency_bins).
    if bN.shape == expected_shape[::-1]:
        bN = bN.T

    if bN.shape != expected_shape:
        raise ValueError(
            "rigid_sphere_modal_coeffs returned shape "
            f"{bN.shape}; expected {expected_shape}."
        )

    # Remove the 4π factor used by the modal-coefficient implementation.
    bN = bN / (4.0 * np.pi)

    # Match the desired DC treatment.
    if max_order >= 1:
        bN[0, 1:] = 0.0

    # ------------------------------------------------------------------
    # Moreau/Daniel regularization parameter
    # ------------------------------------------------------------------
    alpha = np.sqrt(Nmic) * 10.0 ** (amp_threshold / 20.0)

    if alpha <= 1.0:
        raise ValueError(
            "amp_threshold is too low: the resulting alpha must exceed 1."
        )

    root_term = np.sqrt(1.0 - 1.0 / alpha**2)

    beta = np.sqrt(
        (1.0 - root_term)
        / (1.0 + root_term)
    )

    # ------------------------------------------------------------------
    # Filter definitions
    #
    # H_reg  = conj(bN) / (|bN|² + beta²)
    #
    # H_opt  = 1 / bN
    #
    # H_diff = H_reg / H_opt
    #        = H_reg * bN
    #        = |bN|² / (|bN|² + beta²)
    # ------------------------------------------------------------------
    modal_power = np.abs(bN) ** 2

    H_reg = np.conj(bN) / (modal_power + beta**2)

    H_diff = modal_power / (modal_power + beta**2)

    # The unregularized inverse can become infinite at modal zeros.
    H_opt = np.full_like(bN, np.nan + 0.0j)

    nonzero = np.abs(bN) > np.finfo(float).eps
    H_opt[nonzero] = 1.0 / bN[nonzero]

    # Order zero should pass DC in the difference filter.
    H_diff[0, 0] = 1.0

    # ------------------------------------------------------------------
    # Convert responses to decibels
    # ------------------------------------------------------------------
    def magnitude_db(response, floor_db=-300.0):
        magnitude = np.abs(response)

        with np.errstate(divide="ignore", invalid="ignore"):
            result = 20.0 * np.log10(magnitude)

        result = np.where(np.isfinite(result), result, floor_db)
        return np.maximum(result, floor_db)

    H_reg_db = magnitude_db(H_reg)
    H_opt_db = magnitude_db(H_opt)
    H_diff_db = magnitude_db(H_diff)

    # Do not include DC on a logarithmic frequency axis.
    plot_mask = (
        (frequencies >= f_min)
        & (frequencies <= f_max)
    )

    f_plot = frequencies[plot_mask]

    # ------------------------------------------------------------------
    # Combined plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(8, 4),
        constrained_layout=True,
    )

    # Obtain Matplotlib's default color cycle so each order has one color.
    #color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_cycle = [
        "tab:blue",
        "tab:red",
        "gold",
        "tab:purple",
        "tab:green",
    ]

    for order in range(max_order + 1):
        color = color_cycle[order % len(color_cycle)]

        # Optimal radial inverse:
        # faint, dashed, and behind the other curves.
        ax.semilogx(
            f_plot,
            H_opt_db[plot_mask, order],
            color=color,
            linestyle="--",
            linewidth=1.7,
            alpha=0.4,
            zorder=1,
        )

        # Regularized radial inverse:
        # solid, lighter, and above the optimal response.
        ax.semilogx(
            f_plot,
            H_reg_db[plot_mask, order],
            color=color,
            linestyle="-",
            linewidth=1.7,
            alpha=0.4,
            zorder=2,
        )

        # Difference filter:
        # opaque, thicker, and in the foreground.
        ax.semilogx(
            f_plot,
            H_diff_db[plot_mask, order],
            color=color,
            linestyle="-",
            linewidth=3.5,
            alpha=1.0,
            zorder=3,
        )

    ax.set_title(
        "Optimal, Regularized, and Difference Radial Filters"
    )
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Magnitude [dB]")

    ax.set_xlim(f_min, f_max)
    ax.set_ylim(*ylim)

    ax.grid(
        True,
        which="both",
        alpha=0.3,
    )

    # ------------------------------------------------------------------
    # Legends
    #
    # One legend identifies the colors/orders.
    # A second legend identifies the line styles/filter types.
    # ------------------------------------------------------------------
    order_handles = []

    for order in range(max_order + 1):
        color = color_cycle[order % len(color_cycle)]

        order_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linestyle="-",
                linewidth=2.6,
                label=f"Order {order}",
            )
        )

    order_legend = ax.legend(
        handles=order_handles,
        title="Ambisonic order",
        loc="upper right",
        ncol=1,
    )

    filter_type_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            linewidth=1.7,
            alpha=0.3,
            label="Optimal",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="-",
            linewidth=1.7,
            alpha=0.3,
            label="Regularized",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="-",
            linewidth=3.5,
            alpha=1.0,
            label="Difference",
        ),
    ]

    filter_type_legend = ax.legend(
        handles=filter_type_handles,
        title="Filter type",
        loc="lower left",
    )

    # Keep both legends visible.
    ax.add_artist(order_legend)

    filters = {
        "frequencies": frequencies,
        "kR": kR,
        "bN": bN,
        "alpha": alpha,
        "beta": beta,
        "H_regularized": H_reg,
        "H_optimal": H_opt,
        "H_difference": H_diff,
        "H_regularized_db": H_reg_db,
        "H_optimal_db": H_opt_db,
        "H_difference_db": H_diff_db,
    }

    return fig, ax, filters


def rigid_sphere_modal_coeffs(max_order, x):
    """
    Calculate rigid-sphere modal coefficients.

    Parameters
    ----------
    max_order : int
        Highest modal order.

    x : array_like
        Dimensionless frequency-radius product kR.

    Returns
    -------
    b : ndarray
        Complex modal coefficients with shape
        (number_of_frequency_bins, max_order + 1).
    """
    x = np.asarray(x, dtype=float)

    b = np.zeros(
        (x.size, max_order + 1),
        dtype=np.complex128,
    )

    for n in range(max_order + 1):
        jn = spherical_jn(n, x)
        jn_der = spherical_jn(n, x, derivative=True)

        yn = spherical_yn(n, x)
        yn_der = spherical_yn(n, x, derivative=True)

        hn2 = jn - 1j * yn
        hn2_der = jn_der - 1j * yn_der

        with np.errstate(
            divide="ignore",
            invalid="ignore",
            over="ignore",
        ):
            boundary_term = (jn_der / hn2_der) * hn2

        b[:, n] = (
            4.0
            * np.pi
            * (1j**n)
            * (jn - boundary_term)
        )

    return b


# ----------------------------------------------------------------------
# Generate and save the figure
# ----------------------------------------------------------------------
fig, ax, filters = plot_tikhonov_difference_filters(
    fs=48000,
    rigid_sphere_modal_coeffs=rigid_sphere_modal_coeffs,
    R=0.042,
    Nmic=32,
    amp_threshold=15.0,
    c=343.0,
    max_order=3,
    n_fft=65536,
    f_min=20.0,
    f_max=20000.0,
    ylim=(-80.0, 80.0),
)

fig.savefig(
    "radial_filters_combined.pdf",
    bbox_inches="tight",
)

plt.show()