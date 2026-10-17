
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Safe on headless/cluster machines.
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import sofar as sf


DEFAULT_OCTAVE_CENTRES = np.array(
    [63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000],
    dtype=float,
)


def wrap_azimuth_deg(azimuth):
    """Wrap azimuth angles to [-180, 180)."""
    azimuth = np.asarray(azimuth, dtype=float)
    return (azimuth + 180.0) % 360.0 - 180.0


def source_positions_to_spherical(sofa):
    """Return SOFA source azimuth, elevation and radius in degrees/metres."""
    positions = np.squeeze(np.asarray(sofa.SourcePosition, dtype=float))

    if positions.ndim == 1:
        positions = positions[np.newaxis, :]

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            "Expected SourcePosition with shape (M, 3), "
            f"but received {positions.shape}."
        )

    position_type = str(
        getattr(sofa, "SourcePosition_Type", "spherical")
    ).lower()
    units = str(
        getattr(sofa, "SourcePosition_Units", "degree, degree, metre")
    ).lower()

    if "cartesian" in position_type:
        x, y, z = positions.T
        radius = np.sqrt(x**2 + y**2 + z**2)
        azimuth = np.degrees(np.arctan2(y, x))
        elevation = np.degrees(
            np.arctan2(z, np.sqrt(x**2 + y**2))
        )

    elif "spherical" in position_type:
        azimuth = positions[:, 0].copy()
        elevation = positions[:, 1].copy()
        radius = positions[:, 2].copy()

        if "radian" in units:
            azimuth = np.degrees(azimuth)
            elevation = np.degrees(elevation)

    else:
        raise ValueError(
            f"Unsupported SourcePosition_Type: {position_type}"
        )

    return wrap_azimuth_deg(azimuth), elevation, radius


def read_scalar_sampling_rate(value, name):
    """Read a scalar or constant-valued sampling-rate array."""
    values = np.asarray(value, dtype=float).ravel()

    if values.size == 0:
        raise ValueError(f"{name} is empty.")

    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError(f"Invalid {name}: {values}")

    if not np.allclose(values, values[0]):
        raise ValueError(
            f"This script requires a constant sampling rate, but {name} "
            f"contains multiple values: {values}"
        )

    return float(values[0])


def load_sofa_data(sofa_path):
    """Load SOFA data into directions x receivers x samples form."""
    sofa_path = Path(sofa_path)

    if not sofa_path.is_file():
        raise FileNotFoundError(
            f"SOFA file does not exist: {sofa_path}"
        )

    sofa = sf.read_sofa(sofa_path, verify=False)
    impulse_responses = np.asarray(sofa.Data_IR, dtype=float)

    if impulse_responses.ndim != 3:
        raise ValueError(
            "Expected SOFA Data_IR with shape (M, R, N), "
            f"but received {impulse_responses.shape}."
        )

    azimuth, elevation, radius = source_positions_to_spherical(sofa)

    if azimuth.size != impulse_responses.shape[0]:
        raise ValueError(
            "The number of SOFA SourcePosition entries does not match "
            "the measurement dimension of Data_IR."
        )

    sampling_rate = read_scalar_sampling_rate(
        sofa.Data_SamplingRate,
        "SOFA Data_SamplingRate",
    )

    return {
        "path": sofa_path,
        "impulse_responses": impulse_responses,
        "fs": sampling_rate,
        "azimuth_deg": azimuth,
        "elevation_deg": elevation,
        "radius_m": radius,
        "number_of_directions": impulse_responses.shape[0],
        "number_of_receivers": impulse_responses.shape[1],
        "number_of_samples": impulse_responses.shape[2],
    }


def load_em32_npz(npz_path):
    """Load h_mics[taps, microphones, directions] from an EM32 NPZ file."""
    npz_path = Path(npz_path)

    if not npz_path.is_file():
        raise FileNotFoundError(
            f"NPZ file does not exist: {npz_path}"
        )

    with np.load(npz_path, allow_pickle=False) as npz:
        required = {
            "h_mics",
            "measurement_dirs_aziElev",
            "fs",
        }
        missing = required.difference(npz.files)

        if missing:
            raise KeyError(
                f"Missing NPZ entries: {sorted(missing)}. "
                f"Available entries: {list(npz.files)}"
            )

        h_mics = np.asarray(npz["h_mics"], dtype=float)
        measurement_dirs = np.asarray(
            npz["measurement_dirs_aziElev"],
            dtype=float,
        )
        sampling_rate = read_scalar_sampling_rate(
            npz["fs"],
            "NPZ fs",
        )

        radius = (
            float(np.asarray(npz["radius"]).squeeze())
            if "radius" in npz
            else 0.042
        )

    if h_mics.ndim != 3:
        raise ValueError(
            "Expected h_mics with shape "
            "(taps, microphones, directions), "
            f"but received {h_mics.shape}."
        )

    if measurement_dirs.ndim != 2 or measurement_dirs.shape[1] != 2:
        raise ValueError(
            "Expected measurement_dirs_aziElev with shape "
            "(directions, 2), "
            f"but received {measurement_dirs.shape}."
        )

    taps, microphones, directions = h_mics.shape

    if measurement_dirs.shape[0] != directions:
        raise ValueError(
            "The number of measurement directions does not match h_mics.\n"
            f"h_mics directions: {directions}\n"
            f"measurement directions: {measurement_dirs.shape[0]}"
        )

    # taps x microphones x directions
    # -> directions x microphones x taps
    impulse_responses = np.transpose(h_mics, (2, 1, 0))

    azimuth = wrap_azimuth_deg(
        np.degrees(measurement_dirs[:, 0])
    )
    elevation = np.degrees(measurement_dirs[:, 1])

    return {
        "path": npz_path,
        "impulse_responses": impulse_responses,
        "fs": sampling_rate,
        "azimuth_deg": azimuth,
        "elevation_deg": elevation,
        "radius_m": radius,
        "number_of_directions": directions,
        "number_of_receivers": microphones,
        "number_of_samples": taps,
    }


def select_elevation_slice(
    azimuth_deg,
    elevation_deg,
    target_elevation_deg,
    tolerance_deg,
):
    """Select measurements near the available elevation closest to target."""
    azimuth_deg = np.asarray(azimuth_deg, dtype=float)
    elevation_deg = np.asarray(elevation_deg, dtype=float)

    nearest_elevation = elevation_deg[
        np.argmin(
            np.abs(elevation_deg - target_elevation_deg)
        )
    ]

    mask = (
        np.abs(elevation_deg - nearest_elevation)
        <= tolerance_deg
    )
    indices = np.flatnonzero(mask)

    if indices.size < 3:
        raise ValueError(
            "Too few directions were found for the requested elevation.\n"
            f"Requested elevation: {target_elevation_deg:.2f} deg\n"
            f"Nearest measured elevation: {nearest_elevation:.2f} deg\n"
            f"Tolerance: {tolerance_deg:.2f} deg\n"
            f"Selected directions: {indices.size}"
        )

    order = np.argsort(azimuth_deg[indices])
    sorted_indices = indices[order]
    sorted_azimuth = azimuth_deg[sorted_indices]
    actual_elevation = float(
        np.median(elevation_deg[sorted_indices])
    )

    return (
        sorted_indices,
        sorted_azimuth,
        actual_elevation,
    )


def apply_direct_sound_window_2d(
    impulse_responses,
    sampling_rate,
    pre_peak_ms,
    post_peak_ms,
):
    """Apply a peak-relative Hann window to directions x samples data."""
    impulse_responses = np.asarray(
        impulse_responses,
        dtype=float,
    )

    if impulse_responses.ndim != 2:
        raise ValueError(
            "Expected impulse_responses with shape "
            "(directions, samples), "
            f"but received {impulse_responses.shape}."
        )

    if pre_peak_ms < 0 or post_peak_ms < 0:
        raise ValueError(
            "Window durations must not be negative."
        )

    pre_samples = int(
        round(pre_peak_ms * 1e-3 * sampling_rate)
    )
    post_samples = int(
        round(post_peak_ms * 1e-3 * sampling_rate)
    )

    windowed = np.zeros_like(impulse_responses)

    for direction_index, impulse_response in enumerate(
        impulse_responses
    ):
        peak_index = int(
            np.argmax(np.abs(impulse_response))
        )

        start = max(
            0,
            peak_index - pre_samples,
        )
        stop = min(
            impulse_response.size,
            peak_index + post_samples + 1,
        )

        length = stop - start
        window = (
            np.hanning(length)
            if length > 1
            else np.ones(length)
        )

        windowed[direction_index, start:stop] = (
            impulse_response[start:stop] * window
        )

    return windowed


def make_octave_masks(
    number_of_samples,
    sampling_rate,
    octave_centres,
):
    """Create FFT-bin masks for octave bands below Nyquist."""
    frequencies = np.fft.rfftfreq(
        number_of_samples,
        d=1.0 / sampling_rate,
    )

    nyquist = sampling_rate / 2.0
    masks = {}
    skipped = {}

    for centre in np.asarray(
        octave_centres,
        dtype=float,
    ):
        lower = centre / np.sqrt(2.0)
        upper = centre * np.sqrt(2.0)

        if upper > nyquist:
            skipped[centre] = (
                f"upper limit {upper:.1f} Hz exceeds "
                f"Nyquist frequency {nyquist:.1f} Hz"
            )
            continue

        mask = (
            (frequencies >= lower)
            & (frequencies < upper)
        )

        if not np.any(mask):
            skipped[centre] = (
                "no FFT bins inside the octave band"
            )
            continue

        masks[centre] = mask

    return masks, skipped


def calculate_all_receiver_octave_levels(
    impulse_responses,
    sampling_rate,
    octave_centres,
    direct_sound_window_ms=None,
):
    """
    Calculate octave-band levels for all receivers.

    Parameters
    ----------
    impulse_responses : ndarray
        Shape: directions x receivers x samples.

    Returns
    -------
    levels_by_band : dict
        centre_frequency -> directions x receivers array.
    skipped_bands : dict
        Requested bands that could not be calculated.
    """
    impulse_responses = np.asarray(
        impulse_responses,
        dtype=float,
    )

    if impulse_responses.ndim != 3:
        raise ValueError(
            "Expected impulse_responses with shape "
            "(directions, receivers, samples), but received "
            f"{impulse_responses.shape}."
        )

    directions, receivers, samples = (
        impulse_responses.shape
    )

    masks, skipped_bands = make_octave_masks(
        samples,
        sampling_rate,
        octave_centres,
    )

    if not masks:
        raise ValueError(
            "None of the requested octave bands can be used."
        )

    levels_by_band = {
        centre: np.empty(
            (directions, receivers),
            dtype=float,
        )
        for centre in masks
    }

    # Process one receiver at a time to avoid creating
    # one very large three-dimensional complex FFT array.
    for receiver_index in range(receivers):
        receiver_irs = impulse_responses[
            :,
            receiver_index,
            :,
        ]

        if direct_sound_window_ms is not None:
            if len(direct_sound_window_ms) != 2:
                raise ValueError(
                    "direct_sound_window_ms must be None or "
                    "(pre_peak_ms, post_peak_ms)."
                )

            receiver_irs = apply_direct_sound_window_2d(
                receiver_irs,
                sampling_rate,
                direct_sound_window_ms[0],
                direct_sound_window_ms[1],
            )

        spectrum = np.fft.rfft(
            receiver_irs,
            axis=-1,
        )
        power_spectrum = np.abs(spectrum) ** 2

        for centre, mask in masks.items():
            band_power = np.sum(
                power_spectrum[:, mask],
                axis=1,
            )

            band_power = np.maximum(
                band_power,
                np.finfo(float).tiny,
            )

            levels_by_band[centre][
                :,
                receiver_index,
            ] = 10.0 * np.log10(band_power)

    return levels_by_band, skipped_bands


def angular_distance_deg(
    azimuth_deg,
    elevation_deg,
    target_azimuth_deg,
    target_elevation_deg,
):
    """Calculate great-circle angular distance in degrees."""
    azimuth = np.deg2rad(
        np.asarray(azimuth_deg, dtype=float)
    )
    elevation = np.deg2rad(
        np.asarray(elevation_deg, dtype=float)
    )

    target_azimuth = np.deg2rad(
        float(target_azimuth_deg)
    )
    target_elevation = np.deg2rad(
        float(target_elevation_deg)
    )

    cosine_distance = (
        np.sin(elevation)
        * np.sin(target_elevation)
        + np.cos(elevation)
        * np.cos(target_elevation)
        * np.cos(azimuth - target_azimuth)
    )

    return np.degrees(
        np.arccos(
            np.clip(
                cosine_distance,
                -1.0,
                1.0,
            )
        )
    )


def find_nearest_direction(
    azimuth_deg,
    elevation_deg,
    target_azimuth_deg,
    target_elevation_deg,
):
    """Return the nearest measured direction and its angular error."""
    distances = angular_distance_deg(
        azimuth_deg,
        elevation_deg,
        target_azimuth_deg,
        target_elevation_deg,
    )

    index = int(np.argmin(distances))

    return (
        index,
        float(azimuth_deg[index]),
        float(elevation_deg[index]),
        float(distances[index]),
    )


def resolve_ir_plot_duration_ms(plot_window_ms):
    """
    Convert the IR plot-window setting to one total duration in ms.

    A scalar is interpreted as the total displayed duration.

    A two-value tuple is accepted for backward compatibility and its
    values are added. For example, (10.0, 40.0) becomes a total plot
    duration of 50 ms.
    """
    if plot_window_ms is None:
        return None

    values = np.asarray(
        plot_window_ms,
        dtype=float,
    ).ravel()

    if values.size == 1:
        duration_ms = float(values[0])

    elif values.size == 2:
        duration_ms = float(np.sum(values))

    else:
        raise ValueError(
            "ir_plot_window_ms must be None, a scalar total duration "
            "in ms, or a two-value tuple whose values sum to the "
            "total duration."
        )

    if (
        not np.isfinite(duration_ms)
        or duration_ms <= 0
    ):
        raise ValueError(
            "The IR plotting duration must be positive."
        )

    return duration_ms


def prepare_ir_for_plot(
    impulse_response,
    sampling_rate,
    plot_duration_ms,
    peak_plot_time_ms=10.0,
    normalize=True,
):
    """
    Return a fixed-duration IR segment with its peak at a chosen plot time.

    For example, with:

        plot_duration_ms=50.0
        peak_plot_time_ms=10.0

    the returned time axis spans 0–50 ms and the absolute IR peak is
    positioned at 10 ms.

    If the requested segment extends beyond the stored impulse response,
    the missing samples are filled with zeros.
    """
    impulse_response = np.asarray(
        impulse_response,
        dtype=float,
    ).ravel()

    if impulse_response.size == 0:
        raise ValueError(
            "Cannot plot an empty impulse response."
        )

    if (
        not np.isfinite(sampling_rate)
        or sampling_rate <= 0
    ):
        raise ValueError(
            "sampling_rate must be positive."
        )

    if (
        not np.isfinite(plot_duration_ms)
        or plot_duration_ms <= 0
    ):
        raise ValueError(
            "plot_duration_ms must be positive."
        )

    if (
        not np.isfinite(peak_plot_time_ms)
        or peak_plot_time_ms < 0
        or peak_plot_time_ms > plot_duration_ms
    ):
        raise ValueError(
            "peak_plot_time_ms must be inside the plotting window.\n"
            f"Plot duration: {plot_duration_ms:g} ms\n"
            f"Peak position: {peak_plot_time_ms:g} ms"
        )

    peak_index = int(
        np.argmax(np.abs(impulse_response))
    )

    # The extra sample allows the time axis to include both endpoints,
    # for example exactly 0 ms and 50 ms.
    output_samples = (
        int(
            round(
                plot_duration_ms
                * 1e-3
                * sampling_rate
            )
        )
        + 1
    )

    peak_output_index = int(
        round(
            peak_plot_time_ms
            * 1e-3
            * sampling_rate
        )
    )

    peak_output_index = int(
        np.clip(
            peak_output_index,
            0,
            output_samples - 1,
        )
    )

    plotted_ir = np.zeros(
        output_samples,
        dtype=float,
    )

    # Select the source interval so the detected source peak lands at
    # peak_output_index in the output signal.
    source_start = peak_index - peak_output_index
    source_stop = source_start + output_samples

    clipped_source_start = max(
        0,
        source_start,
    )
    clipped_source_stop = min(
        impulse_response.size,
        source_stop,
    )

    destination_start = (
        clipped_source_start - source_start
    )
    destination_stop = (
        destination_start
        + clipped_source_stop
        - clipped_source_start
    )

    if clipped_source_stop > clipped_source_start:
        plotted_ir[
            destination_start:destination_stop
        ] = impulse_response[
            clipped_source_start:clipped_source_stop
        ]

    time_ms = (
        np.arange(
            output_samples,
            dtype=float,
        )
        * 1000.0
        / sampling_rate
    )

    if normalize:
        plotted_peak = np.max(
            np.abs(plotted_ir)
        )

        if plotted_peak > 0:
            plotted_ir /= plotted_peak

    return time_ms, plotted_ir


def style_polar_axis(
    axis,
    radial_range_db,
):
    """Apply compact styling suitable for a 32-row figure."""
    radial_ticks_db = np.array(
        [
            -radial_range_db,
            -20.0,
            -10.0,
            0.0,
        ]
    )

    radial_ticks_db = np.unique(
        np.clip(
            radial_ticks_db,
            -radial_range_db,
            0.0,
        )
    )

    radial_positions = (
        radial_ticks_db + radial_range_db
    )

    axis.set_theta_zero_location("N")
    axis.set_theta_direction(1)

    axis.set_thetagrids(
        [0, 90, 180, 270],
        labels=[
            "0°",
            "+90°",
            "180°",
            "-90°",
        ],
        fontsize=5,
    )

    axis.set_ylim(
        0.0,
        radial_range_db,
    )

    axis.set_yticks(
        radial_positions
    )

    axis.set_yticklabels(
        [
            f"{value:.0f}"
            for value in radial_ticks_db
        ],
        fontsize=5,
    )

    axis.set_rlabel_position(225)
    axis.grid(
        True,
        linewidth=0.45,
        alpha=0.7,
    )


def plot_receiver_directivity(
    axis,
    sorted_azimuth_deg,
    levels_by_band,
    receiver_index,
    radial_range_db,
    octave_centres,
    band_colours,
):
    """Plot independently normalized octave bands for one receiver."""
    closed_azimuth = np.concatenate(
        [
            sorted_azimuth_deg,
            sorted_azimuth_deg[:1] + 360.0,
        ]
    )

    theta = np.deg2rad(
        closed_azimuth
    )

    for centre in octave_centres:
        if centre not in levels_by_band:
            continue

        levels = levels_by_band[
            centre
        ][:, receiver_index]

        normalized = levels - np.max(levels)

        normalized = np.clip(
            normalized,
            -radial_range_db,
            0.0,
        )

        closed_levels = np.concatenate(
            [
                normalized,
                normalized[:1],
            ]
        )

        radius = (
            closed_levels + radial_range_db
        )

        axis.plot(
            theta,
            radius,
            linewidth=0.9,
            color=band_colours[centre],
        )

    style_polar_axis(
        axis,
        radial_range_db,
    )


def compare_sofa_and_npz_all_receivers(
    sofa_path,
    npz_path,
    output_path="sofa_npz_all_32_mics.png",
    number_of_microphones=32,
    target_elevation_deg=0.0,
    elevation_tolerance_deg=1.0,
    octave_centres=None,
    radial_range_db=30.0,
    direct_sound_window_ms=None,
    ir_target_azimuth_deg=0.0,
    ir_target_elevation_deg=0.0,
    ir_plot_window_ms=50.0,
    ir_peak_plot_time_ms=10.0,
    normalize_ir_for_plot=True,
    figure_width_in=30.0,
    row_height_in=3.25,
    dpi=120,
    show=False,
):
    """
    Save one large PNG comparing SOFA and NPZ data for all microphones.

    Each microphone occupies one row with four panels:

        1. SOFA octave-band directivity
        2. NPZ octave-band directivity
        3. SOFA impulse response
        4. NPZ impulse response

    Each displayed IR uses the same duration. The detected absolute peak
    is placed at ir_peak_plot_time_ms inside the plot window.

    For the default settings, the plots span 0–50 ms and the detected
    IR peak is placed at 10 ms.

    Directivity bands are normalized independently to 0 dB for each
    microphone and each input file.
    """
    if octave_centres is None:
        octave_centres = (
            DEFAULT_OCTAVE_CENTRES.copy()
        )
    else:
        octave_centres = np.asarray(
            octave_centres,
            dtype=float,
        )

    if number_of_microphones <= 0:
        raise ValueError(
            "number_of_microphones must be positive."
        )

    if radial_range_db <= 0:
        raise ValueError(
            "radial_range_db must be positive."
        )

    sofa = load_sofa_data(
        sofa_path
    )
    npz = load_em32_npz(
        npz_path
    )

    if (
        sofa["number_of_receivers"]
        < number_of_microphones
    ):
        raise ValueError(
            f"SOFA contains only "
            f"{sofa['number_of_receivers']} receivers; "
            f"{number_of_microphones} were requested."
        )

    if (
        npz["number_of_receivers"]
        < number_of_microphones
    ):
        raise ValueError(
            f"NPZ contains only "
            f"{npz['number_of_receivers']} microphones; "
            f"{number_of_microphones} were requested."
        )

    (
        sofa_slice_indices,
        sofa_slice_azimuth,
        sofa_slice_elevation,
    ) = select_elevation_slice(
        sofa["azimuth_deg"],
        sofa["elevation_deg"],
        target_elevation_deg,
        elevation_tolerance_deg,
    )

    (
        npz_slice_indices,
        npz_slice_azimuth,
        npz_slice_elevation,
    ) = select_elevation_slice(
        npz["azimuth_deg"],
        npz["elevation_deg"],
        target_elevation_deg,
        elevation_tolerance_deg,
    )

    sofa_slice_irs = sofa[
        "impulse_responses"
    ][
        sofa_slice_indices,
        :number_of_microphones,
        :,
    ]

    npz_slice_irs = npz[
        "impulse_responses"
    ][
        npz_slice_indices,
        :number_of_microphones,
        :,
    ]

    print(
        "Calculating SOFA octave-band levels..."
    )

    (
        sofa_levels,
        sofa_skipped,
    ) = calculate_all_receiver_octave_levels(
        sofa_slice_irs,
        sofa["fs"],
        octave_centres,
        direct_sound_window_ms=direct_sound_window_ms,
    )

    print(
        "Calculating NPZ octave-band levels..."
    )

    (
        npz_levels,
        npz_skipped,
    ) = calculate_all_receiver_octave_levels(
        npz_slice_irs,
        npz["fs"],
        octave_centres,
        direct_sound_window_ms=direct_sound_window_ms,
    )

    sofa_ir_direction = find_nearest_direction(
        sofa["azimuth_deg"],
        sofa["elevation_deg"],
        ir_target_azimuth_deg,
        ir_target_elevation_deg,
    )

    npz_ir_direction = find_nearest_direction(
        npz["azimuth_deg"],
        npz["elevation_deg"],
        ir_target_azimuth_deg,
        ir_target_elevation_deg,
    )

    (
        sofa_ir_index,
        sofa_ir_az,
        sofa_ir_el,
        sofa_ir_error,
    ) = sofa_ir_direction

    (
        npz_ir_index,
        npz_ir_az,
        npz_ir_el,
        npz_ir_error,
    ) = npz_ir_direction

    shared_ir_duration_ms = (
        resolve_ir_plot_duration_ms(
            ir_plot_window_ms
        )
    )

    if shared_ir_duration_ms is None:
        # If complete IRs are requested, use the shorter duration so both
        # files still display exactly the same amount of time.
        sofa_duration_ms = (
            sofa["number_of_samples"]
            * 1000.0
            / sofa["fs"]
        )

        npz_duration_ms = (
            npz["number_of_samples"]
            * 1000.0
            / npz["fs"]
        )

        shared_ir_duration_ms = min(
            sofa_duration_ms,
            npz_duration_ms,
        )

    if (
        not np.isfinite(ir_peak_plot_time_ms)
        or ir_peak_plot_time_ms < 0
        or ir_peak_plot_time_ms
        > shared_ir_duration_ms
    ):
        raise ValueError(
            "ir_peak_plot_time_ms must lie inside the IR plot window.\n"
            f"Plot duration: {shared_ir_duration_ms:g} ms\n"
            f"Requested peak position: "
            f"{ir_peak_plot_time_ms:g} ms"
        )

    # Both IR panels use exactly the same displayed range.
    shared_ir_xlim_ms = (
        0.0,
        shared_ir_duration_ms,
    )

    # Use one consistent colour for each octave band in every polar panel.
    colour_map = plt.get_cmap(
        "tab10"
    )

    band_colours = {
        centre: colour_map(index % 10)
        for index, centre
        in enumerate(octave_centres)
    }

    figure_height_in = (
        2.0
        + row_height_in
        * number_of_microphones
    )

    figure = plt.figure(
        figsize=(
            figure_width_in,
            figure_height_in,
        ),
        facecolor="white",
    )

    grid = figure.add_gridspec(
        nrows=number_of_microphones,
        ncols=4,
        width_ratios=[
            1.0,
            1.0,
            1.35,
            1.35,
        ],
        hspace=0.52,
        wspace=0.32,
        left=0.055,
        right=0.985,
        bottom=0.012,
        top=0.965,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=band_colours[centre],
            linewidth=1.5,
            label=f"{centre:g} Hz",
        )
        for centre in octave_centres
        if (
            centre in sofa_levels
            or centre in npz_levels
        )
    ]

    for microphone_index in range(
        number_of_microphones
    ):
        axis_sofa_directivity = figure.add_subplot(
            grid[microphone_index, 0],
            projection="polar",
        )

        axis_npz_directivity = figure.add_subplot(
            grid[microphone_index, 1],
            projection="polar",
        )

        axis_sofa_ir = figure.add_subplot(
            grid[microphone_index, 2]
        )

        axis_npz_ir = figure.add_subplot(
            grid[microphone_index, 3]
        )

        plot_receiver_directivity(
            axis_sofa_directivity,
            sofa_slice_azimuth,
            sofa_levels,
            microphone_index,
            radial_range_db,
            octave_centres,
            band_colours,
        )

        plot_receiver_directivity(
            axis_npz_directivity,
            npz_slice_azimuth,
            npz_levels,
            microphone_index,
            radial_range_db,
            octave_centres,
            band_colours,
        )

        sofa_time_ms, sofa_ir = (
            prepare_ir_for_plot(
                sofa["impulse_responses"][
                    sofa_ir_index,
                    microphone_index,
                    :,
                ],
                sofa["fs"],
                plot_duration_ms=(
                    shared_ir_duration_ms
                ),
                peak_plot_time_ms=(
                    ir_peak_plot_time_ms
                ),
                normalize=(
                    normalize_ir_for_plot
                ),
            )
        )

        npz_time_ms, npz_ir = (
            prepare_ir_for_plot(
                npz["impulse_responses"][
                    npz_ir_index,
                    microphone_index,
                    :,
                ],
                npz["fs"],
                plot_duration_ms=(
                    shared_ir_duration_ms
                ),
                peak_plot_time_ms=(
                    ir_peak_plot_time_ms
                ),
                normalize=(
                    normalize_ir_for_plot
                ),
            )
        )

        axis_sofa_ir.plot(
            sofa_time_ms,
            sofa_ir,
            linewidth=0.75,
        )

        axis_npz_ir.plot(
            npz_time_ms,
            npz_ir,
            linewidth=0.75,
        )

        for axis in (
            axis_sofa_ir,
            axis_npz_ir,
        ):
            axis.grid(
                True,
                linewidth=0.45,
                alpha=0.65,
            )

            axis.tick_params(
                axis="both",
                labelsize=5,
            )

            axis.set_xlim(
                *shared_ir_xlim_ms
            )

            if normalize_ir_for_plot:
                axis.set_ylim(
                    -1.05,
                    1.05,
                )

            # Dashed line marks the requested peak location at 10 ms.
            axis.axvline(
                ir_peak_plot_time_ms,
                linewidth=0.5,
                linestyle="--",
                alpha=0.65,
            )

        microphone_label = (
            f"Mic {microphone_index + 1:02d}"
        )

        axis_sofa_directivity.text(
            -0.22,
            0.5,
            microphone_label,
            transform=(
                axis_sofa_directivity.transAxes
            ),
            rotation=90,
            va="center",
            ha="center",
            fontsize=8,
            fontweight="bold",
        )

        if microphone_index == 0:
            axis_sofa_directivity.set_title(
                "Ivan directivity\n"
                f"Elevation "
                f"{sofa_slice_elevation:.1f}°",
                fontsize=8,
                pad=10,
            )

            axis_npz_directivity.set_title(
                "Akis directivity\n"
                f"Elevation "
                f"{npz_slice_elevation:.1f}°",
                fontsize=8,
                pad=10,
            )

            axis_sofa_ir.set_title(
                "Ivan IR\n"
                f"Nearest direction: "
                f"az {sofa_ir_az:.1f}°, "
                f"el {sofa_ir_el:.1f}°",
                fontsize=8,
            )

            axis_npz_ir.set_title(
                "Akis IR\n"
                f"Nearest direction: "
                f"az {npz_ir_az:.1f}°, "
                f"el {npz_ir_el:.1f}°",
                fontsize=8,
            )

        if (
            microphone_index
            == number_of_microphones - 1
        ):
            axis_sofa_ir.set_xlabel(
                "Time (ms)",
                fontsize=6,
            )

            axis_npz_ir.set_xlabel(
                "Time (ms)",
                fontsize=6,
            )

        amplitude_label = (
            "Normalized amplitude"
            if normalize_ir_for_plot
            else "Amplitude"
        )

        axis_sofa_ir.set_ylabel(
            amplitude_label,
            fontsize=6,
        )

        axis_npz_ir.set_ylabel(
            amplitude_label,
            fontsize=6,
        )

    window_description = (
        "complete IRs"
        if direct_sound_window_ms is None
        else (
            "directivity window: "
            f"{direct_sound_window_ms[0]:g} ms before "
            f"and {direct_sound_window_ms[1]:g} ms "
            "after each peak"
        )
    )

    figure.suptitle(
        "SOFA versus NPZ: octave-band directivity "
        "and impulse responses\n"
        f"{number_of_microphones} microphones | "
        f"IR plot range: 0–{shared_ir_duration_ms:g} ms | "
        f"IR peak positioned at "
        f"{ir_peak_plot_time_ms:g} ms | "
        f"directivity uses {window_description} | "
        "each octave band normalized independently to 0 dB",
        fontsize=12,
        y=0.998,
    )

    if legend_handles:
        figure.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(
                0.5,
                0.978,
            ),
            ncol=min(
                len(legend_handles),
                9,
            ),
            fontsize=7,
            frameon=True,
        )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Saving large PNG to: {output_path}"
    )

    figure.savefig(
        output_path,
        dpi=dpi,
        facecolor="white",
    )

    if show:
        plt.show()
    else:
        plt.close(
            figure
        )

    print(
        "\n================ SUMMARY ================"
    )

    print(
        "SOFA:",
        sofa["path"],
    )
    print(
        "  IR shape:",
        sofa["impulse_responses"].shape,
    )
    print(
        "  Sampling rate:",
        sofa["fs"],
        "Hz",
    )
    print(
        "  Selected directivity directions:",
        sofa_slice_indices.size,
    )
    print(
        "  Selected elevation:",
        sofa_slice_elevation,
        "deg",
    )
    print(
        "  IR direction:",
        f"az={sofa_ir_az:.2f} deg, "
        f"el={sofa_ir_el:.2f} deg,",
        f"angular error={sofa_ir_error:.2f} deg",
    )

    print(
        "NPZ:",
        npz["path"],
    )
    print(
        "  IR shape:",
        npz["impulse_responses"].shape,
    )
    print(
        "  Sampling rate:",
        npz["fs"],
        "Hz",
    )
    print(
        "  Selected directivity directions:",
        npz_slice_indices.size,
    )
    print(
        "  Selected elevation:",
        npz_slice_elevation,
        "deg",
    )
    print(
        "  IR direction:",
        f"az={npz_ir_az:.2f} deg, "
        f"el={npz_ir_el:.2f} deg,",
        f"angular error={npz_ir_error:.2f} deg",
    )

    print(
        "Shared IR duration:",
        shared_ir_duration_ms,
        "ms",
    )
    print(
        "IR peak plot position:",
        ir_peak_plot_time_ms,
        "ms",
    )
    print(
        "Shared IR x limits:",
        shared_ir_xlim_ms,
        "ms",
    )
    print(
        "Output:",
        output_path.resolve(),
    )

    if sofa_skipped:
        print(
            "SOFA skipped octave bands:"
        )

        for centre, reason in sofa_skipped.items():
            print(
                f"  {centre:g} Hz: {reason}"
            )

    if npz_skipped:
        print(
            "NPZ skipped octave bands:"
        )

        for centre, reason in npz_skipped.items():
            print(
                f"  {centre:g} Hz: {reason}"
            )

    return output_path


if __name__ == "__main__":
    SOFA_PATH = (
        "/scratch/elec/t412-techpsychoacoustics/msc-thesis/"
        "2026_fstaub/staubf1/Projects/Master_Thesis/src/"
        "HARP-main/v1/em32_test001_20260702.sofa"
    )

    NPZ_PATH = "Eigenmike_em32_IRs.npz"
    OUTPUT_PATH = "em32_directivities.png"

    compare_sofa_and_npz_all_receivers(
        sofa_path=SOFA_PATH,
        npz_path=NPZ_PATH,
        output_path=OUTPUT_PATH,
        number_of_microphones=32,
        target_elevation_deg=0.0,
        elevation_tolerance_deg=1.0,
        octave_centres=DEFAULT_OCTAVE_CENTRES,
        radial_range_db=30.0,

        # Directivity calculation:
        # None means analyse the complete IR/filter.
        direct_sound_window_ms=None,

        # Example direct-sound window:
        # direct_sound_window_ms=(2.0, 50.0),

        # Plot the measured direction nearest this requested direction.
        ir_target_azimuth_deg=0.0,
        ir_target_elevation_deg=0.0,

        # Display the IR from 0 to 50 ms.
        ir_plot_window_ms=50.0,

        # Position the detected absolute IR peak at 10 ms.
        # This gives 10 ms before the peak and 40 ms after the peak.
        ir_peak_plot_time_ms=10.0,

        normalize_ir_for_plot=True,

        # The output is intentionally tall: 32 rows x 4 panels.
        figure_width_in=20.0,
        row_height_in=2.0,
        dpi=150,

        # Keep False on a cluster; the PNG is still saved.
        show=False,
    )
