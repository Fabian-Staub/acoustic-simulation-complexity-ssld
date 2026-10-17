from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_em32_npz(npz_path):
    """
    Load Eigenmike_em32_IRs.npz for directivity analysis.

    Expected NPZ entries
    --------------------
    h_mics : ndarray, shape (taps, microphones, directions)
        Eigenmike impulse responses / FIR filters.

    measurement_dirs_aziElev : ndarray, shape (directions, 2)
        Measurement azimuth and elevation in radians.

    fs : scalar
        Sampling rate in hertz.

    Optional entries
    ----------------
    eigen_dirs_aziElev : ndarray, shape (microphones, 2)
        Eigenmike capsule directions in radians.

    radius : scalar
        Array radius in metres.

    Returns
    -------
    em32 : dict
        Dictionary containing the data in plotting-friendly form. In
        particular, ``impulse_responses`` has shape
        (directions, microphones, taps).
    """
    npz_path = Path(npz_path)

    if not npz_path.is_file():
        raise FileNotFoundError(f"NPZ file does not exist: {npz_path}")

    with np.load(npz_path, allow_pickle=False) as npz:
        available_keys = list(npz.files)
        required_keys = {
            "h_mics",
            "measurement_dirs_aziElev",
            "fs",
        }
        missing_keys = required_keys.difference(available_keys)

        if missing_keys:
            raise KeyError(
                "The NPZ file is missing required entries: "
                f"{sorted(missing_keys)}. Available entries: "
                f"{available_keys}"
            )

        h_mics = np.asarray(npz["h_mics"], dtype=float)
        measurement_dirs = np.asarray(
            npz["measurement_dirs_aziElev"],
            dtype=float,
        )
        sampling_rate = float(np.asarray(npz["fs"]).squeeze())

        eigen_dirs = (
            np.asarray(npz["eigen_dirs_aziElev"], dtype=float)
            if "eigen_dirs_aziElev" in npz
            else None
        )
        radius = (
            float(np.asarray(npz["radius"]).squeeze())
            if "radius" in npz
            else 0.042
        )

    if h_mics.ndim != 3:
        raise ValueError(
            "Expected h_mics with shape "
            "(taps, microphones, directions), but received "
            f"{h_mics.shape}."
        )

    if measurement_dirs.ndim != 2 or measurement_dirs.shape[1] != 2:
        raise ValueError(
            "Expected measurement_dirs_aziElev with shape "
            "(directions, 2), but received "
            f"{measurement_dirs.shape}."
        )

    number_of_taps, number_of_microphones, number_of_directions = (
        h_mics.shape
    )

    if measurement_dirs.shape[0] != number_of_directions:
        raise ValueError(
            "The number of measurement directions does not match the "
            "direction axis of h_mics.\n"
            f"h_mics directions: {number_of_directions}\n"
            f"measurement directions: {measurement_dirs.shape[0]}"
        )

    if eigen_dirs is not None:
        if eigen_dirs.ndim != 2 or eigen_dirs.shape[1] != 2:
            raise ValueError(
                "Expected eigen_dirs_aziElev with shape "
                "(microphones, 2), but received "
                f"{eigen_dirs.shape}."
            )

        if eigen_dirs.shape[0] != number_of_microphones:
            raise ValueError(
                "The number of Eigenmike directions does not match the "
                "microphone axis of h_mics.\n"
                f"h_mics microphones: {number_of_microphones}\n"
                f"Eigenmike directions: {eigen_dirs.shape[0]}"
            )

    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError(
            f"Invalid sampling rate stored in the NPZ file: {sampling_rate}"
        )

    # Same axis conversion as in _em32_npz_reader:
    # taps x microphones x directions -> directions x microphones x taps.
    impulse_responses = np.transpose(h_mics, (2, 1, 0))

    # The provided Eigenmike reader treats these angles as radians.
    measurement_azimuth_deg = np.degrees(measurement_dirs[:, 0])
    measurement_elevation_deg = np.degrees(measurement_dirs[:, 1])

    # Convert azimuths to [-180°, 180°).
    measurement_azimuth_deg = (
        measurement_azimuth_deg + 180.0
    ) % 360.0 - 180.0

    return {
        "impulse_responses": impulse_responses,
        "h_mics": h_mics,
        "fs": sampling_rate,
        "measurement_dirs_aziElev": measurement_dirs,
        "measurement_azimuth_deg": measurement_azimuth_deg,
        "measurement_elevation_deg": measurement_elevation_deg,
        "eigen_dirs_aziElev": eigen_dirs,
        "radius": radius,
        "number_of_taps": number_of_taps,
        "number_of_microphones": number_of_microphones,
        "number_of_directions": number_of_directions,
    }


def apply_direct_sound_window(
    impulse_responses,
    sampling_rate,
    pre_peak_ms,
    post_peak_ms,
):
    """Window each impulse response around its largest absolute peak."""
    impulse_responses = np.asarray(impulse_responses, dtype=float)

    if impulse_responses.ndim != 2:
        raise ValueError(
            "Expected impulse_responses with shape "
            "(directions, taps), but received "
            f"{impulse_responses.shape}."
        )

    if pre_peak_ms < 0 or post_peak_ms < 0:
        raise ValueError("Window durations must not be negative.")

    windowed = np.zeros_like(impulse_responses)
    pre_samples = int(round(pre_peak_ms * 1e-3 * sampling_rate))
    post_samples = int(round(post_peak_ms * 1e-3 * sampling_rate))

    for direction_index, impulse_response in enumerate(impulse_responses):
        peak_index = int(np.argmax(np.abs(impulse_response)))
        start = max(0, peak_index - pre_samples)
        stop = min(
            impulse_response.size,
            peak_index + post_samples + 1,
        )

        segment_length = stop - start
        window = (
            np.hanning(segment_length)
            if segment_length > 1
            else np.ones(segment_length)
        )

        windowed[direction_index, start:stop] = (
            impulse_response[start:stop] * window
        )

    return windowed


def calculate_octave_band_levels(
    impulse_responses,
    sampling_rate,
    octave_centres,
):
    """
    Calculate octave-band energy levels for all selected directions.

    No normalization is applied in this calculation function. Each octave
    band is normalized to its own maximum later, immediately before plotting.
    """
    impulse_responses = np.asarray(impulse_responses, dtype=float)

    if impulse_responses.ndim != 2:
        raise ValueError(
            "Expected impulse_responses with shape "
            "(directions, taps), but received "
            f"{impulse_responses.shape}."
        )

    number_of_samples = impulse_responses.shape[-1]
    nyquist_frequency = sampling_rate / 2.0

    spectrum = np.fft.rfft(impulse_responses, axis=-1)
    frequencies = np.fft.rfftfreq(
        number_of_samples,
        d=1.0 / sampling_rate,
    )
    power_spectrum = np.abs(spectrum) ** 2

    levels = {}
    skipped = {}

    for centre_frequency in np.asarray(octave_centres, dtype=float):
        lower_frequency = centre_frequency / np.sqrt(2.0)
        upper_frequency = centre_frequency * np.sqrt(2.0)

        if upper_frequency > nyquist_frequency:
            skipped[centre_frequency] = (
                f"upper limit {upper_frequency:.1f} Hz exceeds "
                f"Nyquist frequency {nyquist_frequency:.1f} Hz"
            )
            continue

        band_mask = (
            (frequencies >= lower_frequency)
            & (frequencies < upper_frequency)
        )

        if not np.any(band_mask):
            skipped[centre_frequency] = (
                "no FFT bins inside the octave band"
            )
            continue

        band_power = np.sum(
            power_spectrum[:, band_mask],
            axis=1,
        )
        band_power = np.maximum(
            band_power,
            np.finfo(float).tiny,
        )
        levels[centre_frequency] = 10.0 * np.log10(band_power)

    return levels, skipped


def plot_all_octave_band_directivities(
    npz_path,
    microphone_index=0,
    target_elevation=0.0,
    elevation_tolerance=1.0,
    octave_centres=None,
    radial_range_db=30.0,
    direct_sound_window_ms=None,
    output_path="octave_band_directivity_em32.pdf",
):
    """
    Plot Eigenmike octave-band directivity curves on one polar axis.

    The NPZ filters are interpreted as:
        h_mics[taps, microphones, directions]

    Every octave band is normalized independently. The maximum level of
    each plotted band is set to 0 dB, so the plot compares directional shape
    rather than absolute level differences between octave bands.

    Parameters
    ----------
    npz_path : str or pathlib.Path
        Path to Eigenmike_em32_IRs.npz.

    microphone_index : int
        Zero-based Eigenmike capsule index. Valid values are normally 0--31.

    target_elevation : float
        Requested measurement elevation in degrees.

    elevation_tolerance : float
        Include directions within this many degrees of the nearest measured
        elevation.

    octave_centres : sequence or None
        Octave-band centre frequencies in hertz. When None, bands from
        63 Hz through 16 kHz are requested.

    radial_range_db : float
        Display range below each octave band's normalized 0 dB maximum.

    direct_sound_window_ms : tuple or None
        Optional ``(pre_peak_ms, post_peak_ms)`` window. Use None to analyse
        the complete filters.

    output_path : str or pathlib.Path or None
        Figure output path. Set to None to disable saving.
    """
    if octave_centres is None:
        octave_centres = np.array(
            [63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000],
            dtype=float,
        )

    if radial_range_db <= 0:
        raise ValueError("radial_range_db must be greater than zero.")

    em32 = load_em32_npz(npz_path)
    impulse_responses = em32["impulse_responses"]
    sampling_rate = em32["fs"]
    azimuth = em32["measurement_azimuth_deg"]
    elevation = em32["measurement_elevation_deg"]

    number_of_directions, number_of_microphones, number_of_samples = (
        impulse_responses.shape
    )

    if not 0 <= microphone_index < number_of_microphones:
        raise IndexError(
            f"Microphone {microphone_index} does not exist. Valid indices "
            f"are 0 to {number_of_microphones - 1}."
        )

    nearest_elevation = elevation[
        np.argmin(np.abs(elevation - target_elevation))
    ]
    elevation_mask = (
        np.abs(elevation - nearest_elevation)
        <= elevation_tolerance
    )
    selected_indices = np.flatnonzero(elevation_mask)

    if selected_indices.size < 3:
        raise ValueError(
            "Too few directions were found for the requested elevation "
            "slice.\n"
            f"Requested elevation: {target_elevation:.2f}°\n"
            f"Nearest measured elevation: {nearest_elevation:.2f}°\n"
            f"Tolerance: {elevation_tolerance:.2f}°\n"
            f"Selected directions: {selected_indices.size}"
        )

    selected_azimuth = azimuth[selected_indices]
    selected_ir = impulse_responses[
        selected_indices,
        microphone_index,
        :,
    ]

    if direct_sound_window_ms is not None:
        if len(direct_sound_window_ms) != 2:
            raise ValueError(
                "direct_sound_window_ms must be None or a tuple "
                "(pre_peak_ms, post_peak_ms)."
            )

        selected_ir = apply_direct_sound_window(
            selected_ir,
            sampling_rate,
            direct_sound_window_ms[0],
            direct_sound_window_ms[1],
        )

    levels_by_band, skipped_bands = calculate_octave_band_levels(
        selected_ir,
        sampling_rate,
        octave_centres,
    )
    valid_centres = list(levels_by_band)

    if not valid_centres:
        raise ValueError(
            "None of the requested octave bands could be calculated."
        )

    azimuth_order = np.argsort(selected_azimuth)
    sorted_azimuth = selected_azimuth[azimuth_order]
    closed_azimuth = np.concatenate(
        [sorted_azimuth, sorted_azimuth[:1] + 360.0]
    )
    theta = np.deg2rad(closed_azimuth)

    # Normalize every octave band independently to its own maximum.
    normalized_levels_by_band = {
        centre_frequency: (
            levels_by_band[centre_frequency]
            - np.max(levels_by_band[centre_frequency])
        )
        for centre_frequency in valid_centres
    }

    figure, axis = plt.subplots(
        figsize=(6.0, 6.0),
        subplot_kw={"projection": "polar"},
        constrained_layout=True,
    )

    # Plot non-negative radii for robust polar rendering, but label the
    # rings as dB relative to each band's own maximum.
    radial_tick_values = np.arange(0.0, radial_range_db + 0.1, 5.0)
    radial_tick_labels = radial_tick_values - radial_range_db

    for centre_frequency in valid_centres:
        normalized_levels_db = normalized_levels_by_band[
            centre_frequency
        ][azimuth_order]

        closed_normalized_levels = np.concatenate(
            [normalized_levels_db, normalized_levels_db[:1]]
        )
        plot_radius = closed_normalized_levels + radial_range_db

        axis.plot(
            theta,
            plot_radius,
            marker="o",
            markersize=3,
            linewidth=1.4,
            label=f"{centre_frequency:g} Hz",
        )

    # Measurement convention from measurement_dirs_aziElev:
    # 0° azimuth is at the front and +90° is counter-clockwise/left.
    axis.set_theta_zero_location("N")
    axis.set_theta_direction(1)

    # Display polar azimuth labels as −180° to +180°.
    axis.set_thetagrids(
        angles=[0, 45, 90, 135, 180, 225, 270, 315],
        labels=["0°", "45°", "90°", "135°",
                "±180°", "-135°", "-90°", "-45°"],
        fontsize=13,
    )

    axis.set_ylim(0.0, radial_range_db)
    axis.set_ylim(0.0, radial_range_db)
    axis.set_yticks(radial_tick_values)
    axis.set_yticklabels(
        [
            f"{tick:.0f} dB" if np.isclose(tick, -radial_range_db)
            else f"{tick:.0f}"
            for tick in radial_tick_labels
        ],
        fontsize=12,
    )
    axis.set_rlabel_position(225)
    axis.grid(True)

    axis.legend(
        title="Octave band",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.02),
        fontsize=11,
        title_fontsize=12,
    )

    actual_elevation = float(np.median(elevation[selected_indices]))
    microphone_label = f"EM32_{microphone_index + 1:02d}"

    window_description = (
        "complete filters"
        if direct_sound_window_ms is None
        else (
            f"window {direct_sound_window_ms[0]:g} ms before peak, "
            f"{direct_sound_window_ms[1]:g} ms after peak"
        )
    )

    axis.set_title(
        (
            f"{microphone_label}: octave-band directivity\n"
            f"Elevation ≈ {actual_elevation:.1f}°, "
            f"{window_description}\n"
            "Each octave band normalized to its own maximum = 0 dB"
        ),
        pad=20,
        fontsize=14,
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            output_path,
            format="pdf",
            bbox_inches="tight",
        )

    plt.show()

    print("\n================ ANALYSIS INFORMATION ================\n")
    print("NPZ file:", npz_path)
    print("Original h_mics shape:", em32["h_mics"].shape)
    print(
        "Plotting IR shape (directions, microphones, taps):",
        impulse_responses.shape,
    )
    print("Sampling rate:", sampling_rate, "Hz")
    print("FFT resolution:", sampling_rate / number_of_samples, "Hz")
    print("Microphone index:", microphone_index)
    print("Microphone label:", microphone_label)
    print("Requested elevation:", target_elevation, "degrees")
    print("Selected elevation:", actual_elevation, "degrees")
    print("Selected directions:", selected_indices.size)
    print("Normalization: each octave band maximum = 0 dB")
    print("Array radius:", em32["radius"], "m")

    print("\nPlotted octave bands:")
    print(
        ", ".join(
            f"{frequency:g} Hz"
            for frequency in valid_centres
        )
    )

    if skipped_bands:
        print("\nSkipped octave bands:")
        for frequency, reason in skipped_bands.items():
            print(f"  {frequency:g} Hz: {reason}")

    return figure, axis, normalized_levels_by_band


if __name__ == "__main__":
    npz_path = "Eigenmike_em32_IRs.npz"

    figure, axis, levels = plot_all_octave_band_directivities(
        npz_path=npz_path,
        microphone_index=1,
        target_elevation=0.0,
        elevation_tolerance=1.0,
        radial_range_db=30.0,

        # Analyse the complete filters.
        direct_sound_window_ms=None,

        # Optional peak-relative window:
        # direct_sound_window_ms=(2.0, 50.0),

        output_path="octave_band_directivity_em32.pdf",
    )
