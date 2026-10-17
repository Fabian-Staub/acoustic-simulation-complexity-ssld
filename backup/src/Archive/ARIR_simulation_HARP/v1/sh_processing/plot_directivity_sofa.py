import sofar as sf
import numpy as np
import matplotlib.pyplot as plt


def source_positions_to_spherical(sofa):
    """
    Return source positions as:
        azimuth [degrees], elevation [degrees], radius [metres]

    Supports spherical and Cartesian SOFA positions.
    """
    positions = np.squeeze(
        np.asarray(sofa.SourcePosition, dtype=float)
    )

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
        getattr(
            sofa,
            "SourcePosition_Units",
            "degree, degree, metre"
        )
    ).lower()

    if "cartesian" in position_type:
        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]

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

    # Convert azimuths to [-180°, 180°).
    azimuth = (azimuth + 180.0) % 360.0 - 180.0

    return azimuth, elevation, radius


def apply_direct_sound_window(
    impulse_responses,
    sampling_rate,
    pre_peak_ms,
    post_peak_ms
):
    """
    Window every impulse response around its largest absolute peak.
    """
    impulse_responses = np.asarray(
        impulse_responses,
        dtype=float
    )

    windowed = np.zeros_like(impulse_responses)

    pre_samples = int(
        round(pre_peak_ms * 1e-3 * sampling_rate)
    )

    post_samples = int(
        round(post_peak_ms * 1e-3 * sampling_rate)
    )

    for measurement_index, impulse_response in enumerate(
        impulse_responses
    ):
        peak_index = int(
            np.argmax(np.abs(impulse_response))
        )

        start = max(0, peak_index - pre_samples)

        stop = min(
            impulse_response.size,
            peak_index + post_samples + 1
        )

        segment_length = stop - start

        if segment_length > 1:
            window = np.hanning(segment_length)
        else:
            window = np.ones(segment_length)

        windowed[measurement_index, start:stop] = (
            impulse_response[start:stop] * window
        )

    return windowed


def calculate_octave_band_levels(
    impulse_responses,
    sampling_rate,
    octave_centres
):
    """
    Calculate octave-band levels for multiple centre frequencies.

    Returns
    -------
    levels : dict
        Dictionary mapping centre frequency to level array.

    skipped : dict
        Dictionary describing octave bands that could not be calculated.
    """
    impulse_responses = np.asarray(
        impulse_responses,
        dtype=float
    )

    number_of_samples = impulse_responses.shape[-1]
    nyquist_frequency = sampling_rate / 2.0

    spectrum = np.fft.rfft(
        impulse_responses,
        axis=-1
    )

    frequencies = np.fft.rfftfreq(
        number_of_samples,
        d=1.0 / sampling_rate
    )

    power_spectrum = np.abs(spectrum) ** 2

    levels = {}
    skipped = {}

    for centre_frequency in octave_centres:
        lower_frequency = centre_frequency / np.sqrt(2.0)
        upper_frequency = centre_frequency * np.sqrt(2.0)

        # Require the complete octave band to remain below Nyquist.
        if upper_frequency > nyquist_frequency:
            skipped[centre_frequency] = (
                f"upper limit {upper_frequency:.1f} Hz "
                f"exceeds Nyquist frequency "
                f"{nyquist_frequency:.1f} Hz"
            )
            continue

        band_mask = (
            (frequencies >= lower_frequency)
            & (frequencies < upper_frequency)
        )

        number_of_bins = np.count_nonzero(band_mask)

        if number_of_bins == 0:
            skipped[centre_frequency] = (
                "no FFT bins inside the octave band"
            )
            continue

        band_power = np.sum(
            power_spectrum[:, band_mask],
            axis=1
        )

        band_power = np.maximum(
            band_power,
            np.finfo(float).tiny
        )

        levels[centre_frequency] = (
            10.0 * np.log10(band_power)
        )

    return levels, skipped


def plot_all_octave_band_directivities(
    sofa_path,
    receiver_index=0,
    target_elevation=0.0,
    elevation_tolerance=1.0,
    octave_centres=None,
    radial_range_db=30.0,
    direct_sound_window_ms=None
):
    """
    Plot all octave-band directivity patterns for one SOFA receiver.

    All octave bands are overlaid in one polar plot. Each octave band is
    normalized independently so that its maximum across directions is 0 dB.

    Parameters
    ----------
    sofa_path : str
        Path to the SOFA file.

    receiver_index : int
        Receiver index in Data_IR[:, receiver_index, :].

    target_elevation : float
        Requested source elevation in degrees.

    elevation_tolerance : float
        Include source positions within this tolerance of the nearest
        available elevation.

    octave_centres : sequence or None
        Octave-band centre frequencies. When None, standard bands from
        63 Hz through 16 kHz are used.

    radial_range_db : float
        Display range below each octave band's normalized 0 dB maximum.

    direct_sound_window_ms : tuple or None
        Optional peak-relative window:
            (pre_peak_ms, post_peak_ms)

        Use None to analyse the complete impulse response.
    """
    if octave_centres is None:
        octave_centres = np.array(
            [
                63,
                125,
                250,
                500,
                1000,
                2000,
                4000,
                8000,
                16000
            ],
            dtype=float
        )

    sofa = sf.read_sofa(
        sofa_path,
        verify=False
    )

    impulse_responses = np.asarray(
        sofa.Data_IR,
        dtype=float
    )

    if impulse_responses.ndim != 3:
        raise ValueError(
            "Expected Data_IR with shape (M, R, N), "
            f"but received {impulse_responses.shape}."
        )

    number_of_measurements, number_of_receivers, number_of_samples = (
        impulse_responses.shape
    )

    if not 0 <= receiver_index < number_of_receivers:
        raise IndexError(
            f"Receiver {receiver_index} does not exist. "
            f"Valid indices are 0 to {number_of_receivers - 1}."
        )

    sampling_rate = float(
        np.asarray(
            sofa.Data_SamplingRate
        ).squeeze()
    )

    azimuth, elevation, radius = (
        source_positions_to_spherical(sofa)
    )

    if len(azimuth) != number_of_measurements:
        raise ValueError(
            "The number of SourcePosition entries does not match "
            "the number of Data_IR measurements."
        )

    # Find the measured elevation nearest to the requested elevation.
    nearest_elevation = elevation[
        np.argmin(
            np.abs(elevation - target_elevation)
        )
    ]

    elevation_mask = (
        np.abs(elevation - nearest_elevation)
        <= elevation_tolerance
    )

    selected_indices = np.flatnonzero(
        elevation_mask
    )

    if selected_indices.size < 3:
        raise ValueError(
            "Too few source positions were found for the requested "
            "elevation slice.\n"
            f"Requested elevation: {target_elevation:.2f}°\n"
            f"Nearest available elevation: {nearest_elevation:.2f}°\n"
            f"Selected positions: {selected_indices.size}"
        )

    selected_azimuth = azimuth[selected_indices]

    selected_ir = impulse_responses[
        selected_indices,
        receiver_index,
        :
    ]

    if direct_sound_window_ms is not None:
        pre_peak_ms, post_peak_ms = direct_sound_window_ms

        selected_ir = apply_direct_sound_window(
            selected_ir,
            sampling_rate,
            pre_peak_ms,
            post_peak_ms
        )

    levels_by_band, skipped_bands = (
        calculate_octave_band_levels(
            selected_ir,
            sampling_rate,
            octave_centres
        )
    )

    valid_centres = list(
        levels_by_band.keys()
    )

    if not valid_centres:
        raise ValueError(
            "None of the requested octave bands could be calculated."
        )

    # Sort directions by azimuth once and use the same order for every band.
    azimuth_order = np.argsort(
        selected_azimuth
    )

    sorted_azimuth = selected_azimuth[
        azimuth_order
    ]

    # Repeat the first direction to close each polar curve.
    closed_azimuth = np.concatenate(
        [
            sorted_azimuth,
            sorted_azimuth[:1] + 360.0
        ]
    )

    theta = np.deg2rad(
        closed_azimuth
    )

    # Normalize every octave band independently to its own maximum.
    normalized_levels_by_band = {
        centre_frequency: (
            levels_by_band[centre_frequency]
            - np.max(levels_by_band[centre_frequency])
        )
        for centre_frequency in valid_centres
    }

    figure, axis = plt.subplots(
        figsize=(9.0, 8.0),
        subplot_kw={"projection": "polar"},
        constrained_layout=True
    )

    radial_ticks = np.arange(
        -radial_range_db,
        0.1,
        5.0
    )

    for centre_frequency in valid_centres:
        normalized_levels_db = normalized_levels_by_band[
            centre_frequency
        ][azimuth_order]

        closed_levels = np.concatenate(
            [
                normalized_levels_db,
                normalized_levels_db[:1]
            ]
        )

        axis.plot(
            theta,
            closed_levels,
            marker="o",
            markersize=3,
            linewidth=1.4,
            label=f"{centre_frequency:g} Hz"
        )

    # SOFA spherical convention:
    # 0° = front and +90° = left.
    axis.set_theta_zero_location("N")
    axis.set_theta_direction(1)

    axis.set_ylim(
        -radial_range_db,
        0.0
    )

    axis.set_yticks(
        radial_ticks
    )

    axis.set_yticklabels(
        [
            f"{tick:.0f}"
            for tick in radial_ticks
        ],
        fontsize=8
    )

    axis.grid(True)

    axis.legend(
        title="Octave band",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.02)
    )

    actual_elevation = float(
        np.median(
            elevation[selected_indices]
        )
    )

    window_description = (
        "complete impulse response"
        if direct_sound_window_ms is None
        else (
            f"window {direct_sound_window_ms[0]:g} ms before peak, "
            f"{direct_sound_window_ms[1]:g} ms after peak"
        )
    )

    axis.set_title(
        (
            f"Receiver {receiver_index}: octave-band directivity\n"
            f"Elevation ≈ {actual_elevation:.1f}°, "
            f"{window_description}\n"
            "Shared reference: global maximum = 0 dB"
        ),
        pad=20
    )

    figure.savefig(
        "octave_band_directivity_normalized.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print("\n================ ANALYSIS INFORMATION ================\n")
    print("SOFA file:", sofa_path)
    print("Data_IR shape:", impulse_responses.shape)
    print("Sampling rate:", sampling_rate, "Hz")
    print("FFT resolution:", sampling_rate / number_of_samples, "Hz")
    print("Receiver index:", receiver_index)
    print("Requested elevation:", target_elevation, "degrees")
    print("Selected elevation:", actual_elevation, "degrees")
    print("Selected directions:", selected_indices.size)
    print("Normalization: each octave band maximum = 0 dB")

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
            print(
                f"  {frequency:g} Hz: {reason}"
            )

    return figure, axis, normalized_levels_by_band


# ============================================================
# Example
# ============================================================

sofa_path = (
    "/scratch/elec/t412-techpsychoacoustics/msc-thesis/"
    "2026_fstaub/staubf1/Projects/Master_Thesis/src/"
    "HARP-main/v1/em32_test001_20260702.sofa"
)

figure, axis, levels = plot_all_octave_band_directivities(
    sofa_path=sofa_path,
    receiver_index=1,
    target_elevation=0.0,
    elevation_tolerance=1.0,
    radial_range_db=30.0,

    # Analyse the complete impulse responses.
    direct_sound_window_ms=None

    # Optional direct-sound window:
    # direct_sound_window_ms=(2.0, 50.0)
)