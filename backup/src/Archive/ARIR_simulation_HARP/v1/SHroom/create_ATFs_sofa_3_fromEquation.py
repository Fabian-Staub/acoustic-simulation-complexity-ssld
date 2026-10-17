import numpy as np
from scipy.special import spherical_jn, spherical_yn, eval_legendre
from scipy.fft import irfft
import sofar as sf

# ------------------------------------------------------------
# PARAMETERS
# ------------------------------------------------------------
FS = 48000
NFFT = 1024          # Increased from 256 to reduce circular wrap-around
Rad = 0.042
SH_ORDER = 40
C = 343.0

NUM_SOURCES = 2000

# Causal delay embedded into the FIR.
# This replaces the previous time-domain np.roll(ir, NFFT // 2).
DELAY_SAMPLES = NFFT // 4
DELAY_SECONDS = DELAY_SAMPLES / FS

mic_dirs_deg = np.array([
    [0, 21], [32, 0], [0, -21], [328, 0],
    [0, 58], [45, 35], [69, 0], [45, -35],
    [0, -58], [315, -35], [291, 0], [315, 35],
    [91, 69], [90, 32], [90, -31], [89, -69],
    [180, 21], [212, 0], [180, -21], [148, 0],
    [180, 58], [225, 35], [249, 0], [225, -35],
    [180, -58], [135, -35], [111, 0], [135, 35],
    [269, 69], [270, 32], [270, -32], [271, -69]
], dtype=np.float64)


# ------------------------------------------------------------
# BASIC SPHERICAL GRID HELPERS
# ------------------------------------------------------------
def sph_to_cart(az, el):
    """
    Convert spherical coordinates to Cartesian unit vectors.

    Parameters
    ----------
    az : ndarray
        Azimuth in radians.
    el : ndarray
        Elevation in radians.

    Returns
    -------
    ndarray
        Cartesian unit vectors with shape (..., 3).
    """
    x = np.cos(el) * np.cos(az)
    y = np.cos(el) * np.sin(az)
    z = np.sin(el)

    return np.stack([x, y, z], axis=-1)


def fibonacci_sphere(n):
    """
    Create approximately uniform source directions on the sphere.

    Returns
    -------
    az : ndarray
        Azimuth in radians, range [0, 2*pi).
    el : ndarray
        Elevation in radians, range [-pi/2, pi/2].
    """
    i = np.arange(n, dtype=np.float64)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))

    z = 1.0 - 2.0 * (i + 0.5) / n
    radius = np.sqrt(np.maximum(0.0, 1.0 - z**2))

    az0 = i * golden_angle
    x = radius * np.cos(az0)
    y = radius * np.sin(az0)

    az = np.arctan2(y, x) % (2.0 * np.pi)
    el = np.arcsin(z)

    return az, el


# ------------------------------------------------------------
# SPHERICAL HANKEL FUNCTIONS
# ------------------------------------------------------------
def spherical_hankel2(n, x):
    """
    Spherical Hankel function of the second kind:
        h_n^(2)(x) = j_n(x) - i y_n(x)

    Important:
    Do not call this at x = 0, because y_n(0) is singular.
    """
    return spherical_jn(n, x) - 1j * spherical_yn(n, x)


def spherical_hankel2_derivative(n, x):
    """
    Derivative of spherical Hankel function of the second kind:
        h_n^(2)'(x) = j_n'(x) - i y_n'(x)

    Important:
    Do not call this at x = 0, because y_n'(0) is singular.
    """
    return (
        spherical_jn(n, x, derivative=True)
        - 1j * spherical_yn(n, x, derivative=True)
    )


# ------------------------------------------------------------
# RIGID SPHERE PLANE-WAVE ATF
# ------------------------------------------------------------
def rigid_sphere_atf(
    source_az,
    source_el,
    mic_az,
    mic_el,
    radius,
    fs,
    nfft,
    order,
    c=343.0,
):
    """
    Calculate rigid-sphere plane-wave ATFs analytically.

    Output
    ------
    H : ndarray
        Complex ATFs with shape:
            (num_sources, num_mics, num_freqs)

        where num_freqs = nfft // 2 + 1.

    freqs : ndarray
        Frequency vector in Hz.
    """

    # Cartesian unit vectors
    source_vecs = sph_to_cart(source_az, source_el)
    mic_vecs = sph_to_cart(mic_az, mic_el)

    # cos(gamma), where gamma is the angle between source direction and mic direction
    cos_gamma = source_vecs @ mic_vecs.T
    cos_gamma = np.clip(cos_gamma, -1.0, 1.0)

    # One-sided frequency vector for real-valued impulse responses
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)

    # Wavenumber times radius
    k = 2.0 * np.pi * freqs / c
    x = k * radius

    num_sources = len(source_az)
    num_mics = len(mic_az)
    num_freqs = len(freqs)

    H = np.zeros((num_sources, num_mics, num_freqs), dtype=np.complex128)

    # Positive frequencies only.
    # x[0] is zero and must not be passed to spherical_yn().
    xp = x[1:]

    for n in range(order + 1):
        print(f"Calculating order n = {n}")

        # Legendre polynomial P_n(cos gamma)
        Pn = eval_legendre(n, cos_gamma)

        # Rigid-sphere modal coefficient b_n(kr)
        bn = np.zeros_like(x, dtype=np.complex128)

        # DC / 0 Hz limit:
        # Only the omnidirectional n=0 mode remains.
        if n == 0:
            bn[0] = 1.0
        else:
            bn[0] = 0.0

        # Compute only positive frequency bins.
        jn = spherical_jn(n, xp)
        jn_der = spherical_jn(n, xp, derivative=True)

        hn = spherical_hankel2(n, xp)
        hn_der = spherical_hankel2_derivative(n, xp)

        # Avoid numerical division problems.
        tiny = 1e-300
        hn_der_safe = np.where(np.abs(hn_der) < tiny, tiny + 0j, hn_der)

        bn[1:] = jn - (jn_der / hn_der_safe) * hn

        # Plane-wave expansion coefficient
        coeff = (2 * n + 1) * (1j ** n) * bn

        # Add modal contribution
        H += Pn[:, :, None] * coeff[None, None, :]

    return H, freqs


# ------------------------------------------------------------
# MIC AND SOURCE POSITIONS
# ------------------------------------------------------------
mic_az = np.deg2rad(mic_dirs_deg[:, 0])
mic_el = np.deg2rad(mic_dirs_deg[:, 1])

source_az, source_el = fibonacci_sphere(NUM_SOURCES)


# ------------------------------------------------------------
# CALCULATE ATFS
# ------------------------------------------------------------
H, freqs = rigid_sphere_atf(
    source_az=source_az,
    source_el=source_el,
    mic_az=mic_az,
    mic_el=mic_el,
    radius=Rad,
    fs=FS,
    nfft=NFFT,
    order=SH_ORDER,
    c=C,
)

print("ATF shape:", H.shape)
print("Frequency bins:", len(freqs))
print("Frequency range:", freqs[0], "Hz to", freqs[-1], "Hz")

print("NaNs in H:", np.isnan(H).any())
print("Infs in H:", np.isinf(H).any())

if np.isnan(H).any() or np.isinf(H).any():
    raise RuntimeError("Invalid values found in frequency-domain ATFs.")


# ------------------------------------------------------------
# ADD CAUSAL DELAY IN FREQUENCY DOMAIN
# ------------------------------------------------------------
# This creates a causal FIR directly after irfft.
#
# This replaces the old approach:
#     ir = irfft(H, n=NFFT, axis=-1)
#     ir = np.roll(ir, NFFT // 2, axis=-1)
#
# No time-domain circular roll is used.
# No time-domain window is used.
phase_delay = np.exp(-1j * 2.0 * np.pi * freqs * DELAY_SECONDS)

H_delayed = H * phase_delay[None, None, :]

print("Embedded FIR delay:", DELAY_SAMPLES, "samples")
print("Embedded FIR delay:", DELAY_SECONDS * 1000.0, "ms")


# ------------------------------------------------------------
# CONVERT TO TIME DOMAIN
# ------------------------------------------------------------
ir = irfft(H_delayed, n=NFFT, axis=-1)

print("IR shape:", ir.shape)
print("NaNs in IR:", np.isnan(ir).any())
print("Infs in IR:", np.isinf(ir).any())

if np.isnan(ir).any() or np.isinf(ir).any():
    raise RuntimeError("Invalid values found in time-domain IRs.")


# ------------------------------------------------------------
# OPTIONAL DIAGNOSTIC: FIND PEAK SAMPLE POSITIONS
# ------------------------------------------------------------
# This helps verify that the main energy is near DELAY_SAMPLES.
peak_samples = np.argmax(np.abs(ir), axis=-1)

print("Peak sample min:", peak_samples.min())
print("Peak sample max:", peak_samples.max())
print("Peak sample mean:", peak_samples.mean())
print("Expected approximate delay sample:", DELAY_SAMPLES)


# ------------------------------------------------------------
# SOFA FORMAT
# ------------------------------------------------------------
# SOFA GeneralFIR expects:
# Data.IR shape = (M, R, N)
#
# M = measurements / source positions
# R = receivers / microphones
# N = samples
ir_sofa = np.ascontiguousarray(ir, dtype=np.float64)

M, R, N = ir_sofa.shape
print("SOFA IR shape:", ir_sofa.shape)


# ------------------------------------------------------------
# CREATE SOFA OBJECT
# ------------------------------------------------------------
sofa = sf.Sofa("GeneralFIR")

sofa.Data_IR = ir_sofa
sofa.Data_SamplingRate = np.array([FS], dtype=np.float64)

# IMPORTANT:
# The delay is already embedded into Data.IR using frequency-domain phase delay.
# Therefore Data.Delay should stay zero to avoid double-delay in playback/rendering.
sofa.Data_Delay = np.zeros((M, R), dtype=np.float64)


# ------------------------------------------------------------
# SOURCE POSITIONS
# ------------------------------------------------------------
# SOFA spherical convention:
# azimuth in degrees, elevation in degrees, radius in meter
sofa.SourcePosition = np.column_stack([
    np.rad2deg(source_az),
    np.rad2deg(source_el),
    np.ones(M),
])

sofa.SourcePosition_Type = "spherical"
sofa.SourcePosition_Units = "degree, degree, meter"


# ------------------------------------------------------------
# MICROPHONE POSITIONS
# ------------------------------------------------------------
sofa.ReceiverPosition = np.column_stack([
    np.rad2deg(mic_az),
    np.rad2deg(mic_el),
    np.full(R, Rad),
])

sofa.ReceiverPosition_Type = "spherical"
sofa.ReceiverPosition_Units = "degree, degree, meter"


# ------------------------------------------------------------
# METADATA
# ------------------------------------------------------------
sofa.GLOBAL_Title = "Eigenmike 32ch analytically simulated rigid-sphere ATFs"
sofa.GLOBAL_Comment = (
    "Rigid sphere plane-wave ATFs calculated analytically without SHROOM. "
    "Frequency-domain ATFs converted to FIR using irfft. "
    "DC bin handled separately to avoid spherical Hankel singularity. "
    "A causal delay is embedded in the FIR using frequency-domain linear phase. "
    "No time-domain circular roll and no time-domain windowing are used. "
    f"NFFT={NFFT}, embedded delay={DELAY_SAMPLES} samples."
)


# ------------------------------------------------------------
# WRITE FILE
# ------------------------------------------------------------
output_filename = "eigenmike32_analytic_rigid_sphere_causal_unwindowed.sofa"
sf.write_sofa(output_filename, sofa)

print(f"✔ SOFA file successfully written: {output_filename}")