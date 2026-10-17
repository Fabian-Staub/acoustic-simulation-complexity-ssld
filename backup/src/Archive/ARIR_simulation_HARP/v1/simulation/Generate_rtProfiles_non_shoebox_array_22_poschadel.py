import numpy as np
import pandas as pd
import pyroomacoustics as pra
from pyroomacoustics.directivities import MeasuredDirectivityFile, Rotation3D
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from SphericalHarmonic_3 import HOA_array, SphericalHarmonicDirectivity
import os
import logging
import sys
from typing import Union
import argparse
import resource
import time
import json
import pickle
from functools import lru_cache, partial
import pyfar as pf
from scipy.special import spherical_jn, spherical_yn, sph_harm
from scipy.signal import resample_poly, fftconvolve, find_peaks
from math import gcd
from scipy.ndimage import gaussian_filter1d

# Configure logging
# Create logs directory if it doesn't exist
logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)
#log_file_path = os.path.join(logs_dir, "rir_generation.log")
log_file_path = os.path.join(logs_dir, f"rir_generation_{os.environ.get('SLURM_ARRAY_TASK_ID','0')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path) # Log to file
        #logging.StreamHandler(sys.stdout) # Log to console
    ]
)

# --- Global Constants ---
# Lookup table for materials based on pyroomacoustics examples/defaults
LOOKUP_TABLE = [
    "hard_surface",
    "brickwork",
    "rough_concrete",
    "unpainted_concrete",
    "rough_lime_wash",
    "smooth_brickwork_flush_pointing",
    "smooth_brickwork_10mm_pointing",
    "brick_wall_rough",
    "ceramic_tiles",
    "limestone_wall",
    "reverb_chamber",
    "plasterboard",
    "wooden_lining",
    "glass_3mm",
    "glass_window",
    "double_glazing_30mm",
    "double_glazing_10mm",
    "wood_1.6cm",
    "curtains_cotton_0.5",
    "curtains_0.2",
    "curtains_velvet",
    "curtains_glass_mat",
    "carpet_cotton",
    "carpet_6mm_closed_cell_foam",
    "carpet_6mm_open_cell_foam",
    "carpet_tufted_9m",
    "felt_5mm",
    "carpet_hairy",
    "concrete_floor",
    "marble_floor",
    "orchestra_1.5_m2",
    "panel_fabric_covered_6pcf",
    "panel_fabric_covered_8pcf",
    "ceiling_fibre_abosrber",
]

# Categorized material choices for different surfaces
WALLS = np.concatenate((LOOKUP_TABLE[:22], LOOKUP_TABLE[30:]))
FLOOR = np.concatenate((LOOKUP_TABLE[7:8], LOOKUP_TABLE[22:31]))
CEIL = np.concatenate((LOOKUP_TABLE[3:5], LOOKUP_TABLE[30:]))

# --- Utility Functions ---

def get_random_dimensions(x_range=(2.5, 20.1, 0.5), y_range=(2.5, 20.1, 0.5), z_range=(2.5, 5.1, 0.5),) -> tuple:
    """
    Generate random room dimensions with independent ranges per axis.

    Args:
        x_range: (min, max, step) for x-dimension
        y_range: (min, max, step) for y-dimension
        z_range: (min, max, step) for z-dimension

    Returns:
        tuple: (Lx, Ly, Lz)
    """

    x_vals = np.arange(*x_range)
    y_vals = np.arange(*y_range)
    z_vals = np.arange(*z_range)

    Lx = np.random.choice(x_vals, size=1)
    Ly = np.random.choice(y_vals, size=1)
    Lz = np.random.choice(z_vals, size=1)

    return Lx, Ly, Lz
    #return (np.array([13.5]), np.array([19.5]), np.array([6.5]))

'''
def get_random_dimensions(
    x_range=(1.5, 20),
    y_range=(1.5, 20),
    z_range=(2.5, 7.5),
    step=0.5,
    ab=(1.2, 2.5)   # Beta shape (a, b)
):
    a, b = ab

    def sample_beta(low, high):
        u = np.random.beta(a, b)
        val = low + u * (high - low)
        val = np.round(val / step) * step
        return np.clip(val, low, high)

    def sample_exp(low, high, k=1.2):
        # inverse transform sampling for truncated exponential
        u = np.random.rand()

        # CDF inversion for shifted exponential:
        val = low - (1.0 / k) * np.log(1 - u * (1 - np.exp(-k * (high - low))))

        val = np.round(val / step) * step
        return np.clip(val, low, high)

    Lx = sample_beta(*x_range)
    Ly = sample_beta(*y_range)

    Lz = sample_exp(*z_range, k=0.8)  # adjust k for steepness

    return np.array([Lx]), np.array([Ly]), np.array([Lz])


def generate_frequency_absorption(centerFreqs,
                                  a=2.5,
                                  b_min=1.1,
                                  b_max=2.5,
                                  low=0.01,
                                  high=0.99,
                                  seed=None):
    """
    Generate absorption values across frequency bands where
    higher frequencies have higher Beta 'b' parameter -> lower absorption.

    Parameters
    ----------
    centerFreqs : array-like
        Frequency band centers (only length is used).
    a : float
        Fixed Beta 'a' parameter.
    b_min : float
        Beta parameter at lowest frequency.
    b_max : float
        Beta parameter at highest frequency.
    low, high : float
        Clipping bounds.
    seed : int or None
        Random seed.

    Returns
    -------
    np.ndarray
        Absorption values per frequency band.
    """
    rng = np.random.default_rng(seed)
    n = centerFreqs.shape[0]

    # normalized frequency axis: 0 -> low freq, 1 -> high freq
    t = np.linspace(0, 1, n)

    # interpolate b from b_min (low freq) to b_max (high freq)
    b = b_min + t * (b_max - b_min)

    absorption = rng.beta(a, b, size=n)

    return np.clip(absorption, low, high)
'''

def generate_frequency_absorption(centerFreqs,
                                  a_values,
                                  b_values,
                                  low=0.01,
                                  high=0.99,
                                  seed=None):
    """
    Generate absorption values using manually specified Beta
    distribution parameters for each frequency band.

    Parameters
    ----------
    centerFreqs : array-like
        Frequency band centers.
    a_values : array-like
        Beta 'a' parameter for each frequency band.
    b_values : array-like
        Beta 'b' parameter for each frequency band.
    low, high : float
        Clipping bounds.
    seed : int or None
        Random seed.

    Returns
    -------
    np.ndarray
        Absorption values per frequency band.
    """
    rng = np.random.default_rng(seed)

    centerFreqs = np.asarray(centerFreqs)
    a_values = np.asarray(a_values)
    b_values = np.asarray(b_values)

    n = len(centerFreqs)

    if len(a_values) != n:
        raise ValueError(
            f"a_values must have length {n}, got {len(a_values)}"
        )

    if len(b_values) != n:
        raise ValueError(
            f"b_values must have length {n}, got {len(b_values)}"
        )

    absorption = rng.beta(a_values, b_values)

    return np.clip(absorption, low, high)


def get_random_positions(Lx: float, Ly: float, Lz: float, min_dist_from_wall: float = 0.5, step: float = 0.05) -> tuple:
    """
    Generate random positions for the source and microphone in the room.

    Args:
        Lx: Room length along x-axis.
        Ly: Room length along y-axis.
        Lz: Room length along z-axis.
        min_dist_from_wall: Minimum distance of source/mic from any wall.
        step: Step size for position choices.

    Returns:
        tuple: Random source and receiver positions (Sx, Sy, Sz, Rx, Ry, Rz)
               as numpy arrays (size 1).
    """
    # Determine the valid range for positions
    x_pos_range = np.arange(min_dist_from_wall, Lx - min_dist_from_wall, step)
    y_pos_range = np.arange(min_dist_from_wall, Ly - min_dist_from_wall, step)
    z_pos_range = np.arange(1, Lz - 1, step)

    # Check if valid positions are possible
    if len(x_pos_range) == 0 or len(y_pos_range) == 0 or len(z_pos_range) == 0:
        raise ValueError(f"Room dimensions ({Lx}, {Ly}, {Lz}) too small for minimum distance {min_dist_from_wall}.")

    Sx, Sy, Sz = [np.random.choice(pos_range, size=1) for pos_range in [x_pos_range, y_pos_range, z_pos_range]]
    Rx, Ry, Rz = [np.random.choice(pos_range, size=1) for pos_range in [x_pos_range, y_pos_range, z_pos_range]]

    # Ensure source and receiver are not at the exact same position
    while np.allclose([Rx, Ry, Rz], [Sx, Sy, Sz]):
         Rx, Ry, Rz = [np.random.choice(pos_range, size=1) for pos_range in [x_pos_range, y_pos_range, z_pos_range]]

    return Sx, Sy, Sz, Rx, Ry, Rz
    #return (np.array([5.0]), np.array([5.0]), np.array([5.0]), np.array([3.0]), np.array([5.0]), np.array([5.0]))

def doa_unit_vector(src, mic):
    """Return the world-frame unit vector pointing from microphone to source."""
    src = np.asarray(src, dtype=float).reshape(3)
    mic = np.asarray(mic, dtype=float).reshape(3)
    v = src - mic
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("Source and microphone positions must be different.")
    return v / norm


def doa_in_mic_frame(src, mic, mic_orientation_az_rad):
    """Compute ACCDOA, azimuth, and elevation in the microphone-local frame.

    The microphone orientation is a yaw rotation about +z. Therefore, the
    world-frame DOA is rotated by the inverse yaw (-orientation) so that a
    source directly in front of the rotated microphone has azimuth 0 deg.
    """
    doa_world = doa_unit_vector(src, mic)

    c = np.cos(mic_orientation_az_rad)
    s = np.sin(mic_orientation_az_rad)
    world_to_mic = np.array([
        [ c,  s, 0.0],
        [-s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])
    accdoa = world_to_mic @ doa_world

    # Guard against small numerical drift before inverse trigonometric use.
    accdoa = accdoa / np.linalg.norm(accdoa)
    az_rad = np.arctan2(accdoa[1], accdoa[0])
    el_rad = np.arctan2(accdoa[2], np.hypot(accdoa[0], accdoa[1]))

    return accdoa, np.degrees(az_rad), np.degrees(el_rad)

def is_too_close(new_dir, dirs, min_angle_deg):
    if dirs is None or len(dirs) == 0:
        return False

    dirs = np.asarray(dirs)

    # ensure shape is (N, 3)
    if dirs.ndim == 1:
        dirs = dirs.reshape(1, -1)

    cos_thr = np.cos(np.deg2rad(min_angle_deg))
    dots = dirs @ new_dir

    return np.any(dots >= cos_thr)


def get_random_materials() -> dict:
    """
    Randomly choose materials for the room surfaces from predefined lists.

    Returns:
        dict: Material dictionary with random choices for walls, ceiling, and floor.
    """
    return {
        "east": np.random.choice(WALLS),
        "west": np.random.choice(WALLS),
        "north": np.random.choice(WALLS),
        "south": np.random.choice(WALLS),
        "ceiling": np.random.choice(CEIL),
        "floor": np.random.choice(FLOOR),
    }

def get_room_range(num_rooms: int):
    """
    Determine which rooms this SLURM array task should process.
    If not running in SLURM array mode, process all rooms.
    """

    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    task_count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

    rooms_per_task = num_rooms // task_count
    remainder = num_rooms % task_count

    start = task_id * rooms_per_task + min(task_id, remainder)
    end = start + rooms_per_task + (1 if task_id < remainder else 0)

    return start, end


def inverse_eyring(rt60, room_dims, c=343.0):
    """
    Inverse Eyring formula to estimate mean absorption coefficient.

    Parameters
    ----------
    rt60 : float or array-like
        Reverberation time(s) per octave band
    room_dims : array-like (Lx, Ly, Lz)
        Room dimensions in meters
    c : float
        Speed of sound (default 343 m/s)

    Returns
    -------
    alpha : np.ndarray
        Estimated absorption coefficient(s)
    """

    rt60 = np.asarray(rt60)
    Lx, Ly, Lz = room_dims

    V = Lx * Ly * Lz
    S = 2 * (Lx * Ly + Lx * Lz + Ly * Lz)

    # avoid division by zero / invalid values
    rt60 = np.maximum(rt60, 1e-6)

    alpha = 1.0 - np.exp(-(0.161 * V) / (S * rt60))

    return alpha


def create_ambisonic_array(order_of_ambisonics: int, sample_rate: int, azimuth_rotation: float = 0.0):
    """
    Creates an Ambisonic microphone array definition.

    Args:
        order_of_ambisonics: The desired order of the Ambisonic array.
        sample_rate: The sample rate for the microphone array definition.

    Returns:
        tuple: A tuple containing:
               - mic_positions.T (numpy array): Transposed microphone positions.
               - sample_rate (int): The sample rate.
               - microphone_directivities (list): List of SphericalHarmonicDirectivity objects.
    """
    mic_radius = 0.000001 # Effectively a point mic array at the center
    order = order_of_ambisonics
    # Number of microphones required for a given Ambisonic order
    samples = (order + 1) ** 2

    # HOA_array expects radius=1 for generating positions on a unit sphere
    mic_positions, orientations, degrees = HOA_array(
        samples=samples,
        radius=1,
        n_order=order,
        azimuth_rotation=azimuth_rotation,
    )

    # Scale positions by the actual mic radius
    mic_positions = mic_positions * mic_radius

    microphone_directivities = []
    for i in range(samples):
        orientation = orientations[i]
        # Spherical Harmonic Directivity requires n and m parameters
        directivity = SphericalHarmonicDirectivity(
            orientation,
            n=degrees[i][0],
            m=degrees[i][1],
            azimuth_rotation=azimuth_rotation,
        )
        microphone_directivities.append(directivity)

    return mic_positions.T, sample_rate, microphone_directivities


def _azi_elev_to_cartesian(azi_elev, radius):
    """Convert [azimuth, elevation] directions to 3-D Cartesian positions."""
    azi_elev = np.asarray(azi_elev, dtype=float)
    az = azi_elev[:, 0]
    el = azi_elev[:, 1]
    cos_el = np.cos(el)
    return radius * np.column_stack((
        cos_el * np.cos(az),
        cos_el * np.sin(az),
        np.sin(el),
    ))


def _rotate_positions_about_z(positions, yaw_rad):
    """Rotate an (N, 3) position matrix about +z."""
    c_yaw = np.cos(yaw_rad)
    s_yaw = np.sin(yaw_rad)
    rotation = np.array([
        [c_yaw, -s_yaw, 0.0],
        [s_yaw,  c_yaw, 0.0],
        [0.0,    0.0,   1.0],
    ])
    return np.asarray(positions) @ rotation.T


# Per-process caches. Each ProcessPool worker loads the measured archive once and
# reuses EM32 encoding filters for repeated FFT lengths.
_EM32_DATA_CACHE = {}
_EM32_SHT_FILTER_CACHE = {}


def _get_worker_array(
    array_type,
    ambi_order,
    sample_rate,
    array_azimuth_rad,
    em32_npz_path,
    em32_radius,
):
    """Build a rotated microphone array inside a worker process."""
    if array_type == "em32":
        array = create_em32_array(
            em32_npz_path=em32_npz_path,
            sample_rate=sample_rate,
            azimuth_rotation=array_azimuth_rad,
            radius=em32_radius,
        )
    else:
        array = create_ambisonic_array(
            order_of_ambisonics=ambi_order,
            sample_rate=sample_rate,
            azimuth_rotation=array_azimuth_rad,
        )

    return array


def _load_em32_npz(npz_path, requested_fs=None, radius=0.042):
    """Load, validate, resample, and cache the Eigenmike EM32 archive."""
    abs_path = os.path.abspath(os.fspath(npz_path))
    cache_key = (abs_path, None if requested_fs is None else int(requested_fs), float(radius))
    cached = _EM32_DATA_CACHE.get(cache_key)
    if cached is not None:
        return cached

    data = np.load(abs_path, allow_pickle=False)
    required = {
        "fs", "h_mics", "measurement_area_weights",
        "measurement_dirs_aziElev", "eigen_dirs_aziElev",
    }
    missing = required.difference(data.files)
    if missing:
        raise KeyError(f"Missing EM32 NPZ arrays: {sorted(missing)}")

    fs_file = int(np.asarray(data["fs"]).item())
    h_mics = np.asarray(data["h_mics"], dtype=float)
    measurement_dirs = np.asarray(data["measurement_dirs_aziElev"], dtype=float)
    eigen_dirs = np.asarray(data["eigen_dirs_aziElev"], dtype=float)
    weights = np.asarray(data["measurement_area_weights"], dtype=float).reshape(-1)

    # Stored archive convention supplied by the user: taps x 32 microphones x directions.
    if h_mics.ndim != 3:
        raise ValueError(f"h_mics must be 3-D, got {h_mics.shape}")
    if h_mics.shape[1] == 32 and h_mics.shape[2] == measurement_dirs.shape[0]:
        pass
    elif h_mics.shape[0] == measurement_dirs.shape[0] and h_mics.shape[1] == 32:
        h_mics = np.transpose(h_mics, (2, 1, 0))
    else:
        raise ValueError(
            "Could not identify h_mics axes. Expected (taps, 32, directions), "
            f"got {h_mics.shape} with {measurement_dirs.shape[0]} directions."
        )

    if eigen_dirs.shape != (32, 2):
        raise ValueError(f"eigen_dirs_aziElev must have shape (32, 2), got {eigen_dirs.shape}")
    if weights.size != measurement_dirs.shape[0]:
        raise ValueError("measurement_area_weights and measurement directions differ in length")

    fs_out = fs_file if requested_fs is None else int(requested_fs)
    if fs_out != fs_file:
        g = gcd(fs_out, fs_file)
        h_mics = resample_poly(h_mics, fs_out // g, fs_file // g, axis=0)

    # Normalize quadrature weights to integrate over 4*pi steradians.
    weight_sum = np.sum(weights)
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("Invalid measurement_area_weights")
    weights = weights * (4.0 * np.pi / weight_sum)

    result = {
        "fs": fs_out,
        "h_mics": np.ascontiguousarray(h_mics),
        "measurement_area_weights": np.ascontiguousarray(weights),
        "measurement_dirs_aziElev": np.ascontiguousarray(measurement_dirs),
        "eigen_dirs_aziElev": np.ascontiguousarray(eigen_dirs),
        "radius": float(radius),
    }
    _EM32_DATA_CACHE[cache_key] = result
    return result


def _em32_npz_reader(path, fs=None, radius=0.042):
    """pyroomacoustics reader callback for Eigenmike_em32_IRs.npz."""
    em32 = _load_em32_npz(path, requested_fs=fs, radius=radius)

    # MeasuredDirectivityFile expects: directions x microphones x taps.
    impulse_responses = np.transpose(em32["h_mics"], (2, 1, 0))

    directions = em32["measurement_dirs_aziElev"]
    source_locs = np.vstack((
        directions[:, 0],
        np.pi / 2.0 - directions[:, 1],
        np.ones(directions.shape[0]),
    ))
    mic_locs = _azi_elev_to_cartesian(
        em32["eigen_dirs_aziElev"], em32["radius"]
    ).T

    source_labels = [str(i) for i in range(source_locs.shape[1])]
    mic_labels = [f"EM32_{i + 1:02d}" for i in range(mic_locs.shape[1])]
    return impulse_responses, em32["fs"], source_locs, mic_locs, source_labels, mic_labels


def create_em32_array(
    em32_npz_path,
    sample_rate,
    azimuth_rotation=0.0,
    radius=0.042,
    interp_order=None,
):
    """Create a physical 32-channel Eigenmike array from measured directivities."""
    em32 = _load_em32_npz(em32_npz_path, requested_fs=sample_rate, radius=radius)
    positions = _azi_elev_to_cartesian(em32["eigen_dirs_aziElev"], radius)
    positions = _rotate_positions_about_z(positions, azimuth_rotation)

    reader = partial(_em32_npz_reader, radius=radius)
    measured_file = MeasuredDirectivityFile(
        em32_npz_path,
        fs=sample_rate,
        interp_order=interp_order,
        file_reader_callback=reader,
    )
    orientation = Rotation3D(
        [azimuth_rotation, 0.0, 0.0], rot_order="zyx", degrees=False
    )
    directivities = [
        measured_file.get_mic_directivity(i, orientation) for i in range(32)
    ]
    return positions.T, sample_rate, directivities


# Eigenmike capsule directions: [azimuth, elevation] in degrees.
# These are used only for the theoretical rigid-sphere SHT encoder. The measured
# NPZ remains responsible for the microphone directivities used by the room model.
MIC_DIRS_DEG = np.array(
    [
        [0, 21], [32, 0], [0, -21], [328, 0],
        [0, 58], [45, 35], [69, 0], [45, -35],
        [0, -58], [315, -35], [291, 0], [315, 35],
        [91, 69], [90, 32], [90, -31], [89, -69],
        [180, 21], [212, 0], [180, -21], [148, 0],
        [180, 58], [225, 35], [249, 0], [225, -35],
        [180, -58], [135, -35], [111, 0], [135, 35],
        [269, 69], [270, 32], [270, -32], [271, -69],
    ],
    dtype=np.float64,
)


def acn_index(n: int, m: int) -> int:
    """Return the ACN channel index for degree n and order m."""
    return n * n + n + m


def complex_spherical_harmonic(n, m, azimuth, inclination):
    """Complex orthonormal spherical harmonic using SciPy's angle convention."""
    return sph_harm(m, n, azimuth, inclination)


def real_spherical_harmonics(max_order, azimuth, inclination):
    """Evaluate real ACN/N3D spherical harmonics."""
    azimuth = np.asarray(azimuth, dtype=np.float64)
    inclination = np.asarray(inclination, dtype=np.float64)
    if azimuth.shape != inclination.shape:
        raise ValueError("azimuth and inclination must have identical shapes")

    y = np.zeros((azimuth.size, (max_order + 1) ** 2), dtype=np.float64)
    for n in range(max_order + 1):
        for m in range(-n, n + 1):
            channel = acn_index(n, m)
            if m < 0:
                yc = complex_spherical_harmonic(n, -m, azimuth, inclination)
                y[:, channel] = np.sqrt(2.0) * ((-1.0) ** m) * np.imag(yc)
            elif m == 0:
                yc = complex_spherical_harmonic(n, 0, azimuth, inclination)
                y[:, channel] = np.real(yc)
            else:
                yc = complex_spherical_harmonic(n, m, azimuth, inclination)
                y[:, channel] = np.sqrt(2.0) * ((-1.0) ** m) * np.real(yc)

    return np.sqrt(4.0 * np.pi) * y


def rigid_sphere_modal_coefficients(max_order, kr):
    """Rigid-sphere modal coefficients used by the standalone EM32 converter."""
    kr = np.asarray(kr, dtype=np.float64)
    coefficients = np.zeros((kr.size, max_order + 1), dtype=np.complex128)
    nonzero = kr > 1e-12
    z = kr[nonzero]
    coefficients[0, 0] = 1.0

    for n in range(max_order + 1):
        if z.size == 0:
            continue
        jn = spherical_jn(n, z)
        yn = spherical_yn(n, z)
        jn_derivative = spherical_jn(n, z, derivative=True)
        yn_derivative = spherical_yn(n, z, derivative=True)
        hn = jn - 1j * yn
        hn_derivative = jn_derivative - 1j * yn_derivative
        rigid_response = jn - (jn_derivative / hn_derivative) * hn
        coefficients[nonzero, n] = (1j ** n) * rigid_response

    return coefficients


def repeat_per_order(values):
    """Repeat each order-dependent value for its 2*n+1 ACN channels."""
    values = np.asarray(values)
    return np.concatenate([
        np.repeat(values[n], 2 * n + 1) for n in range(values.shape[-1])
    ])


def generate_sht_filters_regls(
    radius,
    mic_dirs_az_el_rad,
    sht_order,
    filter_length,
    sample_rate,
    max_gain_db,
    speed_of_sound=343.0,
):
    """Generate regularized rigid-sphere EM32-to-SH filters.

    This is the same theoretical regLS conversion used in the standalone
    converter. It does not read h_mics from em32_npz_path.
    """
    mic_dirs_az_el_rad = np.asarray(mic_dirs_az_el_rad, dtype=np.float64)
    if mic_dirs_az_el_rad.ndim != 2 or mic_dirs_az_el_rad.shape[1] != 2:
        raise ValueError("Microphone directions must have shape (microphones, 2)")
    if filter_length <= 0 or filter_length % 2 != 0:
        raise ValueError("filter_length must be a positive even integer")

    n_mics = mic_dirs_az_el_rad.shape[0]
    maximum_supported_order = int(np.floor(np.sqrt(n_mics) - 1))
    if sht_order > maximum_supported_order:
        logging.warning(
            "Requested SH order %d is too high for %d microphones; using %d.",
            sht_order, n_mics, maximum_supported_order,
        )
        sht_order = maximum_supported_order

    frequencies = np.arange(filter_length // 2 + 1) * sample_rate / filter_length
    kr = 2.0 * np.pi * frequencies * radius / speed_of_sound
    array_order = min(30, int(np.floor(2.0 * kr[-1])))

    azimuth = mic_dirs_az_el_rad[:, 0]
    elevation = mic_dirs_az_el_rad[:, 1]
    inclination = np.pi / 2.0 - elevation
    y_array = real_spherical_harmonics(array_order, azimuth, inclination)
    modal_coefficients = rigid_sphere_modal_coefficients(array_order, kr)

    n_output_sh = (sht_order + 1) ** 2
    h_frequency = np.zeros(
        (n_output_sh, n_mics, frequencies.size), dtype=np.complex128
    )
    alpha = 10.0 ** (max_gain_db / 20.0)
    beta = 1.0 / (2.0 * alpha)
    regularization = (beta ** 2) * np.eye(n_mics)

    for frequency_index in range(frequencies.size):
        repeated_modal = repeat_per_order(modal_coefficients[frequency_index])
        h_array = y_array * repeated_modal[np.newaxis, :]
        h_array_truncated = h_array[:, :n_output_sh]
        system_matrix = h_array @ h_array.conj().T + regularization
        solution = np.linalg.solve(system_matrix, h_array_truncated)
        h_frequency[:, :, frequency_index] = solution.conj().T

    full_spectrum = np.concatenate(
        [h_frequency, np.conj(h_frequency[:, :, -2:0:-1])], axis=2
    )
    full_spectrum[:, :, filter_length // 2] = np.abs(
        full_spectrum[:, :, filter_length // 2]
    )
    h_time = np.real(np.fft.ifft(full_spectrum, axis=2))
    h_time = np.fft.fftshift(h_time, axes=2)
    return h_frequency, h_time


def get_em32_sht_filters(
    order_sht,
    filter_length,
    sample_rate,
    amp_threshold=15.0,
    radius=0.042,
):
    """Create and cache the standalone-script-compatible EM32 SHT FIR bank."""
    key = (
        int(order_sht), int(filter_length), int(sample_rate),
        float(amp_threshold), float(radius),
    )
    cached = _EM32_SHT_FILTER_CACHE.get(key)
    if cached is not None:
        return cached

    _, filters_time = generate_sht_filters_regls(
        radius=radius,
        mic_dirs_az_el_rad=np.deg2rad(MIC_DIRS_DEG),
        sht_order=order_sht,
        filter_length=filter_length,
        sample_rate=sample_rate,
        max_gain_db=amp_threshold,
    )
    filters_time = np.ascontiguousarray(filters_time)
    _EM32_SHT_FILTER_CACHE[key] = filters_time
    return filters_time


def encode_em32_to_sh(microphone_signals, filters_time, preserve_length=True):
    """Apply the centered FIR bank exactly as in the standalone converter.

    Parameters use the standalone layout: microphone_signals is samples x mics;
    the returned array is samples x SH channels.
    """
    microphone_signals = np.asarray(microphone_signals, dtype=np.float64)
    filters_time = np.asarray(filters_time, dtype=np.float64)
    if microphone_signals.ndim != 2:
        raise ValueError("microphone_signals must have shape (samples, microphones)")
    if filters_time.ndim != 3:
        raise ValueError("filters_time must have shape (SH, microphones, taps)")

    n_samples, n_input_mics = microphone_signals.shape
    n_sh, n_filter_mics, filter_length = filters_time.shape
    if n_input_mics != n_filter_mics:
        raise ValueError(
            f"Input has {n_input_mics} microphone channels, but filters expect "
            f"{n_filter_mics}."
        )

    output_length = n_samples if preserve_length else n_samples + filter_length - 1
    sh_signals = np.zeros((output_length, n_sh), dtype=np.float64)
    for sh_index in range(n_sh):
        accumulated = np.zeros(n_samples + filter_length - 1, dtype=np.float64)
        for mic_index in range(n_input_mics):
            accumulated += fftconvolve(
                microphone_signals[:, mic_index],
                filters_time[sh_index, mic_index, :],
                mode="full",
            )
        sh_signals[:, sh_index] = accumulated[:output_length]
    return sh_signals


def convert_em32_to_sh(
    microphone_signals,
    order_sht,
    sample_rate,
    amp_threshold=15.0,
    radius=0.042,
    filter_length=1024,
):
    """Encode EM32 channels to real ACN/N3D SH using theoretical regLS filters.

    em32_npz_path is intentionally not used here: the NPZ contains measured
    microphone directional responses, not precomputed microphone-to-SH filters.
    """
    signals = np.asarray(microphone_signals)
    if signals.ndim != 2 or signals.shape[0] != 32:
        raise ValueError(f"Expected EM32 signals with shape (32, samples), got {signals.shape}")

    filters_time = get_em32_sht_filters(
        order_sht=order_sht,
        filter_length=filter_length,
        sample_rate=sample_rate,
        amp_threshold=amp_threshold,
        radius=radius,
    )
    sh = encode_em32_to_sh(
        microphone_signals=signals.T,
        filters_time=filters_time,
        preserve_length=True,
    )
    return sh.T

def getMixingTimeEstimateFromVolume(volume):
    # Cremer, L.; Mueller, H. A. (1978): Die wissenschaftlichen Grundlagen der Raumakustik
    return 2 * np.sqrt(volume) / 1000.0

def addStochasticLateReverb(srir, rt, fs, mixingTime, volume):
        '''
        addIsotropicLateReverb add a diffuse reverberant tail

        '''

        timeConstants = rt * fs / 6.9078

        irLength_samp = srir.shape[0]
        numNoiseSources = srir.shape[1]

        noise = np.random.normal(0, 1, size = (irLength_samp, numNoiseSources))

        path = os.path.join(os.path.dirname(__file__), 'octaveFilters.json')
        with open(path) as f:
            octaveFilters = json.load(f)
            filtersIrs = np.asarray(octaveFilters['ir'])

        numFilterBands = filtersIrs.shape[1]
        noisePerSourceBand = np.zeros((irLength_samp, numNoiseSources, numFilterBands))

        tax = np.arange(irLength_samp)

        #if volume:
        #    mixingTime_samp = int(getMixingTimeEstimateFromVolume(volume) * fs)
        #else:
        mixingTime_samp = int(mixingTime * fs)

        fadeLength_samp = int(15.0 / 1000.0 * fs)

        if irLength_samp - mixingTime_samp - fadeLength_samp > 0:
            window = np.hstack((np.zeros(mixingTime_samp),
                                np.sin(np.arange(fadeLength_samp) / fadeLength_samp * np.pi / 2) ** 2,
                                np.ones(irLength_samp - mixingTime_samp - fadeLength_samp)))
        else:
            window = np.ones(irLength_samp)
            print('no late reverb added - response too short')


        for iBand in range(numFilterBands):
            envelope = np.exp(-tax / timeConstants[iBand])
            for iNoiseSource in range(numNoiseSources):
                noisePerSourceBand[:, iNoiseSource, iBand] = np.multiply(
                    np.convolve(filtersIrs[:, iBand], noise[:, iNoiseSource], mode = 'same'), envelope)

        noisePerSource = np.sum(noisePerSourceBand, axis = 2)

        # compute the rms of the simulated response and match the noise response
        # (needs sufficient IS order to deliver reasonable result)

        rmsIs = np.sqrt(np.sum(srir[mixingTime_samp:, :] ** 2, axis = 0))
        rmsNoise = np.sqrt(np.sum(noisePerSource[mixingTime_samp:, :] ** 2, axis = 0))

        noiseLate = np.multiply(noisePerSource, np.expand_dims(rmsIs / rmsNoise, axis= (0)))

        srir = srir * np.expand_dims(1-window , axis = 1 ) + noiseLate  * np.expand_dims(window , axis = 1 )

        return srir


def addStochasticLateReverbPyfar(srir, rt, fs, mixingTime, volume, centerFreqs=None,
                                 num_fractions=1, filter_order=14,
                                 peak_threshold=0.1, peak_prominence=0.01):
    """
    Add a diffuse stochastic late reverberant tail using pyfar octave filters.
    """

    srir = srir.T
    irLength_samp, numNoiseSources = srir.shape
    rt = np.asarray(rt).flatten()

    if centerFreqs is not None:
        centerFreqs = np.asarray(centerFreqs).flatten()
        valid = centerFreqs <= fs / 2
        rt = rt[valid]
        centerFreqs = centerFreqs[valid]
        frequency_range = (float(centerFreqs[0]), float(centerFreqs[-1]))
    else:
        frequency_range = (20.0, fs / 2)

    noise = np.random.normal(0, 1, size=(irLength_samp, numNoiseSources))
    tax = np.arange(irLength_samp)

    peak_signal = np.abs(srir[:, 0])
    peak_max = np.max(peak_signal)

    peaks, _ = find_peaks(peak_signal,
                          height=peak_threshold * peak_max,
                          prominence=peak_prominence * peak_max)

    first_peak_samp = int(peaks[0]) if len(peaks) else int(np.argmax(peak_signal))
    mixingTime_samp = min(first_peak_samp + int(round(mixingTime * fs)), irLength_samp)

    fadeLength_samp = int(15e-3 * fs)

    if irLength_samp - mixingTime_samp - fadeLength_samp > 0:
        window = np.hstack([np.zeros(mixingTime_samp),
                            np.sin(np.arange(fadeLength_samp) / fadeLength_samp * np.pi / 2) ** 2,
                            np.ones(irLength_samp - mixingTime_samp - fadeLength_samp)])
    else:
        window = np.ones(irLength_samp)

    noise_signal = pf.Signal(noise.T, fs)

    noise_bands_signal = pf.dsp.filter.fractional_octave_bands(
        noise_signal,
        num_fractions=num_fractions,
        frequency_range=frequency_range,
        order=filter_order)

    noisePerSourceBand = np.moveaxis(noise_bands_signal.time, -1, 0)

    if noisePerSourceBand.shape[1] != numNoiseSources:
        noisePerSourceBand = np.swapaxes(noisePerSourceBand, 1, 2)

    numFilterBands = noisePerSourceBand.shape[2]

    if len(rt) != numFilterBands:
        raise ValueError(f"RT vector has {len(rt)} bands, pyfar produced {numFilterBands}")

    timeConstants = rt * fs / 6.9078

    for iBand in range(numFilterBands):
        noisePerSourceBand[:, :, iBand] *= np.exp(-tax / timeConstants[iBand])[:, None]

    noisePerSource = np.sum(noisePerSourceBand, axis=2)

    energy = np.sum(srir ** 2, axis=1)
    late_energy_ism = np.sum(energy[mixingTime_samp:])

    start = first_peak_samp + int(5e-3 * fs)
    end = mixingTime_samp

    schroeder_late_energy = 0.0

    if end > start + 20:
        early_energy = energy[start:end]
        schroeder = np.cumsum(early_energy[::-1])[::-1]

        if schroeder[0] > 0:
            schroeder_db = 10 * np.log10(np.maximum(schroeder / schroeder[0], 1e-12))
            schroeder_db = gaussian_filter1d(schroeder_db, sigma=3)

            t = np.arange(len(schroeder_db)) / fs
            slope, intercept = np.polyfit(t, schroeder_db, 1)

            t_late = (np.arange(mixingTime_samp, irLength_samp) - start) / fs
            predicted_db = slope * t_late + intercept

            schroeder_late_energy = np.sum(10 ** (predicted_db / 10))

    if schroeder_late_energy > 0:
        confidence = late_energy_ism / (late_energy_ism + schroeder_late_energy)
        alpha = np.clip(confidence * 2.0, 0, 1)
        estimated_late_energy = alpha * late_energy_ism + (1 - alpha) * schroeder_late_energy
    else:
        estimated_late_energy = late_energy_ism

    noise_energy = np.sum(noisePerSource[mixingTime_samp:] ** 2)
    scale = np.sqrt(estimated_late_energy / noise_energy) if noise_energy > 0 else 0.0

    noiseLate = noisePerSource * scale

    rir_early = srir * (1 - window)[:, None]
    late_reverb = noiseLate * window[:, None]
    rir = rir_early + late_reverb

    return rir.T, rir_early.T, late_reverb.T



def apply_tikhonov_difference_filters(
    signal,
    fs,
    R=0.042,
    Nmic=32,
    amp_threshold=15,
    c=343.0
):
    
    """
    Apply order-dependent Moreau/Daniel difference filters to Ambisonic signals.

    Parameters
    ----------
    signal : ndarray
        Ambisonic signal with shape (channels, samples).
    fs : float
        Sampling rate in Hz.
    R : float
        Microphone array radius in meters.
    Nmic : int
        Number of microphones.
    amp_threshold : float
        Regularization threshold in dB.
    c : float
        Speed of sound in m/s.

    Returns
    -------
    filtered : ndarray
        Filtered Ambisonic signal with shape (channels, samples).
    """

    signal = signal.T
    signal = np.asarray(signal)
    dtype_in = signal.dtype
    signal = signal.astype(np.float64)

    if signal.ndim != 2:
        raise ValueError("signal must have shape (channels, samples).")

    n_samples, n_channels = signal.shape

    order_sht = int(np.sqrt(n_channels) - 1)

    if (order_sht + 1) ** 2 != n_channels:
        raise ValueError("Number of channels must be a square Ambisonics channel count.")

    # -----------------------------
    # Frequency axis for FFT bins
    # -----------------------------
    f = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    kR = 2 * np.pi * f * R / c

    kR_safe = kR.copy()
    kR_safe[0] = 1e-12

    # -----------------------------
    # Rigid-sphere modal coefficients
    # -----------------------------
    bN = rigid_sphere_modal_coeffs(order_sht, kR_safe) / (4 * np.pi)

    # DC handling
    bN[0, 1:] = 0.0

    # -----------------------------
    # Moreau/Daniel regularization
    # -----------------------------
    alpha = np.sqrt(Nmic) * 10 ** (amp_threshold / 20)

    if alpha <= 1:
        raise ValueError("amp_threshold is too low: alpha must be greater than 1.")

    beta = np.sqrt(
        (1 - np.sqrt(1 - 1 / alpha**2)) /
        (1 + np.sqrt(1 - 1 / alpha**2))
    )

    # -----------------------------
    # Difference filters
    #
    # H_reg = conj(bN) / (abs(bN)^2 + beta^2)
    # H_opt = 1 / bN
    #
    # Difference filter:
    # H_diff = H_reg / H_opt
    #        = H_reg * bN
    #        = abs(bN)^2 / (abs(bN)^2 + beta^2)
    # -----------------------------
    H_diff = np.abs(bN) ** 2 / (np.abs(bN) ** 2 + beta**2)

    # Avoid filtering order 0 at DC incorrectly
    H_diff[0, 0] = 1.0

    # -----------------------------
    # Apply filters in frequency domain
    # -----------------------------
    X = np.fft.rfft(signal, axis=0)

    for n in range(order_sht + 1):
        ch_start = n * n
        ch_end = (n + 1) * (n + 1)

        X[:, ch_start:ch_end] *= H_diff[:, n, None]

    filtered = np.fft.irfft(X, n=n_samples, axis=0)

    return filtered.T.astype(dtype_in)

def rigid_sphere_modal_coeffs(max_order, x):
    b = np.zeros((len(x), max_order + 1), dtype=complex)

    for n in range(max_order + 1):
        jn = spherical_jn(n, x)
        jn_der = spherical_jn(n, x, derivative=True)

        yn = spherical_yn(n, x)
        yn_der = spherical_yn(n, x, derivative=True)

        hn2 = jn - 1j * yn
        hn2_der = jn_der - 1j * yn_der

        b[:, n] = 4 * np.pi * (1j ** n) * (
            jn - (jn_der / hn2_der) * hn2
        )

    return b


def simulate_room(
    room_idx: int,
    pos_idx: int,
    output_path: str,
    physical_params: list,
    array_type: str = "ambisonic",
    ambi_order: int = 3,
    sample_rate: int = 16000,
    em32_npz_path: str = None,
    em32_radius: float = 0.042,
    amp_threshold: float = 15.0,
    em32_filter_length: int = 1024,
    max_order: int = 12,
    export_mode: str = "ism_late",
) -> Union[dict, None]:
    """
    Simulates a single room impulse response (RIR) for a given room configuration
    and source/microphone position using pyroomacoustics. Returns the RIR array and
    acoustic metadata to the parent process. Includes error handling.

    Args:
        room_idx: Index of the current room simulation.
        pos_idx: Index of the current source/microphone position within the room.
        output_path: Output directory. Pickle writing is handled by the parent process.
        ambisonic_microphone: Tuple defining the Ambisonic microphone array
                              (positions, sample_rate, directivities).
        physical_params: A list containing the physical parameters of the room and
                         source/mic positions:
                         [Lx, Ly, Lz, materials, Sx, Sy, Sz, Rx, Ry, Rz].

    Returns:
        dict: A dictionary containing metadata and acoustic parameters for the
              generated RIR if successful.
        None: If an error occurred during simulation.
    """
    valid_export_modes = {"ism", "ism_late", "direct_late"}
    if export_mode not in valid_export_modes:
        raise ValueError(
            f"export_mode must be one of {sorted(valid_export_modes)}, got {export_mode!r}"
        )
    if max_order < 0:
        raise ValueError(f"max_order must be >= 0, got {max_order}")

    room_name = f"R{room_idx:04d}_P{pos_idx}"
    rir_name = f"{room_name}_IR"

    # Extract physical parameters with clear variable names
    (
        Lx, Ly, Lz, Sx, Sy, Sz, Rx, Ry, Rz, rt60,
        e_absorption_rest, e_absorption_selected_wall, centerFreqs,
        idxDifferentWall, floor_corners, volume, array_azimuth_rad,
    ) = physical_params

    logging.info(f"Starting simulation for {room_name} with params: "
                 f"Dims=({Lx[0]:.2f}, {Ly[0]:.2f}, {Lz[0]:.2f}), "
                 f"Src=({Sx[0]:.2f}, {Sy[0]:.2f}, {Sz[0]:.2f}), "
                 f"Mic=({Rx[0]:.2f}, {Ry[0]:.2f}, {Rz[0]:.2f})")

    try:
        # Ensure parameters are simple floats for pyroomacoustics
        room_dims = [Lx[0], Ly[0], Lz[0]]
        source_pos = [Sx[0], Sy[0], Sz[0]]
        mic_center_pos = [Rx[0], Ry[0], Rz[0]]

        # Expensive rotated-directivity/filter generation happens in parallel
        # inside worker processes instead of blocking the serial room loop.
        ambisonic_microphone = _get_worker_array(
            array_type=array_type,
            ambi_order=ambi_order,
            sample_rate=sample_rate,
            array_azimuth_rad=array_azimuth_rad,
            em32_npz_path=em32_npz_path,
            em32_radius=em32_radius,
        )

        # Direction of arrival in the microphone-local coordinate frame.
        # This explicitly accounts for the microphone-array yaw orientation.
        accdoa, doa_az_deg, doa_el_deg = doa_in_mic_frame(
            source_pos,
            mic_center_pos,
            array_azimuth_rad,
        )

        material_properties = [ {'description' : 'uniform_material', 'coeffs' : e_absorption_rest, 'center_freqs' : centerFreqs } for iWall in range(6)]
        material_properties[idxDifferentWall] =  {'description' : 'uniform_material', 'coeffs' : e_absorption_selected_wall, 'center_freqs' : centerFreqs}

        # create wall materials
        materials_side_walls = pra.make_materials(*[(material_properties[i],) for i in range(4)])
        # create ceiling and floor material ceiling and floor
        me = pra.make_materials(floor=(material_properties[4],), ceiling=(material_properties[5],))

        # create the room again, with materals

        wall_keys = ["west", "east", "south", "north", "floor", "ceiling"]
        materials_dict = {wall_keys[i]: (material_properties[i],) for i in range(6)}
        room = pra.ShoeBox(
            p=room_dims,
            materials=pra.make_materials(**materials_dict),
            fs=ambisonic_microphone[1],
            max_order=max_order,
            humidity=0.45,
            air_absorption=True,
        )

        print(f"RT60: {rt60}, Absorption Selected Wall: {e_absorption_selected_wall}, Absorption Rest: {e_absorption_rest}, Selected Wall Index: {idxDifferentWall}, Volume: {volume}, Room Dims: {room_dims}, Source Pos: {source_pos}, Receiver Pos: {mic_center_pos}")


        # Add source and microphone array
        room.add_source(source_pos)
        room.add_microphone_array(
            pra.MicrophoneArray(
                ambisonic_microphone[0] + np.array(mic_center_pos)[:, None], # Adjust mic positions relative to center
                ambisonic_microphone[1], # Sample rate
                ambisonic_microphone[2], # Directivities
            )
        )

        # Compute RIR using Image Source Model
        room.image_source_model()
        room.compute_rir()

        # Post-processing RIRs
        # Find maximum length across all channels to pad
        max_len = np.array([len(r[0]) for r in room.rir]).max()
        # Determine target length for padding (pad to the nearest second)
        target_len = int(np.ceil(max_len / room.fs) * room.fs)

        # Stack and pad RIRs
        RIR = np.stack(
            [np.pad(r[0], (0, target_len - r[0].shape[0])) for r in room.rir],
            axis=0, # Stack along a new dimension (channels)
        )


        # Compute a direct-only RIR only when it is needed for export.
        RIR_direct = None
        if export_mode == "direct_late":
            room_direct = pra.ShoeBox(
                p=room_dims,
                materials=pra.make_materials(**materials_dict),
                fs=ambisonic_microphone[1],
                max_order=0,
                humidity=0.45,
                air_absorption=True,
            )
            room_direct.add_source(source_pos)
            room_direct.add_microphone_array(
                pra.MicrophoneArray(
                    ambisonic_microphone[0] + np.array(mic_center_pos)[:, None],
                    ambisonic_microphone[1],
                    ambisonic_microphone[2],
                )
            )
            room_direct.image_source_model()
            room_direct.compute_rir()

            RIR_direct = np.stack(
                [
                    np.pad(r[0], (0, target_len - r[0].shape[0]))
                    for r in room_direct.rir
                ],
                axis=0,
            )


        '''
        # Normalize RIR by the maximum absolute value across all channels
        max_rir_val = np.max(np.abs(RIR))
        if max_rir_val > 0:
            RIR = RIR / max_rir_val
        else:
             # Handle cases where the RIR might be all zeros (e.g., source/mic outside room)
             logging.warning(f"RIR for {room_name} is all zeros or near zero. Skipping acoustic parameter calculation and file writing.")
             return {
                "Name": rir_name,
                "RT": np.nan, # Use NaN for undefined values
                "c50": np.nan,
                "x_room": Lx[0], "y_room": Ly[0], "z_room": Lz[0],
                "x_source": Sx[0], "y_source": Sy[0], "z_source": Sz[0],
                "x_mic": Rx[0], "y_mic": Ry[0], "z_mic": Rz[0],
                "status": "Warning: RIR all zeros"
             }
        '''

        # The previous code generated a complete stochastic late-reverb tail
        # but did not assign it back to RIR. That expensive no-op is omitted.

        if array_type == "em32":
            if em32_npz_path is None:
                raise ValueError("em32_npz_path is required for EM32 simulation")
            RIR = convert_em32_to_sh(
                RIR,
                order_sht=ambi_order,
                sample_rate=room.fs,
                amp_threshold=amp_threshold,
                radius=em32_radius,
                filter_length=em32_filter_length,
            )
            if RIR_direct is not None:
                RIR_direct = convert_em32_to_sh(
                    RIR_direct,
                    order_sht=ambi_order,
                    sample_rate=room.fs,
                    amp_threshold=amp_threshold,
                    radius=em32_radius,
                    filter_length=em32_filter_length,
                )
        else:
            # Preserve the original ideal-HOA post-filtering path.
            RIR = apply_tikhonov_difference_filters(
                RIR, room.fs, R=em32_radius, Nmic=32,
                amp_threshold=amp_threshold, c=343.0
            )
            if RIR_direct is not None:
                RIR_direct = apply_tikhonov_difference_filters(
                    RIR_direct, room.fs, R=em32_radius, Nmic=32,
                    amp_threshold=amp_threshold, c=343.0
                )

        # Select exactly what is exported.
        if export_mode == "ism":
            # Export only the image-source-model RIR. No direct-only room and no
            # stochastic late reverberation are needed.
            pass
        elif export_mode == "ism_late":
            # Keep the ISM early part and add the stochastic late tail.
            _, RIR_early, RIR_late = addStochasticLateReverbPyfar(
                srir=RIR,
                rt=rt60,
                fs=room.fs,
                mixingTime=0.08,
                volume=volume,
                centerFreqs=centerFreqs,
                num_fractions=1,
                filter_order=14,
            )
            RIR = RIR_early + RIR_late
        else:  # export_mode == "direct_late"
            # The late tail is still estimated from the full ISM response, but
            # the exported early component is direct sound only.
            _, _, RIR_late = addStochasticLateReverbPyfar(
                srir=RIR,
                rt=rt60,
                fs=room.fs,
                mixingTime=0.0035,
                volume=volume,
                centerFreqs=centerFreqs,
                num_fractions=1,
                filter_order=14,
            )
            RIR = RIR_direct + RIR_late

        # Pad final RIR to the next full second
        current_len = RIR.shape[1]
        target_len = int(np.ceil(current_len / room.fs) * room.fs)

        if target_len > current_len:
            RIR = np.pad(
                RIR,
                pad_width=((0, 0), (0, target_len - current_len)),
                mode="constant"
            )

        # Calculate Acoustic Parameters (RT60 and C50)
        # Calculate RT60 from the first channel (often representative)
        # Add a check if the RIR is long enough for RT60 calculation
        rt60_sim = np.nan # Default to NaN
        if len(RIR[0]) > room.fs * 0.1: # Ensure at least 100ms for decay calculation
             rt60_sim = pra.experimental.rt60.measure_rt60(RIR[0], fs=room.fs, decay_db=30)
        else:
             logging.warning(f"RIR for {room_name} too short ({len(RIR[0])} samples) for RT60 calculation.")


        # Calculate C50 from the first channel
        c50 = np.nan # Default to NaN
        # Ensure the RIR is long enough to have samples at 50ms
        if len(RIR[0]) > int(room.fs * 0.05):
            early_energy = np.sum(np.square(np.abs(RIR[0, : int(room.fs * 0.05)])))
            late_energy = np.sum(np.square(np.abs(RIR[0, int(room.fs * 0.05) : ])))

            if late_energy > 0: # Avoid division by zero
                 c50 = 10 * np.log10(early_energy / late_energy)
            else:
                 logging.warning(f"Late energy is zero for C50 calculation in {room_name}.")
                 c50 = np.inf # Or handle as appropriate for your analysis

        else:
             logging.warning(f"RIR for {room_name} too short for C50 calculation (needs > 50ms).")


        logging.info(f"Successfully simulated {room_name}. RT60: {rt60_sim:.2f}, C50: {c50:.2f}")

        # Return metadata and results
        # Frequency-dependent metadata
        rt_profile_metadata = {
            f"rt60_{int(freq)}Hz": float(rt)
            for freq, rt in zip(centerFreqs, rt60)
        }
        return {
            "Name": rir_name,
            "room_idx": int(room_idx),
            "position_id": str(pos_idx),
            "sample_rate": int(room.fs),
            "rir": np.ascontiguousarray(RIR, dtype=np.float32),
            "RT": rt60_sim,
            "c50": c50,
            "x_room": Lx[0], "y_room": Ly[0], "z_room": Lz[0],
            "x_source": round(float(Sx[0]), 2), "y_source": round(float(Sy[0]), 2), "z_source": round(float(Sz[0]), 2),
            "x_mic": round(float(Rx[0]), 2), "y_mic": round(float(Ry[0]), 2), "z_mic": round(float(Rz[0]), 2),
            "mic_orientation_az_deg": float(np.degrees(array_azimuth_rad)),
            "accdoa_x": float(accdoa[0]),
            "accdoa_y": float(accdoa[1]),
            "accdoa_z": float(accdoa[2]),
            "az_deg": float(doa_az_deg),
            "el_deg": float(doa_el_deg),
            **rt_profile_metadata,
            "max_order": int(max_order),
            "export_mode": export_mode,
            "status": "Success"
        }

    except ValueError as ve:
        error_msg = f"ValueError for {room_name}: {ve}"
        logging.error(error_msg)
        return {
            "Name": rir_name,
            "status": "Error: ValueError",
            "error_message": str(ve),
            "x_room": Lx[0], "y_room": Ly[0], "z_room": Lz[0],
            "x_source": Sx[0] if 'Sx' in locals() else np.nan, # Use locals() to check if variable exists
            "y_source": Sy[0] if 'Sy' in locals() else np.nan,
            "z_source": Sz[0] if 'Sz' in locals() else np.nan,
            "x_mic": Rx[0] if 'Rx' in locals() else np.nan,
            "y_mic": Ry[0] if 'Ry' in locals() else np.nan,
            "z_mic": Rz[0] if 'Rz' in locals() else np.nan,
            "mic_orientation_az_deg": float(np.degrees(array_azimuth_rad)) if 'array_azimuth_rad' in locals() else np.nan,
            "accdoa_x": float(accdoa[0]) if 'accdoa' in locals() else np.nan,
            "accdoa_y": float(accdoa[1]) if 'accdoa' in locals() else np.nan,
            "accdoa_z": float(accdoa[2]) if 'accdoa' in locals() else np.nan,
            "az_deg": float(doa_az_deg) if 'doa_az_deg' in locals() else np.nan,
            "el_deg": float(doa_el_deg) if 'doa_el_deg' in locals() else np.nan,
        }
    except Exception as e:
        # Catch any other exceptions
        error_msg = f"An unexpected error occurred during simulation for {room_name}: {e}"
        logging.error(error_msg, exc_info=True) # Log traceback for unexpected errors
        return {
            "Name": rir_name,
            "status": "Error: Unexpected Exception",
            "error_message": str(e),
            "x_room": Lx[0], "y_room": Ly[0], "z_room": Lz[0],
             "x_source": Sx[0] if 'Sx' in locals() else np.nan,
            "y_source": Sy[0] if 'Sy' in locals() else np.nan,
            "z_source": Sz[0] if 'Sz' in locals() else np.nan,
            "x_mic": Rx[0] if 'Rx' in locals() else np.nan,
            "y_mic": Ry[0] if 'Ry' in locals() else np.nan,
            "z_mic": Rz[0] if 'Rz' in locals() else np.nan,
            "mic_orientation_az_deg": float(np.degrees(array_azimuth_rad)) if 'array_azimuth_rad' in locals() else np.nan,
            "accdoa_x": float(accdoa[0]) if 'accdoa' in locals() else np.nan,
            "accdoa_y": float(accdoa[1]) if 'accdoa' in locals() else np.nan,
            "accdoa_z": float(accdoa[2]) if 'accdoa' in locals() else np.nan,
            "az_deg": float(doa_az_deg) if 'doa_az_deg' in locals() else np.nan,
            "el_deg": float(doa_el_deg) if 'doa_el_deg' in locals() else np.nan,
        }


def main(
    output_path: str,
    num_rooms: int,
    num_positions: int,
    ambi_order: int = 3,
    sample_rate: int = 16000,
    num_sources_per_receiver: int = 3,
    array_type: str = "ambisonic",
    em32_npz_path: str = "Eigenmike_em32_IRs.npz",
    em32_radius: float = 0.042,
    amp_threshold: float = 15.0,
    em32_filter_length: int = 1024,
    max_order: int = 12,
    export_mode: str = "ism_late",
):
    """
    Main function to generate a dataset of RIRs with varying room acoustics
    and source/microphone positions.
    Args:
        output_path: Directory to save one pickle per room and metadata CSV files.
        num_rooms: The number of different room geometries to simulate.
        num_positions: The number of different source/microphone positions
                       within each room geometry.
        ambi_order: The Ambisonic order for the microphone array.
        sample_rate: The sample rate for the simulations.
    """
    start = time.time() # Start time for performance measurement

    start_room, end_room = get_room_range(num_rooms)

    # --- NEU: Realistische RT-Profile laden ---
    rtProfiles = np.load('rtProfilesMit.npy')
    centerFreqs = np.asanyarray(125 * 2**np.arange(0, 8)).flatten()
    numRtProfiles = rtProfiles.shape[0]
    logging.info(f"Loaded {numRtProfiles} RT60 profiles")

    c = 343 # Speed of sound in m/s
    max_displacement_m = 0.5
    anisotropic = False #np.random.rand() > 0.5
    anisotropicType = np.random.rand() > 0.5 # should it be more or less absorption

    logging.info(
        f"SLURM task processing rooms {start_room} to {end_room-1}"
    )

    logging.info(f"Starting RIR generation: {num_rooms} rooms, {num_positions} positions per room, "
                 f"Ambisonic Order {ambi_order}, Sample Rate {sample_rate}")

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Validate that an Ambisonic array can be constructed. A separately rotated
    # array is created once for each room below.
    try:
        if array_type == "em32":
            create_em32_array(
                em32_npz_path=em32_npz_path,
                sample_rate=sample_rate,
                azimuth_rotation=0.0,
                radius=em32_radius,
            )
            logging.info(
                "EM32 measured array validated; output will be encoded to "
                f"order {ambi_order} ACN/N3D SH ({(ambi_order + 1) ** 2} channels)."
            )
        else:
            create_ambisonic_array(
                order_of_ambisonics=ambi_order,
                sample_rate=sample_rate,
                azimuth_rotation=0.0,
            )
            logging.info(
                f"Ambisonic array configuration validated for order {ambi_order} "
                f"({(ambi_order + 1) ** 2} channels)."
            )
    except Exception as e:
        logging.critical(f"Failed to create Ambisonic microphone array: {e}")
        return

    all_results = []  # Lightweight metadata rows only; no RIR arrays.
    room_results = {}  # Completed RIRs grouped by room until each pickle is written.
    expected_rirs_per_room = num_positions * num_sources_per_receiver

    # Use ProcessPoolExecutor for parallel processing
    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        # Outer loop for different room geometries
        i = start_room
        with tqdm(total=end_room - start_room, desc="Generating Rooms") as pbar:
            while i < end_room:

                rtProfileIdx = np.random.randint(0, numRtProfiles)
                rt60 = rtProfiles[rtProfileIdx, :]

                while True:
                    # Generate room dimensions
                    Lx, Ly, Lz = get_random_dimensions()
                    room_dims = [Lx[0], Ly[0], Lz[0]]
                    #volume = room_dims[0] * room_dims[1] * room_dims[2]


                    try:
                        # use a shoebox as a base
                        roomShoebox = pra.ShoeBox(
                            room_dims, fs=sample_rate,
                            use_rand_ism = False, max_order=0
                        )
                        floor_corners_shoebox = roomShoebox.walls[4].corners # corners of floor

                        # and displace corners before creating 3D room from it
                        floor_corners = floor_corners_shoebox[:2, :] + np.random.rand(2, 4) * max_displacement_m

                        # now create a dummy room with those dimensions
                        room_test = pra.Room.from_corners(floor_corners,
                                                max_order = 0, fs=sample_rate)

                        room_test.extrude(room_dims[2]) # "add ceiling"

                        room_test.set_sound_speed(c)

                        volume = room_test.get_volume()

                        
                        # mean absorption -  should replace with actual room volume ...
                        #e_absorption = np.array([pra.inverse_sabine(rtOctaveband, room_dims, c = c)[0] for rtOctaveband in rt60])
                        e_absorption = inverse_eyring(rt60, room_dims, c=c)

                                #125  250  500  1000 2000 4000 8000 16000 Hz
                        #a_values = [4.0, 2.3, 2.0, 2.0, 2.3, 5.0, 7.0, 9.0] # keep 'a' constant across frequencies
                        #b_values = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
                        #e_absorption = generate_frequency_absorption(centerFreqs, a_values, b_values)

                        # random anisotropy index between 0.5 and 1.5,
                        # which means 50% more or less absorption on one of the walls
                        #anisotropy_index = np.random.rand() + 0.5

                        if  anisotropic: # for half the samples, make
                            anisotropy_index  = np.random.rand() * 5 + 5 # between 5 and 15 times
                            if anisotropicType:
                                anisotropy_index  = 1 / anisotropy_index # more or less absorption
                        else: # for the other half, don't
                            anisotropy_index = 1

                        # random wall to have a different absorption
                        idxDifferentWall = np.random.randint(0, 6)

                        # get surface are as
                        area = [room_test.wall_area(room_test.walls[iWall]) for iWall in range(6)]

                        area_all = np.sum(np.array(area))
                        area_selected_wall = np.array(area[idxDifferentWall])
                        area_rest = area_all - area_selected_wall

                        e_absorption_rest = np.array(e_absorption * area_rest / (anisotropy_index * area_selected_wall + area_rest))
                        e_absorption_selected_wall = np.array(anisotropy_index * e_absorption_rest)

                        concat_absorption_values = np.hstack((e_absorption_rest, e_absorption_selected_wall))

                        if any(concat_absorption_values < np.array(0.01)) | any(concat_absorption_values > np.array(0.99)):
                            print('absorption values out of bounds, need to find a different room size')
                            continue # regenerate room dimensions and absorption values

                        break

                    except ValueError:
                        print('need to find a different room size')
                        continue
                
                # Inner loop for different source/mic positions within the current room
                accepted_receivers = []
                accepted_sources_per_receiver = []
                j = 0
                tries_receiver = 0
                while j < num_positions and tries_receiver < 1000:
                    min_source_angle_difference = 15  # choose constraint
                    min_dist_from_receiver = 1.0 # in meters
                    max_dist_from_receiver = 50.0 # in meters
                    
                    tries_receiver += 1
                    _, _, _, Rx, Ry, Rz = get_random_positions(Lx[0], Ly[0], Lz[0], min_dist_from_wall=0.5)
                    receiver = np.array([Rx[0], Ry[0], Rz[0]])
                    # Reject if receiver already exists
                    if any(np.allclose(receiver, r) for r in accepted_receivers):
                        continue

                    accepted_dirs = []
                    accepted_sources = []

                    k = 0
                    tries_source = 0
                    while k < num_sources_per_receiver and tries_source < 1000:
                        tries_source += 1
                        Sx, Sy, Sz, _, _, _ = get_random_positions(Lx[0], Ly[0], Lz[0], min_dist_from_wall=0.5)
                        source = np.array([Sx[0], Sy[0], Sz[0]])

                        # avoid degenerate case
                        if np.allclose(source, receiver):
                            continue

                        dist = np.linalg.norm(source - receiver)
                        if dist < min_dist_from_receiver or dist > max_dist_from_receiver:
                            continue

                        direction = doa_unit_vector(source, receiver)

                        # check angular constraint
                        if is_too_close(direction, np.array(accepted_dirs), min_source_angle_difference):
                            continue

                        accepted_dirs.append(direction)
                        accepted_sources.append(source)
                        k += 1

                    if len(accepted_sources) < num_sources_per_receiver:
                        print(f'Could only find {len(accepted_sources)} sources for receiver {j} in room {i} after {tries_source} tries, regenerating receiver position.')
                        continue # regenerate source position for this room and position index

                    accepted_receivers.append(receiver)
                    accepted_sources_per_receiver.append(accepted_sources)
                    j += 1 # only increment position index if we have accepted sources

                if len(accepted_receivers) < num_positions:
                    print(f'Could only find {len(accepted_receivers)} receivers for room {i} after {tries_receiver} tries, regenerating room.')
                    continue # find a different room and start over with position generation

                for k, receiver in enumerate(accepted_receivers):
                    Rx = np.array([receiver[0]])
                    Ry = np.array([receiver[1]])
                    Rz = np.array([receiver[2]])

                    # Draw a new microphone-array yaw for every receiver position.
                    # All sources belonging to this receiver share the same orientation.
                    array_azimuth_rad = float(np.random.uniform(0.0, 2.0 * np.pi))
                    logging.info(
                        f"Room {i}, receiver position {k}: microphone-array azimuth = "
                        f"{np.degrees(array_azimuth_rad):.3f} deg "
                        f"({array_azimuth_rad:.6f} rad)"
                    )

                    for l, source in enumerate(accepted_sources_per_receiver[k]):
                        Sx = np.array([source[0]])
                        Sy = np.array([source[1]])
                        Sz = np.array([source[2]])

                        physical_params = [
                            Lx, Ly, Lz, Sx, Sy, Sz, Rx, Ry, Rz, rt60,
                            e_absorption_rest, e_absorption_selected_wall, centerFreqs,
                            idxDifferentWall, floor_corners, volume, array_azimuth_rad,
                        ]

                        futures.append(
                            executor.submit(
                                simulate_room,
                                i,
                                f"{k}_S{l}",
                                output_path,
                                physical_params,
                                array_type,
                                ambi_order,
                                sample_rate,
                                em32_npz_path,
                                em32_radius,
                                amp_threshold,
                                em32_filter_length,
                                max_order,
                                export_mode,
                            )
                        )
                i += 1
                pbar.update(1)

        # Process the results as they complete. Each worker returns one RIR; the
        # parent groups all receiver/source combinations belonging to one room and
        # writes exactly one pickle for that room.
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Results"):
            result = future.result()
            if not result:
                continue

            if result.get("status") != "Success":
                all_results.append(result)
                continue

            room_idx = int(result["room_idx"])
            position_id = str(result["position_id"])

            room_entry = room_results.setdefault(
                room_idx,
                {
                    "room_idx": room_idx,
                    "sample_rate": int(result["sample_rate"]),
                    "rirs": {},
                },
            )

            # Store channels x samples without transposing or WAV quantization.
            metadata = {
                key: value
                for key, value in result.items()
                if key not in {"rir", "room_idx", "position_id", "sample_rate"}
            }
            room_entry["rirs"][position_id] = {
                "rir": result["rir"],
                "metadata": metadata,
            }

            # The CSV contains metadata only. Keeping the array out prevents pandas
            # from serializing large ndarray representations into cells.
            all_results.append({key: value for key, value in result.items() if key != "rir"})

            if len(room_entry["rirs"]) == expected_rirs_per_room:
                pickle_path = os.path.join(output_path, f"R{room_idx:04d}.pkl")
                with open(pickle_path, "wb") as pickle_file:
                    pickle.dump(room_entry, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

                logging.info(
                    "Saved room %d with %d RIRs to %s",
                    room_idx,
                    expected_rirs_per_room,
                    pickle_path,
                )
                del room_results[room_idx]

    # Save any room for which one or more simulations failed. These files are
    # explicitly marked incomplete instead of silently discarding valid RIRs.
    for room_idx, room_entry in room_results.items():
        pickle_path = os.path.join(output_path, f"R{room_idx:04d}_incomplete.pkl")
        with open(pickle_path, "wb") as pickle_file:
            pickle.dump(room_entry, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
        logging.warning(
            "Saved incomplete room %d with %d of %d RIRs to %s",
            room_idx,
            len(room_entry["rirs"]),
            expected_rirs_per_room,
            pickle_path,
        )

    # Separate successful results from errors for the main data CSV
    successful_results = [res for res in all_results if res and res.get("status") == "Success"]
    error_results = [res for res in all_results if res and res.get("status") != "Success"]

    # Save successful results to the main data CSV
    if successful_results:
        data_df = pd.DataFrame(successful_results)

        # Keep CSV rows in deterministic room/receiver/source order.
        # Expected identifier format: R0000_P<receiver>_S<source>_IR
        sort_keys = data_df["Name"].str.extract(
            r"^R(?P<room>\d+)_P(?P<receiver>\d+)_S(?P<source>\d+)_IR$"
        ).astype("Int64")
        data_df = (
            data_df.assign(
                _room_order=sort_keys["room"],
                _receiver_order=sort_keys["receiver"],
                _source_order=sort_keys["source"],
            )
            .sort_values(
                ["_room_order", "_receiver_order", "_source_order"],
                kind="stable",
                na_position="last",
            )
            .drop(columns=["_room_order", "_receiver_order", "_source_order"])
            .reset_index(drop=True)
        )

        # Keep metadata columns in a predictable order.
        rt_columns = [col for col in data_df.columns if col.startswith("rt60_")]
        ordered_columns = [
            "Name", "room_idx", "position_id", "sample_rate", "RT", "c50",
            "x_room", "y_room", "z_room",
            "x_source", "y_source", "z_source",
            "x_mic", "y_mic", "z_mic",
            "mic_orientation_az_deg",
            "accdoa_x", "accdoa_y", "accdoa_z",
            "az_deg", "el_deg",
            *rt_columns,
            "status",
        ]
        data_df = data_df[[col for col in ordered_columns if col in data_df.columns]]

        task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
        data_output_path = os.path.join(output_path, f"Generated_HOA_SRIR_data_task_{task_id}.csv")
        data_df.to_csv(data_output_path, index=False)
        logging.info(f"Successfully saved metadata for {len(successful_results)} RIRs to {data_output_path}")
    else:
        logging.warning("No successful RIR simulations to save metadata for.")

    # Save error log to a separate CSV
    if error_results:
        error_df = pd.DataFrame(error_results)
        error_log_path = os.path.join(output_path, "rir_error_log.csv")
        error_df.to_csv(error_log_path, index=False)
        logging.warning(f"Encountered {len(error_results)} errors during RIR generation. "
                        f"Error details saved to {error_log_path}")
    else:
        logging.info("No errors reported during RIR generation.")

    logging.info("RIR generation process completed.")

    # Print the summary and performance information to stderr
    print(file=sys.stderr)
    rusage_s = resource.getrusage(resource.RUSAGE_SELF)
    rusage_c = resource.getrusage(resource.RUSAGE_CHILDREN)
    print(f'Walltime {time.time() - start:.2f} s', file=sys.stderr)
    print(f'User time: {rusage_s.ru_utime + rusage_c.ru_utime:.2f} s ({rusage_s.ru_utime:.2f} + {rusage_c.ru_utime:.2f})', file=sys.stderr)
    print(f'System time: {rusage_s.ru_stime + rusage_c.ru_stime:.2f} s ({rusage_s.ru_stime:.2f} + {rusage_c.ru_stime:.2f})', file=sys.stderr)
    print(f'MaxRSS: {(rusage_s.ru_maxrss + rusage_c.ru_maxrss)/2**20:.3f} GiB ({rusage_s.ru_maxrss/2**20:.3f} + {rusage_c.ru_maxrss/2**20:.3f})', file=sys.stderr)


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate Room Impulse Responses (RIRs) using pyroomacoustics.")

    # Add arguments for each parameter
    parser.add_argument(
        "--output_path",
        type=str,
        default="./Generated_HOA_IRs/", # Default value
        help="Directory to save one pickle per room and metadata CSV files."
    )
    parser.add_argument(
        "--num_rooms",
        type=int,
        default=2, # Default value
        help="Number of unique room geometries to simulate."
    )
    parser.add_argument(
        "--num_positions",
        type=int,
        default=10, # Default value
        help="Number of different source/mic positions within each room geometry."
    )
    parser.add_argument(
        "--ambi_order",
        type=int,
        default=3, # Default value (changed from 4 in your example to match previous code)
        help="The Ambisonic order for the microphone array (e.g., 0, 1, 2, 3, ...)."
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000, # Default value
        help="The sample rate for the simulations (e.g., 16000, 48000)."
    )
    parser.add_argument(
        "--num_sources_per_receiver",
        type=int,
        default=3,
        help="Number of random source positions per receiver position."
    )
    parser.add_argument(
        "--array_type",
        choices=("ambisonic", "em32"),
        default="ambisonic",
        help="Use the ideal HOA array or the measured Eigenmike EM32 array."
    )
    parser.add_argument(
        "--em32_npz_path",
        type=str,
        default="Eigenmike_em32_IRs.npz",
        help="Path to Eigenmike_em32_IRs.npz (required when --array_type em32)."
    )
    parser.add_argument(
        "--em32_radius",
        type=float,
        default=0.042,
        help="Eigenmike capsule radius in metres."
    )
    parser.add_argument(
        "--amp_threshold",
        type=float,
        default=15.0,
        help="Maximum regularized SHT-filter amplification in dB."
    )

    parser.add_argument(
        "--em32_filter_length",
        type=int,
        default=1024,
        help="Even FIR/FFT length for the theoretical rigid-sphere EM32-to-SH encoder."
    )


    parser.add_argument(
        "--max_order",
        type=int,
        default=12,
        help="Maximum image-source order used for the full ISM room.",
    )
    parser.add_argument(
        "--export_mode",
        choices=("ism", "ism_late", "direct_late"),
        default="ism_late",
        help=(
            "RIR to export: 'ism' exports only the ISM RIR; 'ism_late' "
            "exports ISM early + stochastic late reverb (80 ms mixing); "
            "'direct_late' exports direct sound + stochastic late reverb "
            "estimated from the full ISM RIR (3.5 ms mixing)."
        ),
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(
        output_path=args.output_path,
        num_rooms=args.num_rooms,
        num_positions=args.num_positions,
        ambi_order=args.ambi_order,
        sample_rate=args.sample_rate,
        num_sources_per_receiver=args.num_sources_per_receiver,
        array_type=args.array_type,
        em32_npz_path=args.em32_npz_path,
        em32_radius=args.em32_radius,
        amp_threshold=args.amp_threshold,
        em32_filter_length=args.em32_filter_length,
        max_order=args.max_order,
        export_mode=args.export_mode,
    )

