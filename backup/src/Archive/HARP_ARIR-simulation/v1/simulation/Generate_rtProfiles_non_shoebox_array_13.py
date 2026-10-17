import numpy as np
import pandas as pd
from scipy.io import wavfile
import pyroomacoustics as pra
from pyroomacoustics.directivities import MeasuredDirectivityFile, Rotation3D
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from SphericalHarmonic_2 import HOA_array, SphericalHarmonicDirectivity
import os
import logging
import sys
from typing import Union
import argparse
import resource
import time
import pyfar as pf

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

def get_random_dimensions(x_range=(1.5, 20, 0.5), y_range=(1.5, 20, 0.5), z_range=(2.5, 8, 0.5),) -> tuple:
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
    src = np.array(src).reshape(3)
    mic = np.array(mic).reshape(3)
    v = src - mic
    return v / np.linalg.norm(v)

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


def create_ambisonic_array(order_of_ambisonics: int, sample_rate: int):
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
        samples=samples, radius=1, n_order=order
    )

    # Scale positions by the actual mic radius
    mic_positions = mic_positions * mic_radius

    microphone_directivities = []
    for i in range(samples):
        orientation = orientations[i]
        # Spherical Harmonic Directivity requires n and m parameters
        directivity = SphericalHarmonicDirectivity(
            orientation, n=degrees[i][0], m=degrees[i][1]
        )
        microphone_directivities.append(directivity)

    return mic_positions.T, sample_rate, microphone_directivities


def create_em32_array(sample_rate: int):
    em32 = MeasuredDirectivityFile(
        path="/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/HARP-main/v1/sh_processing/eigenmike32_simulated.sofa",
        fs=sample_rate,
        #interp_order=12,
        #interp_n_points=1000,
    )

    orientation = Rotation3D([0, 0, 0], "zyx", degrees=True)

    directivities = []
    for i in range(32):
        d = em32.get_mic_directivity(i, orientation=orientation)
        directivities.append(d)

    R = 0.042
    mic_dirs_deg = np.array([
        [0, 21], [32, 0], [0, -21], [328, 0],
        [0, 58], [45, 35], [69, 0], [45, -35],
        [0, -58], [315, -35], [291, 0], [315, 35],
        [91, 69], [90, 32], [90, -31], [89, -69],
        [180, 21], [212, 0], [180, -21], [148, 0],
        [180, 58], [225, 35], [249, 0], [225, -35],
        [180, -58], [135, -35], [111, 0], [135, 35],
        [269, 69], [270, 32], [270, -32], [271, -69]
    ])

    az = np.deg2rad(mic_dirs_deg[:, 0])
    el = np.deg2rad(mic_dirs_deg[:, 1])
    colat = np.pi / 2 - el

    mic_positions = np.vstack([
        R * np.sin(colat) * np.cos(az),
        R * np.sin(colat) * np.sin(az),
        R * np.cos(colat)])
    
    return mic_positions, sample_rate, directivities


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

        if volume:
            mixingTime_samp = int(getMixingTimeEstimateFromVolume(volume) * fs)
        else:
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


def addStochasticLateReverbPyfar(
        srir,
        rt,
        fs,
        mixingTime,
        volume,
        centerFreqs=None,
        num_fractions=1,
        filter_order=14):
    """
    Add a diffuse stochastic late reverberant tail using pyfar octave filters.

    Parameters
    ----------
    srir : np.ndarray
        Shape: samples x channels
    rt : np.ndarray
        RT60 values per octave band
    fs : int
        Sampling rate
    mixingTime : float
        Mixing time in seconds, used if volume is None
    volume : float or None
        Room volume in m^3. If given, mixing time is estimated from volume.
    centerFreqs : np.ndarray or None
        Center frequencies corresponding to rt.
    """

    srir = srir.T
    irLength_samp = srir.shape[0]
    numNoiseSources = srir.shape[1]

    rt = np.asarray(rt).flatten()

    if centerFreqs is not None:
        centerFreqs = np.asarray(centerFreqs).flatten()

        valid = centerFreqs <= fs / 2

        rt = rt[valid]
        centerFreqs = centerFreqs[valid]

        frequency_range = (float(centerFreqs[0]), float(centerFreqs[-1]))
    else:
        frequency_range = (20.0, fs / 2)

    noise = np.random.normal(
        0, 1, size=(irLength_samp, numNoiseSources)
    )

    tax = np.arange(irLength_samp)

    if volume is not None:
        mixingTime_samp = int(getMixingTimeEstimateFromVolume(volume) * fs)
    else:
        mixingTime_samp = int(mixingTime * fs)

    mixingTime_samp = min(mixingTime_samp, irLength_samp)

    fadeLength_samp = int(15.0 / 1000.0 * fs)

    if irLength_samp - mixingTime_samp - fadeLength_samp > 0:
        window = np.hstack((
            np.zeros(mixingTime_samp),
            np.sin(np.arange(fadeLength_samp) / fadeLength_samp * np.pi / 2) ** 2,
            np.ones(irLength_samp - mixingTime_samp - fadeLength_samp)
        ))
    else:
        window = np.ones(irLength_samp)
        print("no smooth fade-in possible - response too short")

    # pyfar expects channels x samples
    noise_signal = pf.Signal(noise.T, fs)

    noise_bands_signal = pf.dsp.filter.fractional_octave_bands(
        noise_signal,
        num_fractions=num_fractions,
        frequency_range=frequency_range,
        order=filter_order,
    )

    noise_bands = noise_bands_signal.time

    # Expected pyfar shape: bands x channels x samples
    # Convert to samples x channels x bands
    noisePerSourceBand = np.moveaxis(noise_bands, -1, 0)

    if noisePerSourceBand.shape[1] != numNoiseSources:
        noisePerSourceBand = np.swapaxes(noisePerSourceBand, 1, 2)

    numFilterBands = noisePerSourceBand.shape[2]

    if len(rt) != numFilterBands:
        raise ValueError(
            f"RT vector has {len(rt)} bands, but pyfar produced "
            f"{numFilterBands} bands. "
            f"rt={rt}, frequency_range={frequency_range}, fs={fs}"
        )

    timeConstants = rt * fs / 6.9078

    for iBand in range(numFilterBands):
        envelope = np.exp(-tax / timeConstants[iBand])
        noisePerSourceBand[:, :, iBand] *= envelope[:, np.newaxis]

    noisePerSource = np.sum(noisePerSourceBand, axis=2)

    rmsIs = np.sqrt(np.sum(srir[mixingTime_samp:, :] ** 2, axis=0))
    rmsNoise = np.sqrt(np.sum(noisePerSource[mixingTime_samp:, :] ** 2, axis=0))

    scale = np.divide(
        rmsIs,
        rmsNoise,
        out=np.zeros_like(rmsIs),
        where=rmsNoise > 0
    )

    noiseLate = noisePerSource * scale[np.newaxis, :]

    # Separate components
    rir_early = srir * (1 - window)[:, np.newaxis]
    late_reverb = noiseLate * window[:, np.newaxis]

    # Add them together after separating them
    rir = rir_early + late_reverb

    return rir.T, rir_early.T, late_reverb.T


def simulate_room(room_idx: int, pos_idx: int, output_path: str, ambisonic_microphone, physical_params: list) -> Union[dict, None]:
    """
    Simulates a single room impulse response (RIR) for a given room configuration
    and source/microphone position using pyroomacoustics. Saves the RIR as a WAV file
    and returns key acoustic parameters. Includes error handling.

    Args:
        room_idx: Index of the current room simulation.
        pos_idx: Index of the current source/microphone position within the room.
        output_path: Directory to save the generated RIR WAV file.
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
    room_name = f"R{room_idx}_P{pos_idx}"
    rir_filename = f"{room_name}_IR.wav"
    output_filepath = os.path.join(output_path, rir_filename)

    # Extract physical parameters with clear variable names
    Lx, Ly, Lz, Sx, Sy, Sz, Rx, Ry, Rz, rt60, e_absorption_rest, e_absorption_selected_wall, centerFreqs, idxDifferentWall, floor_corners, volume = physical_params

    logging.info(f"Starting simulation for {room_name} with params: "
                 f"Dims=({Lx[0]:.2f}, {Ly[0]:.2f}, {Lz[0]:.2f}), "
                 f"Src=({Sx[0]:.2f}, {Sy[0]:.2f}, {Sz[0]:.2f}), "
                 f"Mic=({Rx[0]:.2f}, {Ry[0]:.2f}, {Rz[0]:.2f})")

    try:
        # Ensure parameters are simple floats for pyroomacoustics
        room_dims = [Lx[0], Ly[0], Lz[0]]
        source_pos = [Sx[0], Sy[0], Sz[0]]
        mic_center_pos = [Rx[0], Ry[0], Rz[0]]

        material_properties = [ {'description' : 'uniform_material', 'coeffs' : e_absorption_rest, 'center_freqs' : centerFreqs } for iWall in range(6)]
        material_properties[idxDifferentWall] =  {'description' : 'uniform_material', 'coeffs' : e_absorption_selected_wall, 'center_freqs' : centerFreqs}

        # create wall materials
        materials_side_walls = pra.make_materials(*[(material_properties[i],) for i in range(4)])
        # create ceiling and floor material ceiling and floor
        me = pra.make_materials(floor=(material_properties[4],), ceiling=(material_properties[5],))

        # create the room again, with materals

        wall_keys = ["west", "east", "south", "north", "floor", "ceiling"]
        materials_dict = {wall_keys[i]: (material_properties[i],) for i in range(6)}
        room = pra.ShoeBox(p=room_dims,materials=pra.make_materials(**materials_dict), fs=ambisonic_microphone[1], max_order=2)
        room_direct = pra.ShoeBox(p=room_dims,materials=pra.make_materials(**materials_dict), fs=ambisonic_microphone[1], max_order=0)

        #room = pra.Room.from_corners(floor_corners, max_order = 20, materials=materials_side_walls, fs=ambisonic_microphone[1])
        #room.extrude(room_dims[2], materials = me)

        # create the room again, with materals
        room_ds = pra.Room.from_corners(floor_corners,
                                     max_order = 0, materials=materials_side_walls, fs=ambisonic_microphone[1])
        room_ds.extrude(room_dims[2], materials = me)

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


        # ------------------
        # Direct sound
        # ------------------
        # Add source and microphone array
        room_direct.add_source(source_pos)
        room_direct.add_microphone_array(
            pra.MicrophoneArray(
                ambisonic_microphone[0] + np.array(mic_center_pos)[:, None], # Adjust mic positions relative to center
                ambisonic_microphone[1], # Sample rate
                ambisonic_microphone[2], # Directivities
            )
        )

        # Compute RIR using Image Source Model
        room_direct.image_source_model()
        room_direct.compute_rir()

        # Stack and pad RIRs
        RIR_direct = np.stack(
            [np.pad(r[0], (0, target_len - r[0].shape[0])) for r in room_direct.rir],
            axis=0, # Stack along a new dimension (channels)
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
                "Name": rir_filename,
                "RT": np.nan, # Use NaN for undefined values
                "c50": np.nan,
                "x_room": Lx[0], "y_room": Ly[0], "z_room": Lz[0],
                "x_source": Sx[0], "y_source": Sy[0], "z_source": Sz[0],
                "x_mic": Rx[0], "y_mic": Ry[0], "z_mic": Rz[0],
                "status": "Warning: RIR all zeros"
             }
        '''

        # ------------------------------------------------------------
        # Add stochastic late reverberation
        # ------------------------------------------------------------
        _ , RIR_early, RIR_late = addStochasticLateReverbPyfar(
            srir=RIR,
            rt=rt60,
            fs=room.fs,
            mixingTime=0.08,
            volume=volume,
            centerFreqs=centerFreqs,
            num_fractions=1,
            filter_order=14
        )

        RIR = RIR_early + RIR_late

        # Pad final RIR to the next full second
        current_len = RIR.shape[1]
        target_len = int(np.ceil(current_len / room.fs) * room.fs)

        if target_len > current_len:
            RIR = np.pad(
                RIR,
                pad_width=((0, 0), (0, target_len - current_len)),
                mode="constant"
            )

        # Save the RIR to a WAV file
        # pyroomacoustics RIRs are typically float. Convert to float32.
        # Transpose RIR to be (samples, channels) as expected by wavfile.write
        wavfile.write(output_filepath, room.fs, RIR.astype(np.float32).T)

        # Calculate Acoustic Parameters (RT60 and C50)
        # Calculate RT60 from the first channel (often representative)
        # Add a check if the RIR is long enough for RT60 calculation
        rt60 = np.nan # Default to NaN
        if len(RIR[0]) > room.fs * 0.1: # Ensure at least 100ms for decay calculation
             rt60 = pra.experimental.rt60.measure_rt60(RIR[0], fs=room.fs, decay_db=30)
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


        logging.info(f"Successfully simulated {room_name}. RT60: {rt60:.2f}, C50: {c50:.2f}")

        # Return metadata and results
        return {
            "Name": rir_filename,
            "RT": rt60,
            "c50": c50,
            "x_room": Lx[0], "y_room": Ly[0], "z_room": Lz[0],
            "x_source": Sx[0], "y_source": Sy[0], "z_source": Sz[0],
            "x_mic": Rx[0], "y_mic": Ry[0], "z_mic": Rz[0],
            "status": "Success"
        }

    except ValueError as ve:
        error_msg = f"ValueError for {room_name}: {ve}"
        logging.error(error_msg)
        return {
            "Name": rir_filename,
            "status": "Error: ValueError",
            "error_message": str(ve),
            "x_room": Lx[0], "y_room": Ly[0], "z_room": Lz[0],
            "x_source": Sx[0] if 'Sx' in locals() else np.nan, # Use locals() to check if variable exists
            "y_source": Sy[0] if 'Sy' in locals() else np.nan,
            "z_source": Sz[0] if 'Sz' in locals() else np.nan,
            "x_mic": Rx[0] if 'Rx' in locals() else np.nan,
            "y_mic": Ry[0] if 'Ry' in locals() else np.nan,
            "z_mic": Rz[0] if 'Rz' in locals() else np.nan,
        }
    except Exception as e:
        # Catch any other exceptions
        error_msg = f"An unexpected error occurred during simulation for {room_name}: {e}"
        logging.error(error_msg, exc_info=True) # Log traceback for unexpected errors
        return {
            "Name": rir_filename,
            "status": "Error: Unexpected Exception",
            "error_message": str(e),
            "x_room": Lx[0], "y_room": Ly[0], "z_room": Lz[0],
             "x_source": Sx[0] if 'Sx' in locals() else np.nan,
            "y_source": Sy[0] if 'Sy' in locals() else np.nan,
            "z_source": Sz[0] if 'Sz' in locals() else np.nan,
            "x_mic": Rx[0] if 'Rx' in locals() else np.nan,
            "y_mic": Ry[0] if 'Ry' in locals() else np.nan,
            "z_mic": Rz[0] if 'Rz' in locals() else np.nan,
        }


def main(output_path: str, num_rooms: int, num_positions: int, ambi_order: int = 3, sample_rate: int = 16000, num_sources_per_receiver: int = 3):
    """
    Main function to generate a dataset of RIRs with varying room acoustics
    and source/microphone positions.
    Args:
        output_path: Directory to save the generated RIR WAV files and data CSV.
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

    # Create the Ambisonic microphone array definition once
    try:
        ambisonic_microphone = create_ambisonic_array(
            order_of_ambisonics=ambi_order, sample_rate=sample_rate
        )
        logging.info(f"Ambisonic array created with order {ambi_order} "
                     f"({(ambi_order + 1)**2} channels).")
    except Exception as e:
        logging.critical(f"Failed to create Ambisonic microphone array: {e}")
        return # Exit if mic array creation fails

    all_results = [] # Collect results (success or error details)

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
                            room_dims, fs=ambisonic_microphone[1],
                            use_rand_ism = False, max_order=0
                        )
                        floor_corners_shoebox = roomShoebox.walls[4].corners # corners of floor

                        # and displace corners before creating 3D room from it
                        floor_corners = floor_corners_shoebox[:2, :] + np.random.rand(2, 4) * max_displacement_m

                        # now create a dummy room with those dimensions
                        room_test = pra.Room.from_corners(floor_corners,
                                                max_order = 0, fs=ambisonic_microphone[1])

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
                    min_dist_from_receiver = 0.5 # in meters
                    max_dist_from_receiver = 5.0 # in meters
                    
                    tries_receiver += 1
                    _, _, _, Rx, Ry, Rz = get_random_positions(Lx[0], Ly[0], Lz[0])
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
                        Sx, Sy, Sz, _, _, _ = get_random_positions(Lx[0], Ly[0], Lz[0])
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

                    for l, source in enumerate(accepted_sources_per_receiver[k]):
                        Sx = np.array([source[0]])
                        Sy = np.array([source[1]])
                        Sz = np.array([source[2]])

                        physical_params = [Lx, Ly, Lz, Sx, Sy, Sz, Rx, Ry, Rz, rt60, e_absorption_rest, e_absorption_selected_wall, centerFreqs, idxDifferentWall, floor_corners, volume]

                        futures.append(
                            executor.submit(
                                simulate_room,
                                i,
                                f"{k}_S{l}",
                                output_path,
                                ambisonic_microphone,
                                physical_params,
                            )
                        )
                i += 1
                pbar.update(1)

        # Process the results as they complete
        for future in tqdm(futures, desc="Processing Results"):
            # The result from simulate_room is either a success dict or an error dict
            result = future.result()
            all_results.append(result)


    # Separate successful results from errors for the main data CSV
    successful_results = [res for res in all_results if res and res.get("status") == "Success"]
    error_results = [res for res in all_results if res and res.get("status") != "Success"]

    # Save successful results to the main data CSV
    if successful_results:
        data_df = pd.DataFrame(successful_results)
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
        help="Directory to save the generated RIR WAV files and data CSV."
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


    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(
        output_path=args.output_path,
        num_rooms=args.num_rooms,
        num_positions=args.num_positions,
        ambi_order=args.ambi_order,
        sample_rate=args.sample_rate,
        num_sources_per_receiver=args.num_sources_per_receiver
    )

