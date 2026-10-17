import numpy as np
from scipy.io import wavfile
import pyroomacoustics as pra
from shapely.geometry import Polygon, Point, LineString 
import logging


# --- Global Constants ---

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

# Predefined complex polygon shapes
POLYGON_SHAPES = {
    'L': np.array([
        [0, 0], [4, 0], [4, 2], [2, 2], [2, 4], [0, 4],
    ]).T,
    'T': np.array([
        [0, 0], [6, 0], [6, 2], [4, 2], [4, 4], [2, 4], [2, 2], [0, 2]
    ]).T,
    'H': np.array([
        [0, 0], [0, 5], [1, 5], [1, 2], [2, 2], [2, 5], [3, 5], [3, 0], [2, 0], [2, 1], [1, 1], [1, 0]
    ]).T,
    'C': np.array([
        [0, 0], [4, 0], [4, 1], [1, 1], [1, 3], [4, 3], [4, 4], [0, 4]
    ]).T
}

# Default distribution probabilities for room types
ROOM_TYPE_PROBABILITIES = [.3, .2, .2, .1, .2] # Should match order ['shoebox', 'L', 'T', 'H', 'C']
ROOM_TYPES = ['shoebox', 'L', 'T', 'H', 'C']


'''# Mic/Source Types (used in generate_rir_for_all_combinations_together, but only point source seems implemented)
# mic_types = ['em_32'] #'kemar', # Kept for context but not actively used in add() calls
# source_types = ['point'] #'Genelec_8020', 'HATS_4128C', 'Lambda_labs_CX-1A' # Kept for context'''

# --- Utility Functions ---


def get_random_shoebox_dimensions(min_dim: int = 2, max_dim: int = 8, step: int = 1) -> tuple[float, float, float]:
    """
    Generate random dimensions for a shoebox room.

    Args:
        min_dim: Minimum dimension size (inclusive).
        max_dim: Maximum dimension size (exclusive).
        step: Step size for dimension choices.

    Returns:
        tuple: Random dimensions (Lx, Ly, Lz) as floats.
    """
    dimensions_range = np.arange(min_dim, max_dim, step)
    # Ensure dimensions are positive and return as floats
    return tuple(float(np.random.choice(dimensions_range)) for _ in range(3))


def get_random_shoebox_positions(Lx: float, Ly: float, Lz: float, min_dist_from_wall: float = 0.5, step: float = 0.5) -> tuple[float, float, float, float, float, float]:
    """
    Generate random positions for the source and microphone in a shoebox room.

    Args:
        Lx: Room length along x-axis.
        Ly: Room length along y-axis.
        Lz: Room length along z-axis.
        min_dist_from_wall: Minimum distance of source/mic from any wall.
        step: Step size for position choices.

    Returns:
        tuple: Random source and receiver positions (Sx, Sy, Sz, Rx, Ry, Rz) as floats.
    """
    # Determine the valid range for positions
    x_pos_range = np.arange(min_dist_from_wall, Lx - min_dist_from_wall, step)
    y_pos_range = np.arange(min_dist_from_wall, Ly - min_dist_from_wall, step)
    z_pos_range = np.arange(min_dist_from_wall, Lz - min_dist_from_wall, step)

    # Check if valid positions are possible
    if len(x_pos_range) == 0 or len(y_pos_range) == 0 or len(z_pos_range) == 0:
        raise ValueError(f"Room dimensions ({Lx}, {Ly}, {Lz}) too small for minimum distance {min_dist_from_wall}.")

    # Generate random positions and return as floats
    Sx, Sy, Sz = (float(np.random.choice(pos_range)) for pos_range in [x_pos_range, y_pos_range, z_pos_range])
    Rx, Ry, Rz = (float(np.random.choice(pos_range)) for pos_range in [x_pos_range, y_pos_range, z_pos_range])

    # Ensure source and receiver are not at the exact same position
    while np.allclose([Rx, Ry, Rz], [Sx, Sy, Sz]):
         Rx, Ry, Rz = (float(np.random.choice(pos_range)) for pos_range in [x_pos_range, y_pos_range, z_pos_range])

    return Sx, Sy, Sz, Rx, Ry, Rz


def get_random_materials() -> dict:
    """
    Randomly choose materials for the room surfaces from predefined lists.

    Returns:
        dict: Material dictionary with random choices for walls, ceiling, and floor.
    """
    return {
        'east': np.random.choice(WALLS),
        'west': np.random.choice(WALLS),
        'north': np.random.choice(WALLS),
        'south': np.random.choice(WALLS),
        'ceiling': np.random.choice(CEIL),
        'floor': np.random.choice(FLOOR),
    }

def random_point_in_polygon(polygon_vertices_2d: np.ndarray, height: float, min_dist_from_wall_2d: float = 0.1, min_height: float = 0.5) -> np.ndarray:
    """
    Generates a random 3D point inside a given 2D polygon boundary,
    ensuring it's within a height range and a minimum distance from walls (in 2D).

    Args:
        polygon_vertices_2d (np.ndarray): 2D array of polygon vertices (shape: 2 x N).
        height (float): The total height of the 3D room.
        min_dist_from_wall_2d (float): Minimum 2D distance from the polygon boundary.
        min_height (float): Minimum height for the Z coordinate.

    Returns:
        np.ndarray: A random point inside the 3D room as [x, y, z].
    """
    poly = Polygon(polygon_vertices_2d.T)

    # Create a buffer polygon shrunk by min_dist_from_wall_2d
    # Handle potential errors if the buffer makes the polygon invalid or empty
    try:
        buffered_poly = poly.buffer(-min_dist_from_wall_2d)
    except Exception as e:
        logging.warning(f"Failed to create buffered polygon with distance {min_dist_from_wall_2d}: {e}")
        buffered_poly = poly # Fallback to original polygon

    if buffered_poly.is_empty:
         logging.warning(f"Buffered polygon is empty for distance {min_dist_from_wall_2d}. Using original polygon bounds.")
         target_poly = poly
    else:
        target_poly = buffered_poly

    min_x, min_y, max_x, max_y = target_poly.bounds
    min_z = min_height
    max_z = height - min_height # Ensure distance from ceiling too

    # Check if valid Z range is possible
    if max_z <= min_z:
         raise ValueError(f"Room height ({height}) too small for minimum vertical distance {min_height}.")


    while True:
        # Generate random point within the bounding box of the target polygon
        x = np.random.uniform(min_x, max_x)
        y = np.random.uniform(min_y, max_y)
        z = np.random.uniform(min_z, max_z)
        point_2d = Point(x, y)

        # Check if the 2D point is inside the target polygon (shrunk by min_dist)
        if target_poly.contains(point_2d):
            return np.array([x, y, z])


def scale_polygon_variable(vertices: np.ndarray, scale_factors: dict) -> np.ndarray:
    """
    Scales a polygon variably, applying different scaling factors for x and y directions.

    Args:
        vertices (np.ndarray): The 2D vertices of the polygon (shape: 2 x N).
        scale_factors (dict): A dictionary with keys 'x' and 'y' for scaling along each axis.

    Returns:
        np.ndarray: Scaled vertices (2 x N).
    """
    scaled_vertices = vertices.copy().astype(np.float64)
    # Scale relative to the origin (0,0)
    scaled_vertices[0, :] *= scale_factors.get('x', 1.0)  # Scale x
    scaled_vertices[1, :] *= scale_factors.get('y', 1.0)  # Scale y

    # Optional: recenter the polygon after scaling if needed, but often not necessary for pyroomacoustics polygons

    return scaled_vertices

def makeCoupledPolygonRoom(fs: int, height: float, shape: str) -> tuple:
    """
    Create parameters for a single complex polygon room, such as L-shaped or T-shaped.

    Parameters:
        fs (int): Sample rate.
        height (float): Height of the room.
        shape (str): The shape key ('L', 'T', 'H', 'C').

    Returns:
        tuple: Contains:
               - wall_materials (pyroomacoustics.Material): Materials for vertical walls.
               - vertical_materials (pyroomacoustics.Material): Materials for floor/ceiling.
               - corners (np.ndarray): 2D vertices of the base polygon (shape: 2 x N).
               - height (float): The height of the room.
    Raises:
        ValueError: If an invalid shape is provided.
    """
    if shape not in POLYGON_SHAPES:
        raise ValueError(f"Unsupported polygon shape: {shape}. Choose from {list(POLYGON_SHAPES.keys())}")

    # Get base corners for the shape and scale them
    corners = POLYGON_SHAPES[shape].copy()
    scale_factors = {
        'x': np.random.uniform(1.5, 4.0),  # Stretch/compress in x direction
        'y': np.random.uniform(1.5, 4.0)  # Stretch/compress in y direction
    }
    corners = scale_polygon_variable(corners, scale_factors)

    # Randomized wall absorption coefficients
    # Ensure enough materials are chosen for the number of wall segments
    num_wall_segments = corners.shape[1]
    if num_wall_segments > len(WALLS):
        logging.warning(f"Not enough wall materials ({len(WALLS)}) for polygon with {num_wall_segments} segments. Reusing materials.")
    wall_absorptions = np.random.choice(WALLS, (num_wall_segments,), replace=True)
    wall_materials = pra.make_materials(*[(mat,) for mat in wall_absorptions])

    # Materials for floor and ceiling
    floor_mat = np.random.choice(FLOOR)
    ceiling_mat = np.random.choice(CEIL)
    vertical_materials = pra.make_materials(floor=floor_mat, ceiling=ceiling_mat)
    debugging_mat = [(mat,) for mat in wall_absorptions]
    debugging_mat.append((floor_mat,))
    debugging_mat.append((ceiling_mat,))
    return wall_materials, vertical_materials, corners, height, debugging_mat

def is_line_segment_clear_2d(point1_2d: tuple[float, float], point2_2d: tuple[float, float], polygon_vertices_2d: np.ndarray) -> bool:
    """
    Checks if the 2D line segment between two points intersects any edge of a 2D polygon.
    Ignores intersections at the endpoints.

    Args:
        point1_2d (tuple[float, float]): The (x, y) coordinates of the first point.
        point2_2d (tuple[float, float]): The (x, y) coordinates of the second point.
        polygon_vertices_2d (np.ndarray): 2D array of polygon vertices (shape: 2 x N).

    Returns:
        bool: True if the line segment does NOT intersect any polygon edge, False otherwise.
    """
    line = LineString([point1_2d, point2_2d])
    num_vertices = polygon_vertices_2d.shape[1]

    for i in range(num_vertices):
        # Get current edge vertices
        v1 = polygon_vertices_2d[:, i]
        v2 = polygon_vertices_2d[:, (i + 1) % num_vertices] # Wrap around for the last edge

        edge = LineString([v1, v2])

        # Check for intersection between the source-mic line and the edge
        # Use .crosses() which specifically checks for interior intersection
        # .intersects() would return True if endpoints are on the boundary
        if line.crosses(edge):
            return False # Intersection found, line of sight is blocked

    return True # No intersections found with any edge


def fibonacci_sphere(samples: int, radius: float) -> np.ndarray:
    """
    Generates points on a sphere using the Fibonacci lattice method.
    This creates a near-uniform distribution of points on a sphere.

    Parameters:
        samples (int): Number of points to generate.
        radius (float): Radius of the sphere.

    Returns:
        np.array: 3D coordinates of the points (shape: samples x 3).
    """
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius_at_y = np.sqrt(1 - y * y)  # radius at y
        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius_at_y
        z = np.sin(theta) * radius_at_y

        points.append([x * radius, y * radius, z * radius])

    return np.array(points)

def EM32_mic_config(num_mics: int = 64, mic_center: list[float] = [0,0,0], fs: int = 16000) -> pra.MicrophoneArray:
    """
    Creates a microphone array configuration based on points on a sphere.
    Defaults to 64 points on a sphere with a radius of 0.042m, centered.

    Args:
        num_mics (int): The number of microphones in the array.
        mic_center (list[float]): The [x, y, z] coordinates of the array center.
        fs (int): The sample rate.

    Returns:
        pra.MicrophoneArray: The pyroomacoustics microphone array object.
    """
    # Generate points on a sphere
    all_positions = fibonacci_sphere(num_mics, radius=.42) # Example radius, adjust if needed

    # Translate points to the microphone center
    all_positions = all_positions + np.array(mic_center)

    # pyroomacoustics expects positions as 3 x N_mics
    return pra.MicrophoneArray(all_positions.T, fs)

def shoebox_to_vertices(x: float, y: float, z: float) -> np.ndarray:
    """
    Converts shoebox dimensions (x, y, z) into 3D vertices representing the corners of the room.

    Parameters:
    - x (float): Length of the room (along x-axis)
    - y (float): Width of the room (along y-axis)
    - z (float): Height of the room (along z-axis)

    Returns:
    - vertices (numpy.ndarray): A (8, 3) array where each row represents a vertex (x, y, z).
    """
    # Define the 8 vertices of a rectangular cuboid (shoebox)
    vertices = np.array([
        [0, 0, 0], [x, 0, 0], [x, y, 0], [0, y, 0],
        [0, 0, z], [x, 0, z], [x, y, z], [0, y, z],
    ])

    return vertices

def polygon_to_3d_vertices(x_coords: list[float], y_coords: list[float], z_height: float) -> np.ndarray:
    """
    Converts 2D polygon (x, y) coordinates and height into 3D vertices.

    Parameters:
    - x_coords (list): List of x-coordinates of the polygon vertices.
    - y_coords (list): List of y-coordinates of the polygon vertices.
    - z_height (float): Height of the room (z-coordinate).

    Returns:
    - vertices (numpy.ndarray): A (2 * len(x_coords), 3) array where each row represents a vertex (x, y, z).
    """
    # Number of vertices in the polygon
    num_vertices = len(x_coords)

    # Create the floor vertices (z=0)
    floor_vertices = np.array(list(zip(x_coords, y_coords, np.zeros(num_vertices))))

    # Create the ceiling vertices (z=z_height)
    ceiling_vertices = np.array(list(zip(x_coords, y_coords, np.full(num_vertices, z_height))))

    # Combine floor and ceiling vertices
    vertices = np.vstack([floor_vertices, ceiling_vertices])

    return vertices


def save_rir_wav(rir_data: np.ndarray, output_filepath: str, fs: int):
    """
    Saves the RIR data to a WAV file, handling padding and normalization.

    Args:
        rir_data (np.ndarray): The RIR data (channels x samples).
        output_filepath (str): The full path to save the WAV file.
        fs (int): The sample rate.
    """
    try:
        # Find maximum length across all channels to pad
        max_len = rir_data.shape[1]
        # Determine target length for padding (pad to the nearest second)
        # Or pad to the max RIR length if you prefer not rounding up
        # target_len = int(np.ceil(max_len / fs) * fs) if max_len > 0 else 0
        target_len = max_len # Let's just use the max length

        # Pad RIRs if necessary (shouldn't be needed if compute_rir output is consistent, but kept for safety)
        if rir_data.shape[1] < target_len:
             padded_rir = np.stack([np.pad(r, (0, target_len - r.shape[0])) for r in rir_data], axis=0)
        else:
             padded_rir = rir_data

        # Normalize RIR by the maximum absolute value across all channels
        max_rir_val = np.max(np.abs(padded_rir))
        if max_rir_val > 0:
            normalized_rir = padded_rir / max_rir_val
        else:
             logging.warning(f"RIR is all zeros or near zero for {output_filepath}. Normalization skipped.")
             normalized_rir = padded_rir # Keep as is (all zeros)

        # pyroomacoustics RIRs are typically float. Convert to float32.
        # Transpose RIR to be (samples, channels) as expected by wavfile.write
        wavfile.write(output_filepath, fs, normalized_rir.astype(np.float32).T)

    except Exception as e:
        raise IOError(f"Failed to save WAV file {output_filepath}: {e}")


def calculate_acoustic_params(rir_channel: np.ndarray, fs: int) -> tuple[float, float]:
    """
    Calculates RT60 and C50 from a single RIR channel.

    Args:
        rir_channel (np.ndarray): A single channel of RIR data.
        fs (int): The sample rate.

    Returns:
        tuple: (RT60, C50) values. Returns np.nan if calculation fails or RIR is too short.
    """
    rt60 = np.nan
    c50 = np.nan

    try:
        # Calculate RT60 (need enough samples for the decay measurement)
        # Decay from 0dB to -60dB requires a sufficiently long RIR
        # Let's calculate decay_db based on RIR length if < 60dB is possible
        decay_db_target = 60
        # A rough estimate of max possible decay in a finite RIR
        max_possible_decay_db = 10 * np.log10(np.max(np.square(np.abs(rir_channel))) / (np.min(np.square(np.abs(rir_channel[np.nonzero(rir_channel)]))) + 1e-10)) # avoid log(0)
        decay_db_actual = min(decay_db_target, max_possible_decay_db)
        decay_db_actual = max(10, decay_db_actual) # Ensure at least 10dB decay is attempted

        if len(rir_channel) > fs * 0.1 and decay_db_actual >= 10: # Ensure RIR is long enough and meaningful decay can be calculated
             rt60 = pra.experimental.rt60.measure_rt60(rir_channel, fs=fs, decay_db=decay_db_actual)
             if rt60 == 0.0: # Sometimes measure_rt60 returns 0 for short RIRs or bad decay
                 rt60 = np.nan
        else:
             logging.warning(f"RIR too short or flat for RT60 calculation ({len(rir_channel)} samples).")

        # Calculate C50 (Clarity Index)
        # Needs at least 50ms of RIR data
        samples_50ms = int(fs * 0.05)
        if len(rir_channel) > samples_50ms:
            early_energy = np.sum(np.square(np.abs(rir_channel[:samples_50ms])))
            late_energy = np.sum(np.square(np.abs(rir_channel[samples_50ms:])))

            if late_energy > 1e-10: # Avoid division by zero or near zero
                 c50 = 10 * np.log10(early_energy / late_energy)
            else:
                 logging.warning("Late energy is zero or near zero for C50 calculation.")
                 c50 = np.inf # Or handle as appropriate for your analysis

        else:
             logging.warning(f"RIR too short for C50 calculation (needs > {samples_50ms} samples).")

    except Exception as e:
        logging.warning(f"Error calculating acoustic parameters: {e}")
        rt60 = np.nan
        c50 = np.nan

    return rt60, c50
