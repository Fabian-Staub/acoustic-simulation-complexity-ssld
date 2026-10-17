import numpy as np
import pandas as pd
from scipy.io import wavfile
import pyroomacoustics as pra
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from SphericalHarmonic import HOA_array, SphericalHarmonicDirectivity
import os
import logging
import sys
from typing import Union
import argparse

# Configure logging
# Create logs directory if it doesn't exist
logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)
log_file_path = os.path.join(logs_dir, "rir_generation.log")

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

def get_random_dimensions(min_dim: int = 4, max_dim: int = 15, step: int = 1) -> tuple:
    """
    Generate random dimensions for the room.

    Args:
        min_dim: Minimum dimension size (inclusive).
        max_dim: Maximum dimension size (exclusive).
        step: Step size for dimension choices.

    Returns:
        tuple: Random dimensions (Lx, Ly, Lz) as numpy arrays (size 1).
    """
    dimensions_range = np.arange(min_dim, max_dim, step)
    # Ensure dimensions are positive
    return tuple(np.random.choice(dimensions_range, size=1) for _ in range(3))


def get_random_positions(Lx: float, Ly: float, Lz: float, min_dist_from_wall: float = 0.5, step: float = 0.5) -> tuple:
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
    z_pos_range = np.arange(min_dist_from_wall, Lz - min_dist_from_wall, step)

    # Check if valid positions are possible
    if len(x_pos_range) == 0 or len(y_pos_range) == 0 or len(z_pos_range) == 0:
        raise ValueError(f"Room dimensions ({Lx}, {Ly}, {Lz}) too small for minimum distance {min_dist_from_wall}.")

    Sx, Sy, Sz = [np.random.choice(pos_range, size=1) for pos_range in [x_pos_range, y_pos_range, z_pos_range]]
    Rx, Ry, Rz = [np.random.choice(pos_range, size=1) for pos_range in [x_pos_range, y_pos_range, z_pos_range]]

    # Ensure source and receiver are not at the exact same position
    while np.allclose([Rx, Ry, Rz], [Sx, Sy, Sz]):
         Rx, Ry, Rz = [np.random.choice(pos_range, size=1) for pos_range in [x_pos_range, y_pos_range, z_pos_range]]

    return Sx, Sy, Sz, Rx, Ry, Rz


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
    Lx, Ly, Lz, materials, Sx, Sy, Sz, Rx, Ry, Rz = physical_params

    logging.info(f"Starting simulation for {room_name} with params: "
                 f"Dims=({Lx[0]:.2f}, {Ly[0]:.2f}, {Lz[0]:.2f}), "
                 f"Src=({Sx[0]:.2f}, {Sy[0]:.2f}, {Sz[0]:.2f}), "
                 f"Mic=({Rx[0]:.2f}, {Ry[0]:.2f}, {Rz[0]:.2f}), "
                 f"Materials={materials}")

    try:
        # Ensure parameters are simple floats for pyroomacoustics
        room_dims = [Lx[0], Ly[0], Lz[0]]
        source_pos = [Sx[0], Sy[0], Sz[0]]
        mic_center_pos = [Rx[0], Ry[0], Rz[0]]

        # Create the ShoeBox room
        room = pra.ShoeBox(
            p=room_dims,
            materials=pra.make_materials(**materials),
            fs=ambisonic_microphone[1], # Use sample rate from mic array
            max_order=30, # Keep max_order reasonable to avoid excessive computation/errors
        )

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


def main(output_path: str, num_rooms: int, num_positions: int, ambi_order: int = 3, sample_rate: int = 16000):
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
    with ProcessPoolExecutor() as executor:
        futures = []
        # Outer loop for different room geometries
        for i in tqdm(range(num_rooms), desc="Generating Rooms"):
            try:
                # Generate random room dimensions and materials for the current room
                Lx, Ly, Lz = get_random_dimensions()
                mat = get_random_materials()

                # Inner loop for different source/mic positions within the current room
                for j in range(num_positions):
                    try:
                        # Generate random source and mic positions
                        Sx, Sy, Sz, Rx, Ry, Rz = get_random_positions(Lx[0], Ly[0], Lz[0]) # Pass float values

                        # Prepare parameters for the simulation worker
                        physical_params = [Lx, Ly, Lz, mat, Sx, Sy, Sz, Rx, Ry, Rz]

                        # Submit the simulation task to the executor
                        futures.append(
                            executor.submit(
                                simulate_room,
                                i,
                                j,
                                output_path,
                                ambisonic_microphone,
                                physical_params,
                            )
                        )
                    except ValueError as ve:
                         # Catch specific errors from position generation
                         error_msg = f"ValueError during position generation for Room {i}, Pos {j}: {ve}"
                         logging.error(error_msg)
                         # Log this specific failure without submitting a simulation task
                         all_results.append({
                             "Name": f"R{i}_P{j}_IR.wav", # Still log the expected filename
                             "status": "Error: Position Generation Failed",
                             "error_message": str(ve),
                             "x_room": Lx[0], "y_room": Ly[0], "z_room": Lz[0],
                             "x_source": np.nan, "y_source": np.nan, "z_source": np.nan,
                             "x_mic": np.nan, "y_mic": np.nan, "z_mic": np.nan,
                         })
                    except Exception as e:
                         # Catch any other exceptions during position generation
                         error_msg = f"An unexpected error during position generation for Room {i}, Pos {j}: {e}"
                         logging.error(error_msg, exc_info=True)
                         all_results.append({
                             "Name": f"R{i}_P{j}_IR.wav",
                             "status": "Error: Unexpected Position Generation Exception",
                             "error_message": str(e),
                              "x_room": Lx[0], "y_room": Ly[0], "z_room": Lz[0],
                             "x_source": np.nan, "y_source": np.nan, "z_source": np.nan,
                             "x_mic": np.nan, "y_mic": np.nan, "z_mic": np.nan,
                         })

            except Exception as e:
                 # Catch any exceptions during room dimension or material generation
                 error_msg = f"An unexpected error during room generation (Room {i}): {e}"
                 logging.error(error_msg, exc_info=True)
                 # Log this error as it affects all positions for this room
                 for j in range(num_positions):
                     all_results.append({
                             "Name": f"R{i}_P{j}_IR.wav",
                             "status": "Error: Unexpected Room Generation Exception",
                             "error_message": str(e),
                             "x_room": np.nan, "y_room": np.nan, "z_room": np.nan,
                             "x_source": np.nan, "y_source": np.nan, "z_source": np.nan,
                             "x_mic": np.nan, "y_mic": np.nan, "z_mic": np.nan,
                         })


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
        data_output_path = os.path.join(output_path, "Generated_HOA_SRIR_data.csv")
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

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(
        output_path=args.output_path,
        num_rooms=args.num_rooms,
        num_positions=args.num_positions,
        ambi_order=args.ambi_order,
        sample_rate=args.sample_rate
    )
