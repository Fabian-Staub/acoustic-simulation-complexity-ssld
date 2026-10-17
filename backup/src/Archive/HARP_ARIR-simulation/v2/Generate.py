# This is version 2
# Instead of ISM + SH_Directivity + Shoebox, we use Ray Tracing + Spherical Array + Complex Geometries here
# Has complex geometries
from v2utils import *
import os
from typing import Union
import argparse 
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
# Configure logging (File only)
logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)
log_file_path = os.path.join(logs_dir, "rir_generation_v2.log") # Use a different log file name

file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

if root_logger.hasHandlers():
    root_logger.handlers.clear()

root_logger.addHandler(file_handler)


def run_ray_tracing_simulation(room: pra.Room, mic_position: list[float], source_position: list[float], fs: int, num_mics: int) -> Union[np.ndarray, None]:
    """
    Adds microphone and source to the room, runs ray tracing and RIR computation.

    Args:
        room (pra.Room): The pyroomacoustics room object.
        mic_position (list[float]): [x, y, z] position of the microphone array center.
        source_position (list[float]): [x, y, z] position of the source.
        fs (int): Sample rate.

    Returns:
        np.ndarray: The computed RIR data (channels x samples) if successful.
        None: If simulation fails.
    """
    try:
        # Add microphone array
        mics = EM32_mic_config(num_mics, mic_position, fs)
        room.add_microphone_array(mics)

        # Add source (Assuming only one source is added per call)
        room.add_source(source_position)

        # Run Ray Tracing
        # Adjust parameters as needed for complexity vs accuracy vs computation time
        room.set_ray_tracing(
            n_rays=10000,        # Number of rays
            energy_thres=1e-9,   # Minimum energy to trace a ray
            time_thres=5.0       # Maximum time to trace a ray (seconds)
        )
        room.ray_tracing()

        # Compute RIR from the ray tracing results
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
        RIR = RIR/ np.max(np.abs(RIR))


        return RIR

    except Exception as e:
        # Log the specific simulation error internally
        logging.error(f"Ray Tracing simulation failed: {e}", exc_info=True)
        return None # Indicate failure to the caller


def simulate_complex_room(room_idx: int, pos_idx: int, output_path: str, fs: int, wall_materials, vertical_materials, corners: np.ndarray, height: float, num_mics: int, regularity: str) -> Union[dict, None]:
    """
    Simulates a single RIR for a complex polygonal room using ray tracing.

    Args:
        room_idx, pos_idx: Indices for logging and naming.
        output_path (str): Directory to save results.
        fs (int): Sample rate.
        wall_materials, vertical_materials: pyroomacoustics Material objects.
        corners (np.ndarray): 2D vertices of the base polygon (shape: 2 x N).
        height (float): Room height.
        regularity (str): Room shape type ('L', 'T', 'H', 'C').

    Returns:
        dict: Simulation results dictionary if successful.
        None: If simulation or processing failed.
    """
    room_name = f"R{room_idx}_P{pos_idx}"
    rir_filename = f"{room_name}_IR.wav"
    output_filepath = os.path.join(output_path, rir_filename) # Save in a folder

    logging.info(f"Starting complex room simulation for {room_name} (Shape: {regularity})")
    
    try:
        # Create the 2D polygonal room and extrude to 3D
        room = pra.Room.from_corners(
            corners,
            fs=fs,
            max_order=0, # Keep max_order low for Ray Tracing
            materials=wall_materials,
            ray_tracing=True # Ensure ray tracing is enabled for this room
        )
        room.extrude(height, materials=vertical_materials)

        # Generate random source and mic positions within the room boundary
        # Retry position generation if they are too close or invalid
        max_pos_attempts = 100
        for attempt in range(max_pos_attempts):
            try:
                x_source, y_source, z_source = random_point_in_polygon(corners, height)
                x_mic, y_mic, z_mic = random_point_in_polygon(corners, height)
                # Check distance between source and mic
                dist_sq = (x_source - x_mic)**2 + (y_source - y_mic)**2 + (z_source - z_mic)**2
                src_pos_2d = (x_source, y_source)
                mic_pos_2d = (x_mic, y_mic)
                line_of_sight_clear = is_line_segment_clear_2d(src_pos_2d, mic_pos_2d, corners)
                if dist_sq > 0.1**2 and line_of_sight_clear:
                    pos_generation_successful = True
                    break
            except ValueError as ve:
                 logging.warning(f"Position generation attempt {attempt+1}/{max_pos_attempts} failed for {room_name}: {ve}")
            except Exception as e:
                 logging.warning(f"Unexpected error during position generation attempt {attempt+1}/{max_pos_attempts} for {room_name}: {e}")

        else: # This else block executes if the loop completes without 'break'
            raise ValueError(f"Failed to generate valid source/mic positions after {max_pos_attempts} attempts for {room_name}.")


        source_position = [x_source, y_source, z_source]
        mic_center = [x_mic, y_mic, z_mic]

        logging.info(f"Positions for {room_name}: Src=({x_source:.2f}, {y_source:.2f}, {z_source:.2f}), Mic=({x_mic:.2f}, {y_mic:.2f}, {z_mic:.2f})")

        # Run the ray tracing simulation
        rir_data = run_ray_tracing_simulation(room, mic_center, source_position, fs, num_mics)

        if rir_data is None:
             # run_ray_tracing_simulation already logged the error
             return {
                "Name": rir_filename,
                "status": "Error: Ray Tracing Failed",
                "error_message": "See log file for details",
                "room_type": regularity,
                "vertices": polygon_to_3d_vertices(corners[0, :].tolist(), corners[1, :].tolist(), height).tolist(),
                "x_source": x_source, "y_source": y_source, "z_source": z_source,
                "x_mic": x_mic, "y_mic": y_mic, "z_mic": z_mic,
             }

        # Save the RIR data
        save_rir_wav(rir_data, output_filepath, fs)

        # Calculate acoustic parameters (e.g., RT60 from the first channel)
        rt60, c50 = calculate_acoustic_params(rir_data[0], fs) # Use the first channel for simplicity

        logging.info(f"Successfully simulated {room_name}. RT60: {rt60:.2f}, C50: {c50:.2f}")


        # Return metadata and results
        return {
            "Name": rir_filename,
            "RT": rt60,
            "c50": c50, # Added C50
            "room_type": regularity,
            "vertices": polygon_to_3d_vertices(corners[0, :].tolist(), corners[1, :].tolist(), height).tolist(),
            "x_source": x_source,
            "y_source": y_source,
            "z_source": z_source,
            "x_mic": x_mic,
            "y_mic": y_mic,
            "z_mic": z_mic,
            "status": "Success"
        }

    except ValueError as ve:
        error_msg = f"ValueError for {room_name}: {ve}"
        logging.error(error_msg)
        return {
            "Name": rir_filename,
            "status": "Error: ValueError",
            "error_message": str(ve),
            "room_type": regularity,
            "vertices": polygon_to_3d_vertices(corners[0, :].tolist(), corners[1, :].tolist(), height).tolist() if 'corners' in locals() and 'height' in locals() else None,
            "x_source": x_source if 'x_source' in locals() else np.nan,
            "y_source": y_source if 'y_source' in locals() else np.nan,
            "z_source": z_source if 'z_source' in locals() else np.nan,
            "x_mic": x_mic if 'x_mic' in locals() else np.nan,
            "y_mic": y_mic if 'y_mic' in locals() else np.nan,
            "z_mic": z_mic if 'z_mic' in locals() else np.nan,
        }
    except Exception as e:
        # Catch any other exceptions
        error_msg = f"An unexpected error occurred during complex room simulation for {room_name}: {e}"
        logging.error(error_msg, exc_info=True)
        return {
            "Name": rir_filename,
            "status": "Error: Unexpected Exception",
            "error_message": str(e),
             "room_type": regularity,
            "vertices": polygon_to_3d_vertices(corners[0, :].tolist(), corners[1, :].tolist(), height).tolist() if 'corners' in locals() and 'height' in locals() else None,
             "x_source": x_source if 'x_source' in locals() else np.nan,
            "y_source": y_source if 'y_source' in locals() else np.nan,
            "z_source": z_source if 'z_source' in locals() else np.nan,
            "x_mic": x_mic if 'x_mic' in locals() else np.nan,
            "y_mic": y_mic if 'y_mic' in locals() else np.nan,
            "z_mic": z_mic if 'z_mic' in locals() else np.nan,
        }


def simulate_shoebox_room(room_idx: int, pos_idx: int, output_path: str, fs: int, mat: dict, x_room: float, y_room: float, z_room: float, num_mics:int, regularity: str = 'shoebox') -> Union[dict, None]:
    """
    Simulates a single RIR for a shoebox room using ray tracing.

    Args:
        room_idx, pos_idx: Indices for logging and naming.
        output_path (str): Directory to save results.
        fs (int): Sample rate.
        mat (dict): Dictionary of material names.
        x_room, y_room, z_room: Room dimensions.
        regularity (str): Room shape type ('shoebox').

    Returns:
        dict: Simulation results dictionary if successful.
        None: If simulation or processing failed.
    """
    room_name = f"R{room_idx}_P{pos_idx}"
    rir_filename = f"{room_name}_IR.wav"
    output_filepath = os.path.join(output_path, rir_filename)
    #os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    logging.info(f"Starting shoebox room simulation for {room_name} with Dims=({x_room:.2f}, {y_room:.2f}, {z_room:.2f})")

    try:
        # Create the Shoebox room
        room = pra.ShoeBox(
            p=[x_room, y_room, z_room],
            materials=pra.make_materials(**mat),
            fs=fs,
            max_order=0, # Keep max_order low for Ray Tracing
            ray_tracing=True # Enable ray tracing
        )

        # Generate random source and mic positions within the room boundary
        # Retry position generation if they are too close or invalid
        max_pos_attempts = 100
        for attempt in range(max_pos_attempts):
             try:
                x_source, y_source, z_source, x_mic, y_mic, z_mic = get_random_shoebox_positions(x_room, y_room, z_room)
                dist_sq = (x_source - x_mic)**2 + (y_source - y_mic)**2 + (z_source - z_mic)**2
                if dist_sq > 0.1**2: # Ensure minimum distance (e.g., 10 cm)
                    break # Positions are valid and far enough apart
             except ValueError as ve:
                 logging.warning(f"Position generation attempt {attempt+1}/{max_pos_attempts} failed for {room_name}: {ve}")
             except Exception as e:
                 logging.warning(f"Unexpected error during position generation attempt {attempt+1}/{max_pos_attempts} for {room_name}: {e}")
        else: # This else block executes if the loop completes without 'break'
            raise ValueError(f"Failed to generate valid source/mic positions after {max_pos_attempts} attempts for {room_name}.")


        source_position = [x_source, y_source, z_source]
        mic_center = [x_mic, y_mic, z_mic]

        logging.info(f"Positions for {room_name}: Src=({x_source:.2f}, {y_source:.2f}, {z_source:.2f}), Mic=({x_mic:.2f}, {y_mic:.2f}, {z_mic:.2f})")


        # Run the ray tracing simulation
        rir_data = run_ray_tracing_simulation(room, mic_center, source_position, fs, num_mics)

        if rir_data is None:
             # run_ray_tracing_simulation already logged the error
             return {
                "Name": rir_filename,
                "status": "Error: Ray Tracing Failed",
                "error_message": "See log file for details",
                "room_type": regularity,
                "vertices": shoebox_to_vertices(x_room, y_room, z_room).tolist(),
                "x_source": x_source, "y_source": y_source, "z_source": z_source,
                "x_mic": x_mic, "y_mic": y_mic, "z_mic": z_mic,
             }


        # Save the RIR data
        save_rir_wav(rir_data, output_filepath, fs)

        # Calculate acoustic parameters (e.g., RT60 from the first channel)
        rt60, c50 = calculate_acoustic_params(rir_data[0], fs) # Use the first channel for simplicity

        logging.info(f"Successfully simulated {room_name}. RT60: {rt60:.2f}, C50: {c50:.2f}")

        # Return metadata and results
        return {
            "Name": rir_filename,
            "RT": rt60,
            "c50": c50, # Added C50
            "room_type": regularity,
            "vertices": shoebox_to_vertices(x_room, y_room, z_room).tolist(),
            "x_source": x_source,
            "y_source": y_source,
            "z_source": z_source,
            "x_mic": x_mic,
            "y_mic": y_mic,
            "z_mic": z_mic,
            "status": "Success"
        }

    except ValueError as ve:
        error_msg = f"ValueError for {room_name}: {ve}"
        logging.error(error_msg)
        return {
            "Name": rir_filename,
            "status": "Error: ValueError",
            "error_message": str(ve),
            "room_type": regularity,
             "vertices": shoebox_to_vertices(x_room, y_room, z_room).tolist() if 'x_room' in locals() else None,
            "x_source": x_source if 'x_source' in locals() else np.nan,
            "y_source": y_source if 'y_source' in locals() else np.nan,
            "z_source": z_source if 'z_source' in locals() else np.nan,
            "x_mic": x_mic if 'x_mic' in locals() else np.nan,
            "y_mic": y_mic if 'y_mic' in locals() else np.nan,
            "z_mic": z_mic if 'z_mic' in locals() else np.nan,
        }
    except Exception as e:
        # Catch any other exceptions
        error_msg = f"An unexpected error occurred during shoebox simulation for {room_name}: {e}"
        logging.error(error_msg, exc_info=True)
        return {
            "Name": rir_filename,
            "status": "Error: Unexpected Exception",
            "error_message": str(e),
            "room_type": regularity,
             "vertices": shoebox_to_vertices(x_room, y_room, z_room).tolist() if 'x_room' in locals() else None,
             "x_source": x_source if 'x_source' in locals() else np.nan,
            "y_source": y_source if 'y_source' in locals() else np.nan,
            "z_source": z_source if 'z_source' in locals() else np.nan,
            "x_mic": x_mic if 'x_mic' in locals() else np.nan,
            "y_mic": y_mic if 'y_mic' in locals() else np.nan,
            "z_mic": z_mic if 'z_mic' in locals() else np.nan,
        }


# Removed generate_rir_for_all_combinations_together as its logic was merged and simplified into simulate_complex_room/simulate_shoebox_room


def main(output_path: str, num_rooms: int, num_positions: int, sample_rate: int, ambi_order: int):
    """
    Main function to generate a dataset of RIRs with varying room acoustics
    and source/microphone positions using Ray Tracing for complex geometries.

    Args:
        output_path: Directory to save the generated RIR WAV files and data CSV.
        num_rooms: The number of different room geometries to simulate.
        num_positions: The number of different source/mic positions
                       within each room geometry.
        sample_rate: The sample rate for the simulations.
    """
    logging.info(f"Starting RIR generation (v2 Ray Tracing): {num_rooms} rooms, {num_positions} positions per room, "
                 f"Sample Rate {sample_rate}")

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    all_results = [] # Collect results (success or error details)

    num_mics = (ambi_order + 1)**2

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor:
        futures = []
        # Outer loop for different room geometries
        for i in tqdm(range(num_rooms), desc="Generating Rooms"):
            try:
                # Choose room type and generate its specific parameters
                regularity = np.random.choice(ROOM_TYPES, p=ROOM_TYPE_PROBABILITIES)

                if regularity == 'shoebox':
                    x_room, y_room, z_room = get_random_shoebox_dimensions()
                    mat = get_random_materials()
                    logging.info(f"Shoebox has materials: {mat}")

                    # Inner loop for different source/mic positions within the current room
                    for j in range(num_positions):
                        # Submit the simulation task to the executor
                        futures.append(executor.submit(
                            simulate_shoebox_room,
                            i, j, output_path, sample_rate, mat, x_room, y_room, z_room, num_mics, regularity
                        ))
                else:
                    # For complex shapes, generate room parameters once per room
                    height = np.random.uniform(2.0, 8.0) # Random height for complex rooms
                    wall_mat, vert_mat, corners, height, debug_mat = makeCoupledPolygonRoom(sample_rate, height=height, shape=regularity)
                    logging.info(f"Complex room {i} has materials: {debug_mat}")
                    # Inner loop for different source/mic positions within the current room
                    for j in range(num_positions):
                         # Submit the simulation task to the executor
                         futures.append(executor.submit(
                            simulate_complex_room,
                            i, j, output_path, sample_rate, wall_mat, vert_mat, corners, height, num_mics, regularity
                         ))

            except ValueError as ve:
                 error_msg = f"ValueError during room setup (Room {i}, Type: {regularity}): {ve}"
                 logging.error(error_msg)
                 # Log this specific failure as it affects all positions for this room
                 for j in range(num_positions):
                     all_results.append({
                             "Name": f"R{i}_P{j}_IR.wav",
                             "status": "Error: Room Setup Failed (ValueError)",
                             "error_message": str(ve),
                             "room_type": regularity,
                             "x_room": np.nan, "y_room": np.nan, "z_room": np.nan,
                             "x_source": np.nan, "y_source": np.nan, "z_source": np.nan,
                             "x_mic": np.nan, "y_mic": np.nan, "z_mic": np.nan,
                             "vertices": None
                         })

            except Exception as e:
                 # Catch any other exceptions during room setup
                 error_msg = f"An unexpected error during room setup (Room {i}, Type: {regularity}): {e}"
                 logging.error(error_msg, exc_info=True)
                 # Log this error as it affects all positions for this room
                 for j in range(num_positions):
                     all_results.append({
                             "Name": f"R{i}_P{j}_IR.wav",
                             "status": "Error: Unexpected Room Setup Exception",
                             "error_message": str(e),
                             "room_type": regularity,
                              "x_room": np.nan, "y_room": np.nan, "z_room": np.nan,
                             "x_source": np.nan, "y_source": np.nan, "z_source": np.nan,
                             "x_mic": np.nan, "y_mic": np.nan, "z_mic": np.nan,
                             "vertices": None
                         })


        # Process the results as they complete
        # Use tqdm here to show progress of completed simulations
        for future in tqdm(futures, desc="Processing Simulation Results"):
            # The result from simulate_room/simulate_shoebox_room is the dictionary
            result = future.result()
            if result is not None: # Only append if result is not None
                 all_results.append(result)
            # else: # if result is None, an error occurred and was logged internally


    # Separate successful results from errors for the main data CSV
    successful_results = [res for res in all_results if res and res.get("status") == "Success"]
    # Include results that indicate warnings (like zero RIR) in the success file, but note the status
    warning_results = [res for res in all_results if res and res.get("status", "").startswith("Warning")]
    successful_and_warning_results = successful_results + warning_results

    error_results = [res for res in all_results if res and res.get("status") and not res.get("status", "").startswith(("Success", "Warning"))]


    # Save successful and warning results to the main data CSV 
    if successful_and_warning_results:
        data_df = pd.DataFrame(successful_and_warning_results)
        data_output_path = os.path.join(output_path, "Generated_HOA_SRIR_data.csv") # Save data CSV 
        data_df.to_csv(data_output_path, index=False)
        logging.info(f"Successfully saved metadata for {len(successful_and_warning_results)} RIRs (including warnings) to {data_output_path}")
    else:
        logging.warning("No successful or warning RIR simulations to save metadata for.")

    # Save error log to a separate CSV (in the main output folder)
    if error_results:
        error_df = pd.DataFrame(error_results)
        error_log_path = os.path.join(output_path, "rir_error_log_v2.csv") # Use a different error log name
        error_df.to_csv(error_log_path, index=False)
        logging.warning(f"Encountered {len(error_results)} errors during RIR generation. "
                        f"Error details saved to {error_log_path}")
    else:
        logging.info("No errors reported during RIR generation.")

    logging.info("RIR generation process completed.")


if __name__ == "__main__":
    # multiprocessing.set_start_method('spawn') # Keep this if you have issues on certain OS (like macOS/Windows)

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate Room Impulse Responses (RIRs) using pyroomacoustics Ray Tracing with complex geometries.")

    # Add arguments for each parameter
    parser.add_argument(
        "--output_path",
        type=str,
        default="./Gen_RIRs/", # Default output directory
        help="Directory to save the generated RIR WAV files and data CSV."
    )
    parser.add_argument(
        "--num_rooms",
        type=int,
        default=10, # Default value
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
        default=2, # Default value (7th order recommended)
        help="The Ambisonic order for the microphone array (e.g., 0, 1, 2, 3, ...)."
    )

    # Sample rate is used by the microphone array and room setup
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
        sample_rate=args.sample_rate,
        ambi_order= args.ambi_order
    )
