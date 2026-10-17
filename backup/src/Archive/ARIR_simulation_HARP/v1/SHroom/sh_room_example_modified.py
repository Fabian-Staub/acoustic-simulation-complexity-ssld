import numpy as np
import matplotlib.pyplot as plt
from shroom.acoustics.room import Room
from shroom.acoustics.spherical_array import SphericalArray
from shroom.acoustics.processors import (
    ProcessorChain,
    ArrayDecoder,
    ASMEncoder,
    BinauralDecoder,
    BSMEncoder,
)
from shroom.geometry.sampling import sphereicalGrid
from shroom.utils.grid_utils import from_spaudiopy_grid, from_fibonacci_grid
from shroom.encoders.bsm import BSM
from shroom.encoders.asm import ASM
from shroom.utils.file_utils import load_file
from shroom_dev.sound import play_audio
from scipy.spatial.transform import Rotation
from shroom.paths import DEFAULT_HRTF_PATH, DEFAULT_WAV_PATH

# --- Configuration ---
FS = 48000.0
DURATION = 0.008
SM_SH_ORDER = 20          # accuracy of steering matrix computation
ROOM_DIMS = [13.5, 19.5, 6.5]
ABSORPTION = 0.4
MAX_ISM_ORDER = 0
HEAD_POS = [1, 9.5, 4.5]
SOURCE_POS = [6, 2.0, 5.0]
HEAD_ROTATION = [ 0, 0, 0]


def main():
    print("--- Starting Binaural Room Simulation (ASM Chain) ---")

    # 1. Initialize Room
    room = Room(
        dimensions=ROOM_DIMS, absorption=ABSORPTION, max_ism_order=MAX_ISM_ORDER, fs=FS, sh_order = 4
    )
    print(f"Room initialized: {ROOM_DIMS}m, Absorption: {ABSORPTION}")

    R = 0.042  # Eigenmike radius (~4.2 cm)
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

    # IMPORTANT: convert elevation → colatitude
    el = np.deg2rad(mic_dirs_deg[:, 1])
    co = np.pi/2 - el

    mics_grid = sphereicalGrid(az=az, co=co)

    n_mics = mic_dirs_deg.shape[0]

    source_grid = from_fibonacci_grid(480)

    array = SphericalArray(
        source_grid=source_grid,
        mics_grid=mics_grid,
        r_mics=np.full(n_mics, R),  # Eigenmike radius
        fs=FS,
        duration=DURATION,
        r_sphere=R,
        sh_order_for_sm_calc=SM_SH_ORDER,
        convert_to_time=False,
    )
    # array = load_file("/daxta/sofa_hrtfs/qu_kemar_anechoic_2m.sofa")

    # Prepare array for ASM (Frequency Domain)
    array.toTime()
    array.resample(desired_fs=FS)
    array_time_sh = array.copy()
    array_time_sh.toSH(SM_SH_ORDER)
    array_freq = array.copy()
    array_freq.toFreq()


    asm = ASM(sh_order=4, array=array_freq, fs=FS, duration=DURATION)

    impulse = np.zeros(int(FS * DURATION))
    impulse[0] = 1.0

    # 5. Load and Add Source (Directly passing path)
    print(f"Adding source from {DEFAULT_WAV_PATH}...")
    room.add_source(SOURCE_POS, signal=impulse)
    room.set_receiver(HEAD_POS)
    print(f"Source added at {SOURCE_POS}, Receiver at {HEAD_POS}")

    # 6. Simulate Room (Ambisonics)
    print("Simulating Room (Ambisonics)...")
    amb_room = room.compute_amb()
    print(f"Room Ambisonics Shape: {amb_room.data.shape}")

    # 7. Process Chain
    # Chain: Room(Amb) -> Array(Mics) -> BSM(binaural)
    chain = ProcessorChain(
        [
            ArrayDecoder(array_time_sh),  # Simulate array recording
            #ASMEncoder(asm),  # Encode mics to Ambisonics
        ]
    )

    print("Processing Chain...")
    sh_output = chain.process(amb_room)

    # 8. Listen and Plot
    #print("Playing Binaural Audio...")
    #play_audio(sh_output, FS)

    # Extract numpy array
    print(sh_output.data.shape)
    audio = sh_output.data  # (25, 1, samples)
    audio = audio[:, 0, :]  # (25, samples)
    audio = audio.T         # (samples, 25)

    print(audio.shape)

    # Save WAV
    import soundfile as sf
    audio = audio / np.max(np.abs(audio))
    sf.write("shroom_ambisonics_output.wav", audio.astype(np.float32), int(FS))

    # Save plot
    room.plot()
    plt.savefig("room_simulation.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("yes!")


if __name__ == "__main__":
    main()