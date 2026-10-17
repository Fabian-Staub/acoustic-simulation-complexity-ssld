import pyroomacoustics as pra

# --- MONKEY PATCH START ---
_old_shoebox = pra.ShoeBox

def safe_shoebox(*args, **kwargs):
    # remove unsupported argument (causes crash in your version)
    kwargs.pop("absorption", None)
    return _old_shoebox(*args, **kwargs)

pra.ShoeBox = safe_shoebox
# --- MONKEY PATCH END ---


import numpy as np
import matplotlib.pyplot as plt
from shroom.acoustics.room import Room
from shroom.acoustics.spherical_array import SphericalArray
from shroom.acoustics.processors import (
    ProcessorChain,
    ArrayDecoder,
    ASMEncoder,
    BinauralDecoder,
)
from shroom.acoustics.hrtf_processing import magls_hrtf, array_aware_magls_hrtf
from shroom.geometry.sampling import sphereicalGrid
from shroom.utils.grid_utils import from_fibonacci_grid
from shroom.encoders.asm import ASM
from shroom.utils.file_utils import load_file
from shroom_dev.sound import play_audio
from scipy.spatial.transform import Rotation
from shroom.paths import DEFAULT_HRTF_PATH, DEFAULT_WAV_PATH
from scipy.io.wavfile import write

# --- Configuration ---
FS = 48000.0
DURATION = 0.0026666666666666666
ASM_SH_ORDER = 1
SM_SH_ORDER = 14          # accuracy of steering matrix computation
AA_MAGLS_CUTOFF = 1100    # Hz — LS below, AA-MagLS above
ROOM_DIMS = [6.0, 5.0, 3.0]
ABSORPTION = 0.8
MAX_ISM_ORDER = 0
HEAD_POS = [2.0, 2.0, 1.5]
SOURCE_POS = [4.0, 4.0, 1.5]
HEAD_ROTATION = [0, 0, 0]


def main():
    n_fft = int(DURATION * FS)

    # 1. Initialize Room
    print("--- Starting Binaural Room Simulation (ASM + AA-MagLS) ---")

    material_dict = {
        "description": "Material Name",
        "coeffs": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "center_freqs": [125, 250, 500, 1000, 2000, 4000, 8000, 16000],
    }
    wall_names = ["west", "east", "north", "south", "floor", "ceiling"]
    materials = pra.make_materials(
        **{wall: (material_dict,) for wall in wall_names}
    )

    room = Room(
        dimensions=ROOM_DIMS,
        materials=materials,
        max_ism_order=MAX_ISM_ORDER,
        fs=FS,
    )

    print(f"Room: {ROOM_DIMS}m, absorption={ABSORPTION}")

    # 2. Setup Spherical Array
    print("Setting up spherical array...")
    mics_grid = sphereicalGrid(
        az=np.deg2rad(np.array([-90, -45, 0, 45, 90])),
        co=np.deg2rad(np.array([90, 90 + 18, 90 - 18, 90 + 18, 90])),
    )
    source_grid = from_fibonacci_grid(480)

    array = SphericalArray(
        source_grid=source_grid,
        mics_grid=mics_grid,
        r_mics=np.full(mics_grid.n_points, 0.08),
        fs=FS,
        duration=DURATION,
        r_sphere=0.08,
        sh_order_for_sm_calc=SM_SH_ORDER,
        convert_to_time=False,   # keep in frequency domain for ASM + AA-MagLS
    )

    # Array in time+SH domain — needed by ArrayDecoder to simulate mic recording
    array_time_sh = array.copy()
    array_time_sh.toTime()
    array_time_sh.toSH(ASM_SH_ORDER)

    # 3. Setup ASM Encoder
    print("Computing ASM filters...")
    asm = ASM(sh_order=ASM_SH_ORDER, array=array, fs=FS, duration=DURATION)

    # 4. Prepare HRTF — standard LS (for baseline ASM chain)
    print("Loading and preparing HRTF (standard LS)...")
    hrtf_ls = load_file(DEFAULT_HRTF_PATH)
    if HEAD_ROTATION is not None:
        rot = Rotation.from_euler("zxy", HEAD_ROTATION, degrees=True)
        hrtf_ls.rotate_space_domain(rot)
    hrtf_ls.resample(desired_fs=FS)
    hrtf_ls.zero_pad(n_fft)
    hrtf_ls.toSH(N_sp=ASM_SH_ORDER)

    # 5. Prepare HRTF — AA-MagLS
    print(f"Computing AA-MagLS HRTF (order={ASM_SH_ORDER}, cutoff={AA_MAGLS_CUTOFF} Hz)...")
    hrtf_raw = load_file(DEFAULT_HRTF_PATH)
    if HEAD_ROTATION is not None:
        hrtf_raw.rotate_space_domain(rot)
    hrtf_raw.resample(desired_fs=FS)
    hrtf_raw.zero_pad(n_fft)

    # Bring into space domain at source_grid (via high-order SH interpolation)
    hrtf_raw.toFreq()
    hrtf_raw.toSH(30)
    hrtf_raw.toSpace(source_grid)
    space_hrtf = hrtf_raw  # (2, Q, F) — space + freq domain

    # 5a. MagLS HRTF
    print(f"Computing MagLS HRTF (order={ASM_SH_ORDER}, cutoff={AA_MAGLS_CUTOFF} Hz)...")
    hrtf_magls = magls_hrtf(
        hrtf=space_hrtf,
        sh_order=ASM_SH_ORDER,
        cutoff_over_freq=AA_MAGLS_CUTOFF,
    )
    print(f"MagLS HRTF computed: shape={hrtf_magls.data.shape}")

    # 5b. AA-MagLS HRTF
    print(f"Computing AA-MagLS HRTF (order={ASM_SH_ORDER}, cutoff={AA_MAGLS_CUTOFF} Hz)...")
    hrtf_aa_magls = array_aware_magls_hrtf(
        hrtf=space_hrtf,
        asm=asm,
        array=array,
        sh_order=ASM_SH_ORDER,
        cutoff_over_freq=AA_MAGLS_CUTOFF,
    )
    print(f"AA-MagLS HRTF computed: shape={hrtf_aa_magls.data.shape}")

    # 6. Simulate Room
    print(f"Adding source at {SOURCE_POS}, receiver at {HEAD_POS}...")
    room.add_source(SOURCE_POS)
    room.set_receiver(HEAD_POS)

    print("Simulating room acoustics (Ambisonics)...")
    amb_room = room.compute_amb()
    print(f"Room Ambisonics shape: {amb_room.data.shape}")

    # 7a. Standard ASM chain: Amb -> Array -> ASM -> Binaural (LS HRTF)
    print("\nProcessing: standard ASM chain (LS HRTF)...")
    chain_ls = ProcessorChain(
        [
            ArrayDecoder(array_time_sh),        # Amb(SH) -> mic signals
            ASMEncoder(asm),                    # mic signals -> Amb(SH)
            BinauralDecoder(hrtf_ls, sh_order=ASM_SH_ORDER),
        ]
    )
    binaural_ls = chain_ls.process(amb_room)

    # 7b. MagLS chain: same array path, MagLS HRTF
    print("Processing: ASM chain with MagLS HRTF...")
    chain_magls = ProcessorChain(
        [
            ArrayDecoder(array_time_sh),
            ASMEncoder(asm),
            BinauralDecoder(hrtf_magls, sh_order=ASM_SH_ORDER),
        ]
    )
    binaural_magls = chain_magls.process(amb_room)

    # 7c. AA-MagLS chain: same array path, AA-MagLS HRTF
    print("Processing: ASM chain with AA-MagLS HRTF...")
    chain_aa_magls = ProcessorChain(
        [
            ArrayDecoder(array_time_sh),
            ASMEncoder(asm),
            BinauralDecoder(hrtf_aa_magls, sh_order=ASM_SH_ORDER),
        ]
    )
    binaural_aa_magls = chain_aa_magls.process(amb_room)

    # 8. Save audio
    print("\nSaving ASM (LS HRTF)...")

    # Convert to numpy arrays if needed
    audio_ls = np.asarray(binaural_ls.data if hasattr(binaural_ls, "data") else binaural_ls)
    audio_magls = np.asarray(binaural_magls.data if hasattr(binaural_magls, "data") else binaural_magls)
    audio_aa_magls = np.asarray(binaural_aa_magls.data if hasattr(binaural_aa_magls, "data") else binaural_aa_magls)

    # Ensure shape is (samples, channels)
    if audio_ls.shape[0] == 2:
        audio_ls = audio_ls.T
    if audio_magls.shape[0] == 2:
        audio_magls = audio_magls.T
    if audio_aa_magls.shape[0] == 2:
        audio_aa_magls = audio_aa_magls.T

    # Normalize to avoid clipping
    def normalize(x):
        peak = np.max(np.abs(x))
        return x / peak * 0.99 if peak > 0 else x

    audio_ls = normalize(audio_ls)
    audio_magls = normalize(audio_magls)
    audio_aa_magls = normalize(audio_aa_magls)

    # Save as 32-bit float WAV
    write("binaural_ls.wav", int(FS), audio_ls.astype(np.float32))
    write("binaural_magls.wav", int(FS), audio_magls.astype(np.float32))
    write("binaural_aa_magls.wav", int(FS), audio_aa_magls.astype(np.float32))

    print("Saved:")
    print("  binaural_ls.wav")
    print("  binaural_magls.wav")
    print("  binaural_aa_magls.wav")

    print("Done.")


if __name__ == "__main__":
    main()
