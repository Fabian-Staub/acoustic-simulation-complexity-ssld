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
SM_SH_ORDER = 14
AA_MAGLS_CUTOFF = 1100

ROOM_DIMS = [6.0, 5.0, 3.0]
MAX_ISM_ORDER = 5

HEAD_POS = [2.0, 2.0, 1.5]
SOURCE_POS = [4.0, 4.0, 1.5]
HEAD_ROTATION = [0, 0, 0]


def normalize(x):
    peak = np.max(np.abs(x))
    return x / peak * 0.99 if peak > 0 else x


def main():
    n_fft = int(DURATION * FS)

    print("--- Starting SH-only ASM Simulation (NO Binaural) ---")

    # -------------------------
    # Room setup
    # -------------------------
    material_dict = {
        "description": "Material Name",
        "coeffs": [0.1]*8,
        "center_freqs": [125, 250, 500, 1000, 2000, 4000, 8000, 16000],
    }

    materials = pra.make_materials(
        **{w: (material_dict,) for w in ["west","east","north","south","floor","ceiling"]}
    )

    room = Room(
        dimensions=ROOM_DIMS,
        materials=materials,
        max_ism_order=MAX_ISM_ORDER,
        fs=FS,
    )

    # -------------------------
    # Spherical array
    # -------------------------
    mics_grid = sphereicalGrid(
        az=np.deg2rad(np.array([-90, -45, 0, 45, 90])),
        co=np.deg2rad(np.array([90, 108, 72, 108, 90])),
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
        convert_to_time=False,
    )

    array_time_sh = array.copy()
    array_time_sh.toTime()
    array_time_sh.toSH(ASM_SH_ORDER)

    # -------------------------
    # ASM
    # -------------------------
    print("Computing ASM...")
    asm = ASM(sh_order=ASM_SH_ORDER, array=array, fs=FS, duration=DURATION)

    # -------------------------
    # HRTF processing (still computed but NOT used for decoding anymore)
    # -------------------------
    print("Loading HRTF...")
    hrtf_ls = load_file(DEFAULT_HRTF_PATH)

    if HEAD_ROTATION is not None:
        rot = Rotation.from_euler("zxy", HEAD_ROTATION, degrees=True)
        hrtf_ls.rotate_space_domain(rot)

    hrtf_ls.resample(desired_fs=FS)
    hrtf_ls.zero_pad(n_fft)
    hrtf_ls.toSH(N_sp=ASM_SH_ORDER)

    # -------------------------
    # Room simulation
    # -------------------------
    room.add_source(SOURCE_POS, signal=DEFAULT_WAV_PATH)
    room.set_receiver(HEAD_POS)

    print("Simulating room...")
    amb_room = room.compute_amb()

    print(f"Amb shape: {amb_room.data.shape}")

    # -------------------------
    # PROCESSING CHAINS (SH ONLY)
    # -------------------------
    print("Running ASM chain (SH only)...")

    chain_ls = ProcessorChain([
        ArrayDecoder(array_time_sh),
        ASMEncoder(asm),
    ])

    chain_magls = ProcessorChain([
        ArrayDecoder(array_time_sh),
        ASMEncoder(asm),
    ])

    chain_aa = ProcessorChain([
        ArrayDecoder(array_time_sh),
        ASMEncoder(asm),
    ])

    asm_ls = chain_ls.process(amb_room)
    asm_magls = chain_magls.process(amb_room)
    asm_aa = chain_aa.process(amb_room)

    # -------------------------
    # Convert outputs
    # -------------------------
    def extract(x):
        return np.asarray(x.data if hasattr(x, "data") else x)

    audio_ls = extract(asm_ls)
    audio_magls = extract(asm_magls)
    audio_aa = extract(asm_aa)

    # -------------------------
    # Normalize (important for SH signals too)
    # -------------------------
    audio_ls = normalize(audio_ls)
    audio_magls = normalize(audio_magls)
    audio_aa = normalize(audio_aa)

    # -------------------------
    # Save (SH-domain "audio-like" tensors)
    # -------------------------
    write("asm_ls_sh.wav", int(FS), audio_ls.astype(np.float32))
    write("asm_magls_sh.wav", int(FS), audio_magls.astype(np.float32))
    write("asm_aa_magls_sh.wav", int(FS), audio_aa.astype(np.float32))

    print("Saved SH outputs (NOT binaural audio):")
    print("  asm_ls_sh.wav")
    print("  asm_magls_sh.wav")
    print("  asm_aa_magls_sh.wav")


if __name__ == "__main__":
    main()