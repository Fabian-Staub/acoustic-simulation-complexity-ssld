import os

# ----------------------------
# FORCE SAFE HPC THREADING
# ----------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import numpy as np
from scipy.io import wavfile

# ----------------------------
# DEBUG TIMER
# ----------------------------
def stamp(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ----------------------------
# LOAD WAV
# ----------------------------
stamp("Loading WAV")
fs, rir = wavfile.read(
    "/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/arni_em32/sh_wavs/RIRs_raw/R1_P1_S1.wav"
)

stamp(f"Loaded fs={fs}, shape={rir.shape}, dtype={rir.dtype}")

# ----------------------------
# FIX INPUT FORMAT (IMPORTANT)
# ----------------------------

# Convert to (channels, samples) and force float32 contiguous memory
rir_em32 = np.ascontiguousarray(rir.T, dtype=np.float32)

stamp(f"After transpose + float32: {rir_em32.shape}, dtype={rir_em32.dtype}")

assert rir_em32.ndim == 2, "Expected (channels, samples)"

# Add batch dimension -> (batch, channels, samples)
rir_em32 = rir_em32[np.newaxis, :, :]

stamp(f"After batch reshape: {rir_em32.shape}")

# FINAL SAFETY CHECK (IMPORTANT FOR SHROOM)
assert isinstance(rir_em32, np.ndarray)
assert rir_em32.flags['C_CONTIGUOUS'], "Input must be C-contiguous"

# ----------------------------
# IMPORTS (AFTER ENV SET)
# ----------------------------
from em32_to_sh_1 import arraySHTfiltersTheory_regLS
import pyroomacoustics as pra

# MONKEY PATCH (optional safety)
_old_shoebox = pra.ShoeBox
def safe_shoebox(*args, **kwargs):
    kwargs.pop("absorption", None)
    return _old_shoebox(*args, **kwargs)
pra.ShoeBox = safe_shoebox

from shroom.acoustics.spherical_array import SphericalArray
from shroom.geometry.sampling import sphereicalGrid
from shroom.acoustics.processors import ProcessorChain, ASMEncoder
from shroom.encoders.asm import ASM
from shroom.utils.grid_utils import from_fibonacci_grid

# ----------------------------
# PARAMETERS
# ----------------------------
ambi_order = 4
SM_SH_ORDER = ambi_order
R = 0.042
DURATION = 1024 / fs

stamp("Building microphone geometry")

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
co = np.pi / 2 - el

mics_grid = sphereicalGrid(az=az, co=co)
source_grid = from_fibonacci_grid(480)

n_mics = mic_dirs_deg.shape[0]

stamp("Creating SphericalArray")

array = SphericalArray(
    source_grid=source_grid,
    mics_grid=mics_grid,
    r_mics=np.full(n_mics, R),
    fs=fs,
    duration=DURATION,
    r_sphere=R,
    sh_order_for_sm_calc=SM_SH_ORDER,
    convert_to_time=False,
)

stamp("Converting array to SH")

array_time_sh = array.copy()
array_time_sh.toTime()
array_time_sh.toSH(ambi_order)

# ----------------------------
# ASM INIT
# ----------------------------
stamp("Initializing ASM")

asm = ASM(
    sh_order=ambi_order,
    array=array,
    fs=fs,
    duration=DURATION
)

stamp("ASM ready")

# ----------------------------
# PROCESSOR CHAIN
# ----------------------------
chain = ProcessorChain([
    ASMEncoder(asm),
])

stamp("Starting chain.process()")

# ----------------------------
# RUN PIPELINE
# ----------------------------
import signal

class TimeoutError(Exception):
    pass

def handler(signum, frame):
    raise TimeoutError("PROCESS TIMED OUT")

signal.signal(signal.SIGALRM, handler)
signal.alarm(300)

try:
    t0 = time.time()

    stamp(f"Input shape: {rir_em32.shape}, dtype: {rir_em32.dtype}")

    sh_output = chain.process(rir_em32)

    stamp(f"chain.process DONE in {time.time() - t0:.2f}s")

    signal.alarm(0)

except Exception as e:
    stamp(f"FAILED inside chain.process: {e}")
    raise

# ----------------------------
# OUTPUT
# ----------------------------
stamp("Extracting output")

RIR = sh_output.data

stamp(f"Output shape: {np.shape(RIR)}")

# ----------------------------
# SAVE
# ----------------------------
stamp("Saving WAV")

wavfile.write("rir_sh.wav", fs, RIR.astype(np.float32).T)

stamp("DONE")