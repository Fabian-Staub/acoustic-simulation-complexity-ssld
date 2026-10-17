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

FS = 44100
DURATION = 1024/FS
SM_SH_ORDER = 20
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

# convert elevation → colatitude
el = np.deg2rad(mic_dirs_deg[:, 1])
co = np.pi/2 - el

mics_grid = sphereicalGrid(az=az, co=co)

n_mics = mic_dirs_deg.shape[0]

source_grid = from_fibonacci_grid(2000)

array = SphericalArray(
    fs = FS,
    duration = DURATION,
    r_sphere = R,
    r_mics=np.full(n_mics, R),
    source_grid=source_grid,
    mics_grid=mics_grid,
    sphere_type = "rigid",
    sh_order_for_sm_calc = 14,
    convert_to_time = True,
    source_type = "plane_wave",
    apply_damping = False,
    normalize_columns = False,
)

print(array.data.shape)


import numpy as np
import sofar as sf

# ------------------------------------------------------------
# INPUT: SHROOM ARRAY
# ------------------------------------------------------------
print("Original shape:", array.data.shape)
# (32, 480, 1024)

R_EIGENMIKE = 0.042

# ------------------------------------------------------------
# FIX 1: Correct SOFA dimension order
# SOFA expects: (M, R, N)
# M = 480 (sources)
# R = 32  (mics)
# N = 1024 (samples)
# ------------------------------------------------------------
ir = np.transpose(array.data, (1, 0, 2))

# Ensure clean numeric contiguous float array
ir = np.ascontiguousarray(ir, dtype=np.float64)

M, R, N = ir.shape
print("SOFA IR shape:", ir.shape)

# ------------------------------------------------------------
# CREATE SOFA OBJECT
# ------------------------------------------------------------
sofa = sf.Sofa("GeneralFIR")

# ------------------------------------------------------------
# REQUIRED DATA
# ------------------------------------------------------------
sofa.Data_IR = ir
sofa.Data_SamplingRate = np.array([FS], dtype=np.float64)

# IMPORTANT FIX: Data_Delay MUST be (M, R)
sofa.Data_Delay = np.zeros((M, R), dtype=np.float64)

# ------------------------------------------------------------
# SOURCE POSITIONS (480 directions)
# ------------------------------------------------------------
az = np.rad2deg(source_grid.az)
el = 90 - np.rad2deg(source_grid.co)

sofa.SourcePosition = np.column_stack([
    az,
    el,
    np.ones(M)  # radius (far-field convention)
])

sofa.SourcePosition_Type = "spherical"
sofa.SourcePosition_Units = "degree, degree, meter"

# ------------------------------------------------------------
# MICROPHONE POSITIONS (Eigenmike)
# ------------------------------------------------------------
mic_az = np.rad2deg(mics_grid.az)
mic_el = 90 - np.rad2deg(mics_grid.co)

sofa.ReceiverPosition = np.column_stack([
    mic_az,
    mic_el,
    np.full(R, R_EIGENMIKE)
])

sofa.ReceiverPosition_Type = "spherical"
sofa.ReceiverPosition_Units = "degree, degree, meter"

# ------------------------------------------------------------
# LISTENER POSITION
# ------------------------------------------------------------
sofa.ListenerPosition = np.array([[0.0, 0.0, 0.0]])
sofa.ListenerPosition_Type = "cartesian"
sofa.ListenerPosition_Units = "meter"

# ------------------------------------------------------------
# GLOBAL METADATA (IMPORTANT for validation)
# ------------------------------------------------------------
sofa.GLOBAL_Title = "Eigenmike 32ch SHROOM Simulated ATFs"

# optional but often required by stricter validators
sofa.GLOBAL_Comment = "Generated from SHROOM SphericalArray simulation"

# ------------------------------------------------------------
# WRITE FILE
# ------------------------------------------------------------
sf.write_sofa("eigenmike32_simulated.sofa", sofa)

print("✔ SOFA file successfully written!")