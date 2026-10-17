import numpy as np
import matplotlib.pyplot as plt
from shroom.acoustics.spherical_array import SphericalArray
from shroom.geometry.sampling import sphereicalGrid
from shroom.utils.grid_utils import from_fibonacci_grid
import sofar as sf

# ------------------------------------------------------------
# PARAMETERS
# ------------------------------------------------------------
FS = 48000
DURATION = 256 / FS
Rad = 0.042
SH_ORDER = 40

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
co = np.pi/2 - el

mics_grid = sphereicalGrid(az=az, co=co)
source_grid = from_fibonacci_grid(2000)

# ------------------------------------------------------------
# ARRAY (FREQUENCY DOMAIN OUTPUT)
# ------------------------------------------------------------
array = SphericalArray(
    fs=FS,
    duration=DURATION,
    r_sphere=Rad,
    r_mics=np.full(len(mic_dirs_deg), Rad),
    source_grid=source_grid,
    mics_grid=mics_grid,
    sphere_type="rigid",
    sh_order_for_sm_calc=SH_ORDER,
    convert_to_time=True,
    source_type="plane_wave",
    apply_damping=False,
    normalize_columns=False,
)

print("Array shape:", array.data.shape)
# Center the impulse responses
array.data = np.roll(array.data, array.data.shape[-1] // 2, axis=-1)

# Optional: remove time-aliasing tails
keep_samples = int(0.001 * FS)  # 2 ms

window = np.zeros(array.data.shape[-1])
start = array.data.shape[-1]//2 - keep_samples//2
end = start + keep_samples

from scipy.signal.windows import tukey
window[start:end] = tukey(keep_samples, alpha=0.5)

array.data *= window[None, None, :]



# ------------------------------------------------------------
# SOFA formatting
# ------------------------------------------------------------
ir = np.transpose(array.data, (1, 0, 2))  # (M, R, N)
ir = np.ascontiguousarray(ir, dtype=np.float64)

M, R, N = ir.shape
print("SOFA IR shape:", ir.shape)

# ------------------------------------------------------------
# CREATE SOFA OBJECT
# ------------------------------------------------------------
sofa = sf.Sofa("GeneralFIR")

sofa.Data_IR = ir
sofa.Data_SamplingRate = np.array([FS], dtype=np.float64)
sofa.Data_Delay = np.zeros((M, R), dtype=np.float64)

# ------------------------------------------------------------
# SOURCE POSITIONS
# ------------------------------------------------------------
sofa.SourcePosition = np.column_stack([
    np.rad2deg(source_grid.az),
    np.rad2deg(source_grid.co),
    np.ones(M)
])

sofa.SourcePosition_Type = "spherical"
sofa.SourcePosition_Units = "degree, degree, meter"

# ------------------------------------------------------------
# MICROPHONE POSITIONS
# ------------------------------------------------------------
sofa.ReceiverPosition = np.column_stack([
    np.rad2deg(mics_grid.az),
    np.rad2deg(mics_grid.co),
    np.full(R, Rad)
])

sofa.ReceiverPosition_Type = "spherical"
sofa.ReceiverPosition_Units = "degree, degree, meter"

# ------------------------------------------------------------
# METADATA
# ------------------------------------------------------------
sofa.GLOBAL_Title = "Eigenmike 32ch SHROOM Simulated ATFs"
sofa.GLOBAL_Comment = "Converted from frequency-domain SHROOM output with causal alignment"

# ------------------------------------------------------------
# WRITE FILE
# ------------------------------------------------------------
sf.write_sofa("eigenmike32_simulated.sofa", sofa)

print("✔ SOFA file successfully written!")