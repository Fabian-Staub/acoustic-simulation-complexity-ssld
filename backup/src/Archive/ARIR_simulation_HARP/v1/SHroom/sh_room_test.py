import pyroomacoustics as pra
import numpy as np
from shroom.acoustics.spherical_array import SphericalArray

fs = 48000
room_dim = [13.5, 19.5, 6.5]

room = pra.ShoeBox(
    room_dim,
    fs=fs,
    max_order=0,
    absorption=0,
)

source_pos = [5.5, 2.0, 5.0]

room.add_source(source_pos)

# --- Eigenmike parameters ---
R = 0.042  # Eigenmike radius (~4.2 cm)
head_pos = np.array([0.5, 9.5, 4.5])

# --- Eigenmike directions (deg) ---
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

# --- Convert to radians ---
az = np.deg2rad(mic_dirs_deg[:, 0])
el = np.deg2rad(mic_dirs_deg[:, 1])

# --- Spherical → Cartesian ---
x = R * np.cos(el) * np.cos(az)
y = R * np.cos(el) * np.sin(az)
z = R * np.sin(el)

mic_positions = np.vstack((x, y, z))

# --- Translate to head position ---
mic_positions += head_pos[:, np.newaxis]

room.add_microphone_array(
    pra.MicrophoneArray(mic_positions, fs)
)

room.image_source_model()
room.compute_rir()
mic_signals = room.mic_array.signals


array = SphericalArray(
    source_grid=None,
    mics_grid=mic_positions,
    fs=fs,
    duration=mic_signals.shape[1] / fs,
)

array.data = mic_signals

chain = ProcessorChain([
    ASMEncoder(asm),
    BinauralDecoder(hrtf, sh_order=1),
])

binaural = chain.process(array)