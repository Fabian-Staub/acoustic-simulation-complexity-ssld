import numpy as np
import pyroomacoustics as pra

fs = 16000

# Vertices of a 6-wall room (counter-clockwise)
corners = np.array([
    [0.0, 1.0, 0.0, -1.0],  # x
    [0.0, 10.0, 20.0, 10.0],  # y
])

room = pra.Room.from_corners(
    corners,
    fs=fs,
    max_order=20
)

# Source
room.add_source([0.5, 11.0])

# One microphone
mic_locs = np.array([
    [0.5],
    [10.0]
])

room.add_microphone_array(
    pra.MicrophoneArray(mic_locs, fs)
)

room.image_source_model()

src = room.sources[0]

print("image source array shape:", src.images.shape)
print("including direct source:", src.images.shape[1])
print("image sources only:", src.images.shape[1] - 1)