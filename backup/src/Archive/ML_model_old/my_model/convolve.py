import os
import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf

# === User variable: base data folder ===
data_folder = "../data"

# === File paths ===
mono_speech_folder = os.path.join(data_folder, "audio_exports_Harvard_Sets")
ARIR_folder = os.path.join(data_folder, "hifi-harp/example_RIRs")

# Helper function to get the first WAV in a folder
def first_wav_file(folder):
    for f in os.listdir(folder):
        if f.lower().endswith(".wav"):
            return os.path.join(folder, f)
    raise FileNotFoundError(f"No WAV files found in {folder}")

# === Load mono source (3-channel, take second channel) ===
mono_file = first_wav_file(mono_speech_folder)
mono_data, fs_mono = sf.read(mono_file, dtype='float32')
if mono_data.ndim < 2 or mono_data.shape[1] < 2:
    raise ValueError("Mono file does not have at least 2 channels")
mono_channel = mono_data[:, 1]  # second channel

print(f"Loaded mono source: {mono_file}, shape: {mono_channel.shape}, fs: {fs_mono}")

# === Load Ambisonics RIR (64 channels) ===
ambi_file = first_wav_file(ARIR_folder)
ambi_data, fs_ambi = sf.read(ambi_file, dtype='float32')
if ambi_data.ndim != 2 or ambi_data.shape[1] != 64:
    raise ValueError("Ambisonics file does not have 64 channels")
if fs_mono != fs_ambi:
    raise ValueError("Sample rates of mono and Ambisonics files do not match")

print(f"Loaded Ambisonics RIR: {ambi_file}, shape: {ambi_data.shape}, fs: {fs_ambi}")

# === Convolve mono with each Ambisonics channel ===
num_channels = ambi_data.shape[1]
conv_length = len(mono_channel) + ambi_data.shape[0] - 1
ambisonic_out = np.zeros((conv_length, num_channels), dtype='float32')

for n in range(num_channels):
    ambisonic_out[:, n] = fftconvolve(mono_channel, ambi_data[:, n])

# === Normalize signal to avoid clipping ===
max_val = np.max(np.abs(ambisonic_out))
if max_val > 0:
    ambisonic_out /= max_val  # scale to [-1, 1]

# Optional: scale slightly below 1 to be extra safe
ambisonic_out *= 0.99

# === Save output ===
output_folder = os.path.join(data_folder, "output")
os.makedirs(output_folder, exist_ok=True)
output_file = os.path.join(output_folder, "ambisonic_output.wav")
sf.write(output_file, ambisonic_out, fs_mono, subtype='FLOAT')
print(f"Saved normalized 64-channel Ambisonics output to {output_file}")
