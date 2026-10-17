import os
import numpy as np
import soundfile as sf

input_dir = "/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/SummerAaltoMeasurements/processed_data/sh_wavs/RIRs"
output_dir = os.path.join(os.path.dirname(input_dir), "RIRs_truncated")

os.makedirs(output_dir, exist_ok=True)

def trim_zeros(audio):
    """
    Remove leading and trailing zeros across all channels.
    """
    if audio.ndim == 1:
        mask = audio != 0
    else:
        mask = np.any(audio != 0, axis=1)

    if not np.any(mask):
        return audio[:0]  # empty array if all zeros

    start = np.argmax(mask)
    end = len(mask) - np.argmax(mask[::-1])

    return audio[start:end]

for fname in os.listdir(input_dir):
    if not fname.lower().endswith(".wav"):
        continue

    in_path = os.path.join(input_dir, fname)
    out_path = os.path.join(output_dir, fname)

    audio, sr = sf.read(in_path)

    trimmed = trim_zeros(audio)

    sf.write(out_path, trimmed, sr)

    print(f"Processed: {fname} -> {out_path}")