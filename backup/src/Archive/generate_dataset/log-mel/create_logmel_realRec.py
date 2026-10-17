import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

# Parameters
filename = "mix064.wav"
n_mels = 64
window_ms = 40       # window length in ms
hop_ms = 20          # hop length in ms
output_plot = "logmel_all_channels.png"

# Load audio
audio, sr = sf.read(filename)  # shape: (samples, channels)
if audio.ndim == 1:
    audio = audio[:, np.newaxis]

# Convert ms to samples
window_length = int(window_ms / 1000 * sr)
hop_length = int(hop_ms / 1000 * sr)

# Compute log-mel spectrogram for each channel
logmel_list = []
for ch in range(audio.shape[1]):
    S = librosa.feature.melspectrogram(
        y=audio[:, ch],
        sr=sr,
        n_fft=window_length,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0
    )
    log_S = librosa.power_to_db(S, ref=np.max)
    logmel_list.append(log_S)

# Stack channels: shape (channels, n_mels, time_frames)
logmel_array = np.stack(logmel_list, axis=0)

# Plot all channels stacked
fig, axes = plt.subplots(nrows=audio.shape[1], ncols=1, figsize=(12, 2*audio.shape[1]), sharex=True)

for ch, ax in enumerate(axes):
    im = ax.imshow(
        logmel_array[ch],
        aspect='auto',
        origin='lower',
        interpolation='nearest',
        cmap='magma',  # custom colormap
        extent=[0, logmel_array[ch].shape[1]*hop_length/sr, 0, sr/2]
    )
    ax.set_ylabel('Mel bin')
    ax.set_title(f'Channel {ch}')
    ax.label_outer()  # hide inner x labels

# Single colorbar
cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.01)
cbar.set_label('dB')

plt.xlabel('Time [s]')
plt.tight_layout()
plt.savefig(output_plot)
plt.close()

print(f"Log-mel spectrogram for all channels saved to {output_plot}")
print("Log-mel spectrogram array shape:", logmel_array.shape)