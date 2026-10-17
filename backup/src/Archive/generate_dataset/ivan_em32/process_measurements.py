import sofar as sf
import os
from scipy.io import wavfile
from matplotlib import pyplot as plt
import numpy as np
from scipy.fft import fft, ifft, next_fast_len
from scipy import signal, ndimage
import pandas as pd
from tqdm import tqdm

def get_rooms(path_to_measurements):
    
    # get room directories
    rooms = [d for d in os.listdir(path_to_measurements) if os.path.isdir(os.path.join(path_to_measurements, d))]
    
    # exclude test measurements
    rooms = [d for d in rooms if not d.startswith('arni6') and not d.startswith('arni7')]
    return rooms

def set_global_parameters(sofa, array):

    sofa.GLOBAL_Comment = 'Summer Measurements Aalto'
    sofa.GLOBAL_History = 'Created on ' + str(np.datetime64('now'))
    sofa.GLOBAL_Organization = 'Aalto Acoustics Lab'

    if array == 'KEMAR':
        sofa.GLOBAL_ReceiverDescription = 'KEMAR Head and Torso Simulator'
        sofa.ReceiverDescriptions = np.array(['Left ear', 'Right ear'])
        sofa.ReceiverView = np.tile(np.array([[[1],[0],[0]]]), (2, 1, 1))
        sofa.ReceiverUp = np.tile(np.array([[[1],[0],[0]]]), (2, 1, 1))

    if array == 'ARRAI':
        sofa.GLOBAL_ReceiverDescription = 'ARRAI'
        sofa.ReceiverDescriptions = np.array([f'Mic {i}' for i in range(1, 6)])
        sofa.ReceiverView = np.tile(np.array([[[1],[0],[0]]]), (5, 1, 1))
        sofa.ReceiverUp = np.tile(np.array([[[1],[0],[0]]]), (5, 1, 1))

    if array == 'EM':
        sofa.GLOBAL_ReceiverDescription = 'EM'
        sofa.ReceiverDescriptions = np.array([f'Mic {i}' for i in range(1, 33)])
        sofa.ReceiverView = np.tile(np.array([[[1],[0],[0]]]), (32, 1, 1))
        sofa.ReceiverUp = np.tile(np.array([[[1],[0],[0]]]), (32, 1, 1))

def convert_coords_unity_to_sofa(coords_unity):

    # change the coordinate system from Unity to SOFA
    # also consider what is front

    coords = np.empty_like(coords_unity)

    coords[0]  = coords_unity[2] # x used to be z
    coords[1]  = -coords_unity[0] # y used to be -x
    coords[2]  = coords_unity[1] # z used to be y

    return coords

def deconvolve_sweep(sweep, sweep_response, start=None, stop=None):

    # trim if needed
    if start is not None and stop is not None:
        sweep_response = sweep_response[start:stop]

    nfft = next_fast_len(len(sweep_response))

    X = fft(sweep_response, n=nfft)
    S = fft(sweep, n=nfft)

    S[S == 0] = np.finfo(S.dtype).eps

    H = X / S

    return np.real(ifft(H))


def select_by_rule_of_two(irs_repeats):

    num_repeats = irs_repeats.shape[0]
    num_channels = irs_repeats.shape[1]

    corr = np.zeros((num_repeats, num_repeats))

    for iCh in range(num_channels):
        for i in range(num_repeats - 1):
            for j in range(i + 1, num_repeats):
                corr[i, j] += np.corrcoef(irs_repeats[i, iCh, :], irs_repeats[j, iCh, :])[0, 1]

    corr /= num_channels

    if corr.any() < 0.9:
        print("Warning: low correlation between sweeps!")

    row_mx, col_mx = np.unravel_index(np.argmax(corr), corr.shape)

    return (row_mx, col_mx) # return the indices of the two most similar IRs


def time_align_rirs(upsampled_rir):
    # time-align upsampled sweeps

    num_repeat = upsampled_rir.shape[0]

    upsampled_rir = upsampled_rir.T
   
    for rep in range(1, num_repeat):
        # calculate the cross-correlation of upsampled sweeps
        corr = signal.correlate(upsampled_rir[ :,  0], upsampled_rir[ :,  rep])
        # get the time lag as well
        lags = signal.correlation_lags(len(upsampled_rir[:,  0]), len(upsampled_rir[:,  rep]))
        # find where the zero lag is - the lag is symmetric on both sides
        lag_ind = np.nonzero(lags==0)
        zero_lag = lag_ind[0][0] # the position of zero lag
        
        # find the lag of the max correlation - this will define the amount of samples we need to shift
        max_lag = np.argmax(corr)
        sign = np.sign(max_lag - zero_lag) # sign will tell us whether to shift to left or right
        shift = np.abs(max_lag- zero_lag)+1 # needs to be shifted by 1 because of indexing through zero
        
        # shift the IR
        shifted = np.roll(upsampled_rir[ :,  rep], sign*shift)
        upsampled_rir[ :,  rep]= shifted

    return upsampled_rir


def find_start_end_indices(irs_repeats):

    wL = 0.01*fs
    win = signal.windows.boxcar(int(wL))
    win = win/np.sum(win)

    rir_len = irs_repeats.shape[2]
    upFactor = 10

    num_repeats = irs_repeats.shape[0]
    num_channels = irs_repeats.shape[1]

    irs_aligned = np.zeros([num_repeats, num_channels, rir_len])

    irs_repeats = irs_repeats / np.max(np.abs(irs_repeats))
    start_idx = np.zeros(num_channels, dtype=int)
    end_idx = np.zeros(num_channels, dtype=int)     

    for iCh in range(num_channels):
        rirs_upsampled = signal.resample(irs_repeats[:, iCh, :], (rir_len)*upFactor, axis=1)
        rirs_aligned_up = time_align_rirs(rirs_upsampled)
        irs_aligned[:, iCh, :] = signal.resample(rirs_aligned_up, rir_len, axis=0).T

        r_coh = np.convolve(irs_aligned[0, iCh, :]*irs_aligned[1, iCh, :], win, mode='same')

        # Select the last 2 seconds of r_coh
        r_coh_thresh = np.mean(np.abs(r_coh[-int(2*fs):]))

        valid = np.where(r_coh > r_coh_thresh)[0]

        #plt.figure()
        #plt.plot(r_coh)
        #plt.plot(r_coh> r_coh_thresh)

        # Find breaks in the sequence
        breaks = np.where(np.diff(valid) != 1)[0]
        segments = np.split(valid, breaks + 1)
        # Get the longest segment
        longest_segment = max(segments, key=len)
        start_idx[iCh] = longest_segment[0]
        end_idx[iCh] = longest_segment[-1] 

    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # ax1.plot(20*np.log10(np.abs(irs_aligned[:, iCh, :].T)))
    # ax1.plot(np.arange(start_idx[iCh], end_idx[iCh]), 20*np.log10(np.abs(irs_aligned[:, iCh, start_idx[iCh]:end_idx[iCh]].T)))
    # ax1.set_title(f"Channel {iCh+1}")
    
    # ax2.plot(np.log(r_coh))
    # ax2.plot(np.log(np.ones_like(r_coh)*r_coh_thresh))
    # ax2.plot([start_idx[iCh],start_idx[iCh]], [-35, 0], 'r--')
    # ax2.plot([end_idx[iCh],end_idx[iCh]], [-35, 0], 'r--')
    # plt.tight_layout()

    start_idx_min = np.min(start_idx)
    end_idx_max = np.max(end_idx)

    return start_idx_min, end_idx_max

# %% Main script

if __name__ == "__main__":
    path_to_measurements = r"/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/SummerAaltoMeasurements/rooms"
    path_to_output = r"/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/SummerAaltoMeasurements/processed_data/"

    array = 'EM' #'KEMAR', 'EM', etc

    for room in tqdm(sorted(get_rooms(path_to_measurements)), desc="Processing rooms"):

        geometry = pd.read_csv(os.path.join(path_to_measurements, room, 'Geometry', 'Pos_sketch.csv'))

        fs_sweep, sweep = wavfile.read('/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/SummerAaltoMeasurements/triple_sweep_clean_cut_60_24k_48k_5sec0pad.wav')

        sweep_len_s = 5
        sweep = sweep[:fs_sweep*sweep_len_s]

        path = os.path.join(path_to_measurements, room, array)

        position_coords_all = np.empty((0, 3))
        source_coords_all = np.empty((0, 3))

        # Highpass filter at 20 Hz
        sos_highpass = signal.butter(4, 20, btype='high', fs=fs_sweep, output='sos')

        start_indices, end_indices = [], []
        irs_all = []

        date_created_all = []

        for file in tqdm(sorted(os.listdir(path)), desc=f"{room} files", disable=False, leave=True):
            # Select files for position 1 and 2, and LS1 and LS2
            if (
                file.endswith(".wav")
                and "sweep" in file
                and (
                    ("position1" in file and ("LS1" in file or "LS2" in file))
                    or ("position2" in file and ("LS1" in file or "LS2" in file))
                )
            ):
                # Extract position and loudspeaker from filename
                parts = file.split('_')
                position = parts[1]  # e.g., 'position1' or 'position2'
                loudspeaker = parts[2].replace('LS', 'source')  # e.g., 'LS1' or 'LS2'

                # Get date created from wav file
                wav_path = os.path.join(path, file)
                try:
                    stat = os.stat(wav_path)
                    date_created_all.append(pd.to_datetime(stat.st_birthtime, unit='s'))
                except Exception:
                    date_created_all.append(pd.NaT)

                # Extract coordinates for the current position and loudspeaker (source)
                position_coords_unity = geometry[geometry['prefix'] == position][['x', 'y', 'z']].values.flatten()
                source_coords_unity = geometry[geometry['prefix'] == loudspeaker][['x', 'y', 'z']].values.flatten()

                position_coords = convert_coords_unity_to_sofa(position_coords_unity)
                source_coords = convert_coords_unity_to_sofa(source_coords_unity)


                position_coords_all = np.vstack([position_coords_all, position_coords])
                source_coords_all = np.vstack([source_coords_all, source_coords])

                fs, sweep_response = wavfile.read(os.path.join(path, file))
                assert fs == fs_sweep, "Sampling frequency mismatch between se!"

                num_channels = sweep_response.shape[1]
                num_repeats = 3

                irs = np.zeros((num_repeats, num_channels, fs*sweep_len_s))

                for iCh in range(sweep_response.shape[1]):

                    ir = deconvolve_sweep(sweep.T, sweep_response[:, iCh].T)

                    ir = signal.sosfilt(sos_highpass, ir)

                    irs[:, iCh, :] = ir.reshape(-1, fs*sweep_len_s)[:3, :]

                irs_all.append(irs)

                most_similar_pair_idx = select_by_rule_of_two(irs)
                start_index, end_index = find_start_end_indices(irs[most_similar_pair_idx, :, :])

                start_indices.append(start_index)
                end_indices.append(end_index)

                irs_cut = irs[most_similar_pair_idx[0], :, :]

        start_index = np.min(start_indices)
        end_index = np.max(end_indices)

        # Stack all truncated IRs from irs_all into a single array with dimensions (num_measurements, num_channels, ir_length)
        irs_truncated = np.stack(
            [irs[most_similar_pair_idx[0], :, start_index:end_index] for irs in irs_all], axis=0
        )

        # normalize IRs with a fixed value
        overall_level_db = 20
        irs_truncated = irs_truncated * 10**(overall_level_db/20)

        sofa = sf.Sofa("SingleRoomSRIR")
        set_global_parameters(sofa, array)
        sofa.ListenerPosition = position_coords_all
        sofa.SourcePosition = source_coords_all
        sofa.Data_SamplingRate = fs
        sofa.Data_IR = irs_truncated
        sofa.Data_Delay = np.zeros(irs_truncated.shape[:2])
        # Convert pandas Timestamps to float (Unix timestamps), use NaN for missing values
        sofa.MeasurementDate = np.array([
            ts.timestamp() if pd.notnull(ts) else np.nan for ts in date_created_all
        ])

        sf.write_sofa(os.path.join(path_to_output, 'sofafiles', f"{array}_{room}_SRIR.sofa"), sofa)

        # plot the IRs for one position
        tax = np.arange(irs_truncated.shape[2]) / fs
        
        plt.figure()

        num_measurements = irs_truncated.shape[0]
        num_channels = irs_truncated.shape[1]
        
        fig, axes = plt.subplots(num_measurements, 1, figsize=(10, 3 * num_measurements ))

        for iPos in range(num_measurements):
            for ch in range(num_channels):
                axes[iPos].plot(tax, 10*np.log10(irs_truncated[iPos, ch, :]**2))
            axes[iPos].set_ylim([-100, 0])
            axes[iPos].set_xlabel("Time in s")
            axes[iPos].set_ylabel("Level in dB")
            axes[iPos].grid()
            axes[iPos].set_title(f"position_coords_all: {position_coords_all[iPos]}, source_coords_all: {source_coords_all[iPos]}")    
            axes[iPos].set_xlim([0, 1])
        plt.tight_layout()
        plt.savefig(os.path.join(path_to_output, 'plots', f"impulse_responses_{array}_{room}.png"), dpi=500)

        
        #plt.show()

        # Time axis (0 to 2 s, 1 ms step)

        # T60_1 = 0.4  # seconds
        # T60_2 = 0.6  # seconds

        # tau1 = T60_1 / np.log(1000.0)
        # tau2 = T60_2 / np.log(1000.0)

        # e1 = np.exp(-tax / tau1)
        # e2 = np.exp(-tax / tau2)

        # ratio = e1 / e2  

        # plt.plot(tax, 10*np.log10((irs_cut[0] * ratio)**2), color='cornflowerblue')
        # plt.plot(tax, 10*np.log10((irs_cut[1] * ratio) ** 2) , color='firebrick')
        # plt.ylim([-60, 0])
        # plt.xlabel("Time in s")
        # plt.ylabel("Normalized sound pressure in dB")
        # plt.grid()
        # plt.savefig("/Users/meyern2/Library/CloudStorage/OneDrive-AaltoUniversity/Funding/SPIRIT/midterm/impulse_responses_lower_rt.png", dpi=500)
        # plt.show()

        # start_reverb = int(np.ceil(0.0015 * fs))
        # irs_cut_drr_changed = []
        # irs_cut_drr_changed.append(irs_cut[0])
        # irs_cut_drr_changed.append(irs_cut[1])

        # irs_cut_drr_changed[0][start_reverb:] = irs_cut_drr_changed[0][start_reverb:]  * 10 ** (-10/ 20)
        # irs_cut_drr_changed[1][start_reverb:] = irs_cut_drr_changed[1][start_reverb:]  * 10 ** (-10/ 20)

        # plt.plot(tax, 10*np.log10(irs_cut_drr_changed[0] ** 2), color='cornflowerblue')
        # plt.plot(tax, 10*np.log10(irs_cut_drr_changed[1] ** 2) , color='firebrick')
        # plt.ylim([-60, 0])
        # plt.xlabel("Time in s")
        # plt.ylabel("Normalized sound pressure in dB")
        # plt.grid()
        # plt.savefig("/Users/meyern2/Library/CloudStorage/OneDrive-AaltoUniversity/Funding/SPIRIT/midterm/impulse_responses_lower_drr.png", dpi=500)
        # plt.show()


