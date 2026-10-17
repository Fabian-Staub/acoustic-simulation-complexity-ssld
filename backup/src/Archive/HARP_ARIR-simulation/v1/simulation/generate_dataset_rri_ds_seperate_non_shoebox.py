from anechoic_noise import NonStationaryInterference
from pathlib import Path
import pyroomacoustics as pra
import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt
import os
import soundfile as sf
import json
from utils import cart2sph
import argparse
import sys

from pyroomacoustics.directivities import (
    DirectivityPattern,
    DirectionVector,
    CardioidFamily,
)

def getMixingTimeEstimateFromVolume(volume):
    # Cremer, L.; Mueller, H. A. (1978): Die wissenschaftlichen Grundlagen der Raumakustik
    return 2 * np.sqrt(volume) / 1000.0

def addStochasticLateReverb(srir, rt, fs, mixingTime, volume):
        '''
        addIsotropicLateReverb add a diffuse reverberant tail

        '''

        timeConstants = rt * fs / 6.9078

        irLength_samp = srir.shape[0]
        numNoiseSources = srir.shape[1]

        noise = np.random.normal(0, 1, size = (irLength_samp, numNoiseSources))

        path = os.path.join(os.path.dirname(__file__), 'octaveFilters.json')
        with open(path) as f:
            octaveFilters = json.load(f)
            filtersIrs = np.asarray(octaveFilters['ir'])

        numFilterBands = filtersIrs.shape[1]
        noisePerSourceBand = np.zeros((irLength_samp, numNoiseSources, numFilterBands))

        tax = np.arange(irLength_samp)

        if volume:
            mixingTime_samp = int(getMixingTimeEstimateFromVolume(volume) * fs)
        else:
            mixingTime_samp = int(mixingTime * fs)

        fadeLength_samp = int(15.0 / 1000.0 * fs)

        if irLength_samp - mixingTime_samp - fadeLength_samp > 0:
            window = np.hstack((np.zeros(mixingTime_samp),
                                np.sin(np.arange(fadeLength_samp) / fadeLength_samp * np.pi / 2) ** 2,
                                np.ones(irLength_samp - mixingTime_samp - fadeLength_samp)))
        else:
            window = np.ones(irLength_samp)
            print('no late reverb added - response too short')


        for iBand in range(numFilterBands):
            envelope = np.exp(-tax / timeConstants[iBand])
            for iNoiseSource in range(numNoiseSources):
                noisePerSourceBand[:, iNoiseSource, iBand] = np.multiply(
                    np.convolve(filtersIrs[:, iBand], noise[:, iNoiseSource], mode = 'same'), envelope)

        noisePerSource = np.sum(noisePerSourceBand, axis = 2)

        # compute the rms of the simulated response and match the noise response
        # (needs sufficient IS order to deliver reasonable result)

        rmsIs = np.sqrt(np.sum(srir[mixingTime_samp:, :] ** 2, axis = 0))
        rmsNoise = np.sqrt(np.sum(noisePerSource[mixingTime_samp:, :] ** 2, axis = 0))

        noiseLate = np.multiply(noisePerSource, np.expand_dims(rmsIs / rmsNoise, axis= (0)))

        srir = srir * np.expand_dims(1-window , axis = 1 ) + noiseLate  * np.expand_dims(window , axis = 1 )

        return srir


# note that for each instance we should create 2 pairs of signals coming from the same room and the same signals coming from two different rooms.

def generate_example_rri(num_instance, nsi, ADDLATE = False):
#generate data, inspired by real room data

    rtProfiles = np.load('rtProfilesMit.npy')
    centerFreqs = 125 * 2**np.arange(0, 8)
    numRtProfiles = rtProfiles.shape[0]
    numRtBands = rtProfiles.shape[0]

    #create new folder with num instance

    #path = '/home/frls/impulse_response_estimation/generated_nils_positives'
    path = '/home/frls/impulse_response_estimation/generated_data_with_late'

    # frequency sample
    fs = 48000

    # speed of sound in m/s
    c = 343

    # minimum distance between source and receiver
    min_dist_m = 0.5
    min_toa_samp = np.round(min_dist_m / c * fs).astype(np.int32)

    # this is how much the floor plan is changed relative to a shoebox
    max_displacement_m = 0.5

    # how far sources and receivers must be away from the ceiling
    dist_from_ceiling_m = 1

    # how many sources in include
    sample_density_1 = np.random.randint(1, 4) #high number is exlusive, so number of sources from 1 to 3.
    print('sample density: '+str(sample_density_1))
    # GET ANECHOIC SIGNALS!
    anechoic_signals_1 = []
    for num_signal in range(sample_density_1):
        sig = nsi.generate_signal(duration=4, sample_density=1)
        anechoic_signals_1.append(sig)


    sample_density_2 = np.random.randint(1, 4) #high number is exlusive, so number of sources from 1 to 3.
    anechoic_signals_2 = []
    for num_signal in range(sample_density_2):
        sig = nsi.generate_signal(duration=4, sample_density=1)
        anechoic_signals_2.append(sig)

    sample_densities = []
    sample_densities.append(sample_density_1)
    sample_densities.append(sample_density_2)

    output_dir = os.path.join(path, '{:05d}'.format(num_instance))
    Path(output_dir).mkdir(parents = True, exist_ok = True)

    metadata = {'0': {}, '1': {}}

    anisotropic = np.random.rand() > 0.5
    anisotropicType = np.random.rand() > 0.5 # should it be more or less absorption

    while True:
        # Generate room dimensions
        aux = np.random.uniform(1.5, 20, size=2), np.random.uniform(2.5, 8, size=1)
        room_dim = np.concatenate(aux)
        print('room_dim')
        print(room_dim)
        #volume = room_dim[0] * room_dim[1] * room_dim[2]

        rtProfileIdx = np.random.randint(0, numRtProfiles)
        rt60 = rtProfiles[rtProfileIdx, :]

        try:

            # use a shoebox as a base
            roomShoebox = pra.ShoeBox(
                room_dim, fs=fs,
                use_rand_ism = False, max_order=0
            )
            floor_corners_shoebox = roomShoebox.walls[4].corners # corners of floor

            # and displace corners before creating 3D room from it
            floor_corners = floor_corners_shoebox[:2, :] + np.random.rand(2, 4) * max_displacement_m

            # now create a dummy room with those dimensions
            room = pra.Room.from_corners(floor_corners,
                                     max_order = 0, fs=fs)

            room.extrude(room_dim[2]) # "add ceiling"

            room.set_sound_speed(c)

            volume = room.get_volume()

            # mean absorption -  should replace with actual room volume ...
            e_absorption = np.array([pra.inverse_sabine(rtOctaveband, room_dim, c = c)[0] for rtOctaveband in rt60])

            # random anisotropy index between 0.5 and 1.5,
            # which means 50% more or less absorption on one of the walls
            #anisotropy_index = np.random.rand() + 0.5

            if  anisotropic: # for half the samples, make
                anisotropy_index  = np.random.rand() * 5 + 5 # between 5 and 15 times
                if anisotropicType:
                     anisotropy_index  = 1 / anisotropy_index # more or less absorption
            else: # for the other half, don't
                anisotropy_index = 1

            # random wall to have a different absorption
            idxDifferentWall = np.random.randint(0, 6)

            # get surface are as
            area = [room.wall_area(room.walls[iWall]) for iWall in range(6)]

            area_all = np.sum(np.array(area))
            area_selected_wall = np.array(area[idxDifferentWall])
            area_rest = area_all - area_selected_wall

            e_absorption_rest =  e_absorption * area_rest / (anisotropy_index * area_selected_wall + area_rest)
            e_absorption_selected_wall = anisotropy_index * e_absorption_rest

            print(e_absorption_rest)
            print(e_absorption_selected_wall)

            concat_absorption_values = np.hstack((e_absorption_rest, e_absorption_selected_wall))

            if not(any(concat_absorption_values < np.array(0.01)) | any(concat_absorption_values > np.array(0.99))):
                break

        except ValueError:
            print('need to find a different room size')

    #Invert Sabine's formula to obtain the parameters for the ISM simulator
    #e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)

    material_properties = [ {'description' : 'uniform_material', 'coeffs' : e_absorption_rest, 'center_freqs' : centerFreqs } for iWall in range(6)]
    material_properties[idxDifferentWall] =  {'description' : 'uniform_material', 'coeffs' : e_absorption_selected_wall, 'center_freqs' : centerFreqs}

    #m = pra.make_materials(
    #    ceiling = material_properties[0],
    #    floor = material_properties[1],
    #    east = material_properties[2],
    #    west = material_properties[3],
    #    north = material_properties[4],
    #    south = material_properties[5],
    #)

    for j in range(2):

        # create wall materials
        materials_side_walls = pra.make_materials(*[(material_properties[i],) for i in range(4)])
        # create ceiling and floor material ceiling and floor
        me = pra.make_materials(floor=(material_properties[4],), ceiling=(material_properties[5],))

        # create the room again, with materals
        room = pra.Room.from_corners(floor_corners,
                                     max_order = 12, materials=materials_side_walls, fs=fs)
        room.extrude(room_dim, materials = me)

        # create the room again, with materals
        room_ds = pra.Room.from_corners(floor_corners,
                                     max_order = 0, materials=materials_side_walls, fs=fs)
        room_ds.extrude(room_dim, materials = me)

        direct_toa_samples = np.zeros(sample_density_1).astype(np.int32)

        while(np.any(direct_toa_samples < min_toa_samp)): # make sure that minimum distance is kept
            # Generate source locations
            x_pos = np.random.uniform(0+max_displacement_m, room_dim[0]-max_displacement_m, size=sample_densities[j]+1) #The plus one is for the source position
            y_pos = np.random.uniform(0+max_displacement_m, room_dim[1]-max_displacement_m, size=sample_densities[j]+1) #The plus one is for the source position
            z_pos = np.random.uniform(0 + dist_from_ceiling_m, room_dim[2] - dist_from_ceiling_m, size=sample_densities[j]+1) #The plus one is for the source position

            all_pos = np.column_stack((x_pos, y_pos, z_pos))
            sources_pos = all_pos[:-1]
            receiver_pos = all_pos[-1] #assuming single receiver (for now)

            direct_toa_samples = np.round(np.linalg.norm(sources_pos - receiver_pos, axis = 1) / room.c * room.fs).astype(np.int32)

        metadata[str(j)]['rt60'] = rt60.tolist()
        metadata[str(j)]['absorption_selected_wall'] = e_absorption_selected_wall.tolist()
        metadata[str(j)]['absorption_rest'] = e_absorption_rest.tolist()
        metadata[str(j)]['selected_wall_index'] = idxDifferentWall
        metadata[str(j)]['volume'] = volume
        metadata[str(j)]['room_dim'] = room_dim.tolist()
        metadata[str(j)]['sources_pos'] = sources_pos.tolist()
        metadata[str(j)]['receiver_pos'] = receiver_pos.tolist()

        #add sources
        for source_num, source_pos in enumerate(sources_pos):
            #room.add_source(position=source_pos, signal=anechoic_signals[source_num])
            if j:
                room.add_source(position=source_pos, signal=anechoic_signals_2[source_num])
                room_ds.add_source(position=source_pos, signal=anechoic_signals_2[source_num])
            else:
                room.add_source(position=source_pos, signal=anechoic_signals_1[source_num])
                room_ds.add_source(position=source_pos, signal=anechoic_signals_1[source_num])


        # microphone array pointing directions
        mic_array_normals = np.array([[0.5774 ,   0.5774 ,   0.5774 ], # right front up
            [0.5774,   -0.5774 ,  -0.5774 ], # right left down
            [-0.5774 ,   0.5774 ,  -0.5774] , # back right down
            [-0.5774,   -0.5774,    0.5774]]) # back left up

        mic_array_radius = 0.02 # same as SPS 200
        mic_locs = mic_array_normals * mic_array_radius + receiver_pos

        # make directivities - four cardioids
        dir_objects = []
        for iMic in range(mic_array_normals.shape[0]):
            azizen = cart2sph(mic_array_normals[iMic, :])
            dir_objects.append(CardioidFamily(
                orientation=DirectionVector(azimuth=azizen[0], colatitude=azizen[1], degrees=False),
                pattern_enum=DirectivityPattern.CARDIOID))

        # omni position in the center
        mic_locs = np.vstack((mic_locs, receiver_pos))

         # make directivities - one omni in the center
        dir_objects.append(CardioidFamily(
            orientation=DirectionVector(azimuth=0, colatitude=0, degrees=True),
            pattern_enum=DirectivityPattern.OMNI))

        num_mics_in_array = mic_locs.shape[0]

        #  place the mic array in the room
        room.add_microphone_array(mic_locs.T, dir_objects) #SHOULD WE PUT FS HERE?????????????????????????????
        room_ds.add_microphone_array(mic_locs.T, dir_objects) #SHOULD WE PUT FS HERE?????????????????????????????

        # simulate
        room.compute_rir()
        room_ds.compute_rir()

        for num_source in range(sample_densities[j]):

            # create a mulichannel spatial room impulse response
            rir_lens_samp = np.array([room.rir[iMic][num_source].shape[0] for iMic in range(num_mics_in_array)])
            rir_lens_samp = rir_lens_samp - direct_toa_samples[num_source]
            max_len_samp = np.max(rir_lens_samp)
            srir = np.zeros((max_len_samp, num_mics_in_array))
            for iMic in range(num_mics_in_array):
                srir[:rir_lens_samp[iMic], iMic] = room.rir[iMic][num_source][direct_toa_samples[num_source]:]

            # if desired, generate late reverb
            if ADDLATE:
                srir = addStochasticLateReverb(srir, rt60, fs, False, volume)

            # direct sound response
            ds_lens_samp = np.array([room_ds.rir[iMic][num_source].shape[0] for iMic in range(num_mics_in_array)])
            ds_lens_samp = ds_lens_samp - direct_toa_samples[num_source]
            srir_ds = np.zeros((max_len_samp, num_mics_in_array))
            for iMic in range(num_mics_in_array):
                srir_ds[:ds_lens_samp[iMic], iMic] = room_ds.rir[iMic][num_source][direct_toa_samples[num_source]:]

            # subtract direct sound to obtain only reverb
            srir_reverb = srir - srir_ds

            # get normalization from the omni direction sound and some
            # additional attenuation to avoid clipping
            normalization = 10**(-20/20) / np.sqrt(np.sum(srir_ds[:, 4]**2))

            #plt.figure()
            #plt.plot(srir[1:5000, 4])
            #plt.plot(srir_ds[1:5000, 4], 'k-', alpha=0.4)
            #plt.plot(srir_reverb[1:5000, 4], 'r-', alpha=0.4)
            #plt.show()

            #room.plot(plot_directivity=False)

            print('max value ' + str(np.max(np.abs(srir))))

            out_file = os.path.join(output_dir, 'rir_source_{:01d}_from_mixture_{:02d}_srir.wav'.format(num_source, j))
            print(f"Writing file: {out_file}")
            sf.write(out_file, srir * normalization, samplerate=fs) #maybe need to cut it after 5.7 seconds or something

            out_file = os.path.join(output_dir, 'rir_source_{:01d}_from_mixture_{:02d}_ds.wav'.format(num_source, j))
            print(f"Writing file: {out_file}")
            sf.write(out_file, srir_ds * normalization, samplerate=fs) #maybe need to cut it after 5.7 seconds or something

            out_file = os.path.join(output_dir, 'rir_source_{:01d}_from_mixture_{:02d}_rev.wav'.format(num_source, j))
            print(f"Writing file: {out_file}")
            sf.write(out_file, srir_reverb * normalization, samplerate=fs) #maybe need to cut it after 5.7 seconds or something

        room.simulate()

        out_file = os.path.join(output_dir, 'mixture_{:02d}_positive.wav'.format(j))
        print(f"Writing file: {out_file}")
        sf.write(out_file, room.mic_array.signals.T, samplerate=fs) #maybe need to cut it after 5.7 seconds or something

        #print('metadata')
        #print(metadata)

        metadata[str(j)]['normalization'] = normalization

        metadata_file = os.path.join(output_dir, "metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent = 4)

    return None

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", help="when running on multiple instances, this is the nr of examples in a patch", type=int, default=10000)
parser.add_argument("--batch_index", help="when running on multiple instances, this is the index of the instance", type=int, default=0)

args = parser.parse_args()

# AID dataset path
AID_path = "./AID/wavs/"
nsi = NonStationaryInterference(data_dir=AID_path)

for num_instance in range(args.batch_size * args.batch_index, args.batch_size * (args.batch_index+1)):
    print(num_instance)
    generate_example_rri(num_instance, nsi, ADDLATE = True)

