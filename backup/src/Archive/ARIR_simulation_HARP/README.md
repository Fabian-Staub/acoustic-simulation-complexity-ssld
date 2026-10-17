# HARP HOA Dataset
HARP: A Large-Scale Higher-Order Ambisonic Room Impulse Response Dataset

Accepted at ICASSP GenDA 2025 workshop.

Provided code for generating HOA RIRs using ISM. 
This is first version of HARP that uses SphericalHarmonicDirectivity class added to Pyroomacoustics library to generate 7th order HOA.
The details can be found in the paper.

If requirements are met, just run Generate.py to generate RIRs as follow:

      python Generate.py --output_path "./Generated_HOA_IRs/" --num_rooms 10000 --num_positions 10 --ambi_order 7 --sample_rate 48000

Since the data generated for 100,000 RIRs is more than 1TB, it might take some time to be uploaded.

In Addition, V2 is provided that uses Ray Tracing in Complex Geometries using a spherical microphone array, raw RIRs can be saved for now, the same command can be used inside V2 directory to generate RIRs.

Cite this work: 

@inproceedings{saini2025harp,
  title={HARP: A Large-Scale Higher-Order Ambisonic Room Impulse Response Dataset},
  author={Saini, Shivam and Peissig, Juergen},
  booktitle={2025 IEEE International Conference on Acoustics, Speech, and Signal Processing Workshops (ICASSPW)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}
