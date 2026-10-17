import torch

path = "/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/data/harp/R4000_P8_S3_Order3_24kHz_shoeBox_rtProfilesMIT_ISMOrder30/hopiv_accdoa_mel128_TIMIT_babble_test/pt/room_0000-0009.pt"

data = torch.load(
    path,
    map_location="cpu",
    weights_only=False,
)

print(type(data))

if isinstance(data, dict):
    print("Old format:", data.keys())
elif isinstance(data, (tuple, list)):
    print("Tuple length:", len(data))
    for i, value in enumerate(data):
        print(i, type(value), getattr(value, "shape", None))