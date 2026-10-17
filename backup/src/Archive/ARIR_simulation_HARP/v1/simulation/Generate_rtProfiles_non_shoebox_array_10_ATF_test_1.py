import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
from scipy.io import wavfile

from pyroomacoustics.directivities import MeasuredDirectivityFile, Rotation3D


# -----------------------------
# CONFIG
# -----------------------------
fs = 24000
room_dim = np.array([10,10,10])
R = 0.042


# -----------------------------
# LOAD DIRECTIVITIES (OK)
# -----------------------------
def load_em32(fs):
    print("[DEBUG] Loading EM32 SOFA directivities...")

    em32 = MeasuredDirectivityFile(
        path="EM32_Directivity.sofa",
        fs=fs,
        interp_order=12,
        interp_n_points=1000,
    )

    orientation = Rotation3D([0, 0, 0], "zyx", degrees=True)

    directivities = []
    for i in range(32):
        d = em32.get_mic_directivity(i, orientation=orientation)
        directivities.append(d)
        print(f"[OK] Loaded mic directivity {i}")

    return directivities


# -----------------------------
# MIC ARRAY GEOMETRY
# -----------------------------
def build_mic_array():
    print("[DEBUG] Building mic geometry...")

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
    colat = np.pi / 2 - el

    mic_positions = np.vstack([
        R * np.sin(colat) * np.cos(az),
        R * np.sin(colat) * np.sin(az),
        R * np.cos(colat)
    ])

    print("[DEBUG] mic_positions shape:", mic_positions.shape)
    return mic_positions


# -----------------------------
# ROOM
# -----------------------------
def create_room():
    room = pra.ShoeBox(
        p=room_dim,
        materials=pra.Material(0.1),
        fs=fs,
        max_order=20
    )

    room.add_source([2.5, 2.0, 1.5])
    return room



def sample_sphere(n_az=80, n_el=40):
    az = np.linspace(0, 2*np.pi, n_az)
    el = np.linspace(-np.pi/2, np.pi/2, n_el)
    az, el = np.meshgrid(az, el)
    return az, el

def plot_directivity_3d(directivity, title="Directivity"):
    az, el = sample_sphere()

    # unit sphere directions
    x = np.cos(el) * np.cos(az)
    y = np.cos(el) * np.sin(az)
    z = np.sin(el)

    dirs = np.stack([x, y, z], axis=-1)

    # evaluate response (robust handling)
    try:
        resp = directivity(dirs)
    except Exception:
        try:
            resp = np.vectorize(lambda a, e: directivity(a, e))(az, el)
        except Exception:
            resp = np.ones_like(az)

    resp = np.abs(resp)
    resp /= (resp.max() + 1e-12)

    # map to radius
    X = resp * x
    Y = resp * y
    Z = resp * z

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, alpha=0.9)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.tight_layout()
    plt.savefig(title, dpi=300)


# -----------------------------
# MAIN
# -----------------------------
def main():
    print("\n========== DEBUG START ==========\n")

    room = pra.ShoeBox(
        p=room_dim,
        materials=pra.Material(0.1),
        fs=fs,
        max_order=20
    )
 
    room.add_source([5.5, 9, 5])

    mic_positions = build_mic_array()

    # shift mic array into room coordinates
    mic_positions = mic_positions + room_dim[:, None] / 1.5

    print("[DEBUG] shifted mic center into room")

    # OPTIONAL: sanity check
    assert np.all(mic_positions[0] > 0) and np.all(mic_positions[0] < room_dim[0])
    assert np.all(mic_positions[1] > 0) and np.all(mic_positions[1] < room_dim[1])
    assert np.all(mic_positions[2] > 0) and np.all(mic_positions[2] < room_dim[2])

    directivities = load_em32(fs)

    sofa = directivities.sofa

    print("\n===== SOFA ROOT =====")
    print(type(sofa))
    print("\n===== DIMENSIONS =====")
    print(sofa.dimensions)
    print("\n===== GLOBAL ATTRIBUTES =====")
    for k, v in sofa.attributes.items():
        print(f"{k}: {v}")

    print("\n===== KEY DATA SHAPES =====")

    print("DataIR:", sofa.DataIR.shape)
    print("SourcePosition:", sofa.SourcePosition.shape)
    print("ReceiverPosition:", sofa.ReceiverPosition.shape)
    print("\n===== SINGLE IR CHECK =====")
    print("Example IR shape:", sofa.DataIR[0, 0].shape)




    #plot_directivity_3d(directivities[0], "EM32 Mic 0")


    # IMPORTANT: do NOT pass directivities yet (known fragile API)
    mic_array = pra.MicrophoneArray(mic_positions, fs, directivities)
    room.add_microphone_array(mic_array)

    print("[DEBUG] Running compute_rir...")

    room.image_source_model()
    room.compute_rir()

    rir = room.rir

    # Post-processing RIRs
    # Find maximum length across all channels to pad
    max_len = np.array([len(r[0]) for r in room.rir]).max()
    # Determine target length for padding (pad to the nearest second)
    target_len = int(np.ceil(max_len / room.fs) * room.fs)

    # Stack and pad RIRs
    RIR = np.stack(      
        [np.pad(r[0], (0, target_len - r[0].shape[0])) for r in room.rir],
        axis=0, # Stack along a new dimension (channels)
    )

    wavfile.write("em32_sim.wav", room.fs, RIR.astype(np.float32).T)

    print("\n========== RESULTS ==========")
    print("Sources:", len(rir))
    print("Mics per source:", len(rir[0]))
    print("Samples:", len(rir[0][0]))


if __name__ == "__main__":
    main()