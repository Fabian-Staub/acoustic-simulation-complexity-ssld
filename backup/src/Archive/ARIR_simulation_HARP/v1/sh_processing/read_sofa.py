import sofar as sf
import numpy as np

sofa_path = "/scratch/elec/t412-techpsychoacoustics/msc-thesis/2026_fstaub/staubf1/Projects/Master_Thesis/src/HARP-main/v1/em32_test001_20260702.sofa"

sofa = sf.read_sofa(sofa_path, verify=False)

print("\n================ BASIC INFO ================\n")

print("SOFA Convention:")
print(getattr(sofa, "GLOBAL_SOFAConventions", "N/A"))

print("\nSOFA Version:")
print(getattr(sofa, "GLOBAL_SOFAConventionsVersion", "N/A"))

print("\nData Type:")
print(getattr(sofa, "GLOBAL_DataType", "N/A"))

print("\nRoom Type:")
print(getattr(sofa, "GLOBAL_RoomType", "N/A"))

print("\nSampling Rate:")
print(getattr(sofa, "Data_SamplingRate", "N/A"))

print("\n================ IMPULSE RESPONSES ================\n")

rir = sofa.Data_IR

print("Data_IR shape:", rir.shape)

if len(rir.shape) == 3:
    M, R, N = rir.shape

    print(f"Measurements (M): {M}")
    print(f"Receivers    (R): {R}")
    print(f"Samples      (N): {N}")

print("\n================ SOURCE POSITIONS ================\n")

if hasattr(sofa, "SourcePosition"):
    src = sofa.SourcePosition

    print("Shape:", src.shape)
    print("Type:", getattr(sofa, "SourcePosition_Type", "Unknown"))
    print("Units:", getattr(sofa, "SourcePosition_Units", "Unknown"))

    print("\nFirst 10 source positions:")
    print(src[:10])

else:
    print("SourcePosition not found")

print("\n================ RECEIVER POSITIONS ================\n")

if hasattr(sofa, "ReceiverPosition"):
    rec = sofa.ReceiverPosition

    print("Shape:", rec.shape)
    print("Type:", getattr(sofa, "ReceiverPosition_Type", "Unknown"))
    print("Units:", getattr(sofa, "ReceiverPosition_Units", "Unknown"))

    print("\nReceiver positions:")
    print(rec)

else:
    print("ReceiverPosition not found")

print("\n================ LISTENER POSITIONS ================\n")

if hasattr(sofa, "ListenerPosition"):
    listener = sofa.ListenerPosition

    print("Shape:", listener.shape)
    print("Type:", getattr(sofa, "ListenerPosition_Type", "Unknown"))
    print("Units:", getattr(sofa, "ListenerPosition_Units", "Unknown"))

    print(listener)

else:
    print("ListenerPosition not found")

print("\n================ LISTENER ORIENTATION ================\n")

if hasattr(sofa, "ListenerView"):
    print("ListenerView:")
    print(sofa.ListenerView)

if hasattr(sofa, "ListenerUp"):
    print("\nListenerUp:")
    print(sofa.ListenerUp)

print("\n================ EM32 DIRECTIVITY CHECK ================\n")

try:
    src = sofa.SourcePosition

    print("Number of source measurements:", src.shape[0])

    if hasattr(sofa, "ReceiverPosition"):
        print("Number of microphones:", sofa.ReceiverPosition.shape[0])

except Exception as e:
    print("Could not evaluate:", e)

print("\n================ ALL AVAILABLE ATTRIBUTES ================\n")

for attr in sorted(dir(sofa)):
    if not attr.startswith("_"):
        try:
            value = getattr(sofa, attr)

            if isinstance(value, np.ndarray):
                print(f"{attr:35s} {value.shape}")
            else:
                print(f"{attr:35s} {type(value).__name__}")

        except Exception:
            pass

print("\n================ RECEIVER POSITIONS (DEGREES) ================\n")

if hasattr(sofa, "ReceiverPosition"):
    rec = np.squeeze(np.array(sofa.ReceiverPosition))

    print("Shape:", rec.shape)
    print("Type:", getattr(sofa, "ReceiverPosition_Type", "Unknown"))
    print("Units:", getattr(sofa, "ReceiverPosition_Units", "Unknown"))

    print("\nAll receiver positions:\n")

    # Expected format: (M, 3) = [azimuth, elevation, radius]
    if rec.ndim == 2 and rec.shape[1] == 3:
        for i, r in enumerate(rec):
            az = r[0]
            el = r[1]
            radius = r[2]

            print(
                f"Mic {i:02d}: "
                f"azimuth={az:7.2f}°, "
                f"elevation={el:7.2f}°, "
                f"radius={radius:.4f} m"
            )

    else:
        print("Unexpected format:")
        print(rec)

else:
    print("ReceiverPosition not found")





















import sofar as sf
import numpy as np

sofa_path = "eigenmike32_simulated.sofa"
sofa = sf.read_sofa(sofa_path, verify=False)

# ================= SOURCE POSITIONS =================
print("\n================ SOURCE POSITIONS ================\n")

if hasattr(sofa, "SourcePosition"):
    src = np.squeeze(np.array(sofa.SourcePosition))

    print("Shape:", src.shape)
    print("Units:", getattr(sofa, "SourcePosition_Units", "Unknown"))

    print("\nAll source positions (azimuth°, elevation°, radius):\n")

    if src.ndim == 2 and src.shape[1] == 3:
        for i, s in enumerate(src):
            az, el, r = s
            print(f"Src {i:03d}: azimuth={az:8.2f}°, elevation={el:8.2f}°, radius={r:.3f} m")
    else:
        print(src)

else:
    print("SourcePosition not found")


# ================= RECEIVER POSITIONS =================
print("\n================ RECEIVER POSITIONS ================\n")

if hasattr(sofa, "ReceiverPosition"):
    rec = np.squeeze(np.array(sofa.ReceiverPosition))

    print("Shape:", rec.shape)
    print("Units:", getattr(sofa, "ReceiverPosition_Units", "Unknown"))

    print("\nAll receiver positions (azimuth°, elevation°, radius):\n")

    if rec.ndim == 2 and rec.shape[1] == 3:
        for i, r in enumerate(rec):
            az, el, rdist = r
            print(f"Mic {i:02d}: azimuth={az:8.2f}°, elevation={el:8.2f}°, radius={rdist:.3f} m")
    else:
        print(rec)

else:
    print("ReceiverPosition not found")