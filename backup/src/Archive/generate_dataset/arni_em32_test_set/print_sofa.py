import numpy as np
import sofar as sf

# Load SOFA
sofa = sf.read_sofa("../../../data/arni_em32/sofa/arni_em32_reverberant_S1.sofa", verify=False)

# Extract data
listener_pos = sofa.ListenerPosition      # (M, 3)
source_pos   = sofa.SourcePosition        # (M, 3)
listener_view = sofa.ListenerView         # (M, 3)
listener_up   = sofa.ListenerUp           # (M, 3)

M = listener_pos.shape[0]

azimuths = []
elevations = []

for m in range(M):
    # --- 1. Direction vector (listener → source)
    d = source_pos[m] - listener_pos[m]
    d = d / np.linalg.norm(d)

    # --- 2. Listener coordinate frame
    forward = listener_view[m] / np.linalg.norm(listener_view[m])
    up = listener_up[m] / np.linalg.norm(listener_up[m])
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)

    # (optional but safer: re-orthogonalize up)
    up = np.cross(right, forward)

    # --- 3. Project into listener frame
    x_local = np.dot(d, forward)
    y_local = np.dot(d, right)
    z_local = np.dot(d, up)

    # --- 4. Angles
    az = np.arctan2(y_local, x_local)
    el = np.arctan2(z_local, np.sqrt(x_local**2 + y_local**2))

    azimuths.append(np.degrees(az))
    elevations.append(np.degrees(el))

# Convert to arrays
azimuths = np.array(azimuths)
elevations = np.array(elevations)

# Print results
for i in range(M):
    print(
        f"Measurement {i}: "
        f"Az={azimuths[i]:.2f}°, El={elevations[i]:.2f}° | "
        f"LP={listener_pos[i]}, SP={source_pos[i]}, "
        f"LV={listener_view[i]}, LU={listener_up[i]}"
    )