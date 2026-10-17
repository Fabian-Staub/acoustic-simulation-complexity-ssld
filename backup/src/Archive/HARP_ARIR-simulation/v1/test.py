import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Load NPZ
# ============================================================
npz_path = "Eigenmike_em32_IRs.npz"

data = np.load(npz_path)

measurement_dirs = data["measurement_dirs_aziElev"]  # (1225, 2), radians
eigen_dirs = data["eigen_dirs_aziElev"]              # (32, 2), radians


# ============================================================
# [azimuth, elevation] -> Cartesian
#
# IMPORTANT:
# The NPZ already stores the angles in radians.
#
# azimuth:
#   0       -> +x
#   pi/2    -> +y
#
# elevation:
#   0       -> xy plane
#   +pi/2   -> +z
#   -pi/2   -> -z
# ============================================================
def azi_elev_to_cartesian(azi_elev, radius=1.0):
    azi_elev = np.asarray(azi_elev, dtype=float)

    az = azi_elev[:, 0]
    el = azi_elev[:, 1]

    cos_el = np.cos(el)

    x = radius * cos_el * np.cos(az)
    y = radius * cos_el * np.sin(az)
    z = radius * np.sin(el)

    return x, y, z


mx, my, mz = azi_elev_to_cartesian(measurement_dirs)
ex, ey, ez = azi_elev_to_cartesian(eigen_dirs)


# ============================================================
# Print ranges for verification
# ============================================================
print("Measurement directions:")
print(
    "  azimuth:",
    np.rad2deg(measurement_dirs[:, 0]).min(),
    "to",
    np.rad2deg(measurement_dirs[:, 0]).max(),
    "deg",
)
print(
    "  elevation:",
    np.rad2deg(measurement_dirs[:, 1]).min(),
    "to",
    np.rad2deg(measurement_dirs[:, 1]).max(),
    "deg",
)

print("\nEigenmike directions:")
print(
    "  azimuth:",
    np.rad2deg(eigen_dirs[:, 0]).min(),
    "to",
    np.rad2deg(eigen_dirs[:, 0]).max(),
    "deg",
)
print(
    "  elevation:",
    np.rad2deg(eigen_dirs[:, 1]).min(),
    "to",
    np.rad2deg(eigen_dirs[:, 1]).max(),
    "deg",
)


# ============================================================
# Create unit sphere
# ============================================================
az_sphere = np.linspace(0, 2 * np.pi, 120)
el_sphere = np.linspace(-np.pi / 2, np.pi / 2, 60)

AZ, EL = np.meshgrid(az_sphere, el_sphere)

sphere_x = np.cos(EL) * np.cos(AZ)
sphere_y = np.cos(EL) * np.sin(AZ)
sphere_z = np.sin(EL)


# ============================================================
# Plot
# ============================================================
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")

# transparent sphere
ax.plot_surface(
    sphere_x,
    sphere_y,
    sphere_z,
    alpha=0.08,
    linewidth=0,
    antialiased=True,
)

# measurement directions
ax.scatter(
    mx,
    my,
    mz,
    s=8,
    alpha=0.45,
    label=f"Measurement directions ({len(measurement_dirs)})",
)

# Eigenmike capsule directions
ax.scatter(
    ex,
    ey,
    ez,
    s=70,
    marker="x",
    linewidths=2,
    label=f"Eigenmike capsules ({len(eigen_dirs)})",
)


# ============================================================
# Coordinate axes
# ============================================================
axis_length = 1.25

ax.quiver(
    0, 0, 0,
    axis_length, 0, 0,
    arrow_length_ratio=0.08,
)
ax.quiver(
    0, 0, 0,
    0, axis_length, 0,
    arrow_length_ratio=0.08,
)
ax.quiver(
    0, 0, 0,
    0, 0, axis_length,
    arrow_length_ratio=0.08,
)

ax.text(axis_length, 0, 0, "+X")
ax.text(0, axis_length, 0, "+Y")
ax.text(0, 0, axis_length, "+Z")


# ============================================================
# Formatting
# ============================================================
limit = 1.3

ax.set_xlim(-limit, limit)
ax.set_ylim(-limit, limit)
ax.set_zlim(-limit, limit)

ax.set_box_aspect((1, 1, 1))

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.set_title("EM32 Measurement and Microphone Directions")
ax.legend()

# Nice initial camera angle
ax.view_init(elev=25, azim=35)

plt.tight_layout()


# ============================================================
# Save
# ============================================================
output_path = "em32_directions_sphere.png"

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print(f"\nSaved plot to: {output_path}")