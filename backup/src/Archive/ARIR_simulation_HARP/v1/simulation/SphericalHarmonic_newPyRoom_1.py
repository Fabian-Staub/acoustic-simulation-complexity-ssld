import numpy as np
import matplotlib.pyplot as plt

import pyroomacoustics as pra
from pyroomacoustics.directivities import DirectionVector, Directivity
from pyroomacoustics.utilities import all_combinations

from scipy.special import sph_harm
from spaudiopy.sph import sh_matrix


# =========================================================
# Spherical Harmonics Helper
# =========================================================
def spherical_harmonics(order, theta, phi):
    harmonics = []
    for n in range(order + 1):
        for m in range(-n, n + 1):
            Y_nm = sph_harm(m, n, phi, theta)
            harmonics.append(Y_nm)
    return np.array(harmonics)


# =========================================================
# HOA Array Generator
# =========================================================
def HOA_array(samples=25, radius=1.0, n_order=4):

    positions = np.zeros((samples, 3))
    orientations = []
    degrees = []

    offset = 2.0 / samples
    increment = np.pi * (3.0 - np.sqrt(5.0))

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2)
        r = np.sqrt(1 - y ** 2)

        phi = i * increment

        x = np.cos(phi) * r
        z = np.sin(phi) * r

        positions[i] = np.array([x, y, z]) * radius

        azimuth = np.arctan2(y, x)
        colatitude = np.arccos(z / radius)

        orientation = DirectionVector(azimuth, colatitude, degrees=False)
        orientations.append(orientation)

    # Assign SH degrees
    idx = 0
    for n in range(n_order + 1):
        for m in range(-n, n + 1):
            if idx < samples:
                degrees.append((n, m))
                idx += 1

    return positions, orientations, degrees


# =========================================================
# Spherical Harmonic Directivity
# =========================================================
class SphericalHarmonicDirectivity(Directivity):

    def __init__(self, orientation, n, m):
        super().__init__()
        self._orientation = orientation

        self._n = n
        self._m = m
        self._acn = n * (n + 1) + m

    # -----------------------------------------------------
    # REQUIRED by new pyroomacoustics
    # -----------------------------------------------------
    @property
    def is_impulse_response(self):
        return False

    @property
    def filter_len_ir(self):
        return 1

    def sample_rays(self, n_rays):
        # deterministic sampling (Fibonacci sphere)
        i = np.arange(n_rays)
        phi = (i * (np.pi * (3 - np.sqrt(5)))) % (2 * np.pi)
        cos_theta = 1 - (2 * i + 1) / n_rays
        theta = np.arccos(cos_theta)

        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)

        directions = np.vstack((x, y, z)).T

        gains = self.get_response(phi, theta, degrees=False)
        gains = np.abs(gains)

        return directions, gains

    # -----------------------------------------------------
    # SH RESPONSE
    # -----------------------------------------------------
    def get_response(
        self,
        azimuth,
        colatitude=None,
        magnitude=False,
        frequency=None,
        degrees=True,
    ):

        azimuth = np.asarray(azimuth)

        if degrees:
            azimuth = np.radians(azimuth)

        if colatitude is None:
            colatitude = np.pi / 2 * np.ones_like(azimuth)
        else:
            colatitude = np.asarray(colatitude)
            if degrees:
                colatitude = np.radians(colatitude)

        # broadcast shapes safely
        azimuth, colatitude = np.broadcast_arrays(azimuth, colatitude)

        # flatten for SH computation
        Y = sh_matrix(
            self._n,
            azimuth.flatten(),
            colatitude.flatten(),
            sh_type="real",
        )

        resp = Y[:, self._acn].reshape(azimuth.shape)

        return np.abs(resp) if magnitude else resp

    # -----------------------------------------------------
    # PLOTTING
    # -----------------------------------------------------
    def plot_response(
        self,
        azimuth,
        colatitude=None,
        degrees=True,
        ax=None,
        offset=None,
        axis_at_zero=False,
    ):

        if offset is not None:
            x_offset, y_offset, z_offset = (
                offset[0],
                offset[1],
                offset[2] if len(offset) > 2 else 0,
            )
        else:
            x_offset = y_offset = z_offset = 0

        if degrees:
            azimuth = np.radians(azimuth)
            if colatitude is not None:
                colatitude = np.radians(colatitude)

        if colatitude is not None:
            if ax is None:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")

            spher_coord = all_combinations(azimuth, colatitude)
            azi_flat = spher_coord[:, 0]
            col_flat = spher_coord[:, 1]

            resp = self.get_response(
                azimuth=azi_flat,
                colatitude=col_flat,
                magnitude=True,
                degrees=False,
            )

            RESP = resp.reshape(len(azimuth), len(colatitude))

            AZI, COL = np.meshgrid(azimuth, colatitude)

            X = RESP.T * np.sin(COL) * np.cos(AZI) + x_offset
            Y = RESP.T * np.sin(COL) * np.sin(AZI) + y_offset
            Z = RESP.T * np.cos(COL) + z_offset

            ax.plot_surface(X, Y, Z)

            if axis_at_zero:
                for v in range(3):
                    axis = np.zeros((2, 3))
                    axis[:, v] = [-1, 1]
                    ax.plot(axis[:, 0], axis[:, 1], axis[:, 2], "k-", linewidth=0.5)

            ax.set_title(f"SH Order {self._n}, Degree {self._m}")
            ax.axis("off")

        else:
            if ax is None:
                fig = plt.figure()
                ax = plt.subplot(111)

            resp = self.get_response(azimuth=azimuth, magnitude=True, degrees=False)

            X = resp * np.cos(azimuth) + x_offset
            Y = resp * np.sin(azimuth) + y_offset

            ax.plot(X, Y)
            ax.set_title(f"SH Order {self._n}, Degree {self._m}")

        return ax


# =========================================================
# TEST
# =========================================================
if __name__ == "__main__":

    positions, orientations, degrees = HOA_array(samples=25, n_order=4)

    # pick one mic
    n, m = degrees[12]
    d = SphericalHarmonicDirectivity(orientations[10], n, m)

    az = np.linspace(0, 360, 100)
    col = np.linspace(0, 180, 100)

    d.plot_response(az, col)
    plt.savefig("spherical_harmonic_response.png")