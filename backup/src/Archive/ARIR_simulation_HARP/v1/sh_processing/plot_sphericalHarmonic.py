import numpy as np
import SphericalHarmonic as SH
import pyroomacoustics as pra

# Dummy-Orientierung (z.B. nach vorne)
orientation = pra.directivities.DirectionVector(
    azimuth=0,
    colatitude=np.pi/2,
    degrees=False
)

sh = SH.SphericalHarmonicDirectivity(orientation=orientation, n=3, m=-2)

azimuth = np.linspace(0, 360, 100)
colatitude = np.linspace(0, 180, 100)

sh.plot_response(azimuth, colatitude, degrees=True)

import matplotlib.pyplot as plt
plt.savefig("spherical_harmonic_response.png")
plt.show()