import numpy as np
from scipy.special import spherical_jn, spherical_yn
from spaudiopy.sph import sh_matrix


# ----------------------------
# Spherical Bessel helpers
# ----------------------------
def sph_besselj(n, x):
    return spherical_jn(n, x)


def dsph_besselj(n, x):
    # derivative w.r.t. x
    return spherical_jn(n, x, derivative=True)


def sph_hankel2(n, x):
    # spherical Hankel function of 2nd kind: jn - i yn
    return spherical_jn(n, x) - 1j * spherical_yn(n, x)


def dsph_hankel2(n, x):
    return dsph_besselj(n, x) - 1j * spherical_yn(n, x, derivative=True)


# ----------------------------
# replicatePerOrder
# ----------------------------
def replicate_per_order(x):
    """
    Replicates element n into 2n+1 copies along first axis.
    Equivalent to MATLAB replicatePerOrder.
    """
    N = len(x) - 1
    out = []
    for n in range(N + 1):
        out.append(np.repeat(x[n], 2 * n + 1))
    return np.concatenate(out, axis=0)


# ----------------------------
# sphModalCoeffs
# ----------------------------
def sph_modal_coeffs(N, kr, array_type="rigid", dir_coeff=0.0):
    kr = np.asarray(kr)
    b_N = np.zeros((len(kr), N + 1), dtype=complex)

    for n in range(N + 1):

        jn = sph_besselj(n, kr)

        if array_type == "open":
            b_N[:, n] = 4 * np.pi * (1j ** n) * jn

        elif array_type == "rigid":
            jnprime = dsph_besselj(n, kr)
            hn = sph_hankel2(n, kr)
            hnprime = dsph_hankel2(n, kr)

            temp = 4 * np.pi * (1j ** n) * (jn - (jnprime / hnprime) * hn)

            # handle kr = 0
            if n == 0:
                temp = np.where(kr == 0, 4 * np.pi, temp)
            else:
                temp = np.where(kr == 0, 0.0, temp)

            b_N[:, n] = temp

        elif array_type == "directional":
            jnprime = dsph_besselj(n, kr)
            temp = 4 * np.pi * (1j ** n) * (dir_coeff * jn - 1j * (1 - dir_coeff) * jnprime)
            b_N[:, n] = temp

        else:
            raise ValueError("Wrong array type")

    b_N[np.isnan(b_N)] = 0
    return b_N


# ----------------------------
# getSH
# ----------------------------
def getSH(N, angles_azi_zen, sh_type="real"):
    """
    Wrapper matching your expected interface.
    
    angles_azi_zen: (Nmic, 2) -> [azi, zen]
    """
    azi = angles_azi_zen[:, 0]
    zen = angles_azi_zen[:, 1]
    
    return sh_matrix(N, azi, zen, sh_type=sh_type)


# ----------------------------
# Main function
# ----------------------------
def arraySHTfiltersTheory_regLS(R, mic_dirsAziElev, order_sht, Lfilt, fs, amp_threshold):
    c = 343.0

    f = np.arange(0, Lfilt // 2 + 1) * fs / Lfilt
    kR = 2 * np.pi * f * R / c
    kR_max = kR[-1]

    order_array = min(30, int(np.floor(2 * kR_max)))

    mic_dirsAziElev = np.asarray(mic_dirsAziElev)
    Nmic = mic_dirsAziElev.shape[0]

    if order_sht > np.sqrt(Nmic) - 1:
        print("Warning: order too high for mic count")
        order_sht = int(np.floor(np.sqrt(Nmic) - 1))

    # convert elevation to inclination
    mic_dirsAziIncl = np.column_stack([
        mic_dirsAziElev[:, 0],
        np.pi / 2 - mic_dirsAziElev[:, 1]
    ])

    # SH matrix (placeholder: you must provide getSH equivalent)
    Y_array = getSH(order_array, mic_dirsAziIncl, "real") * np.sqrt(4 * np.pi)

    # modal responses
    bN = sph_modal_coeffs(order_array, kR, "rigid") / (4 * np.pi)

    # array response in SH domain
    H_array = np.zeros((Nmic, (order_array + 1) ** 2, len(f)), dtype=complex)

    for kk in range(len(f)):
        temp_b = bN[kk, :]
        B = np.diag(replicate_per_order(temp_b))
        H_array[:, :, kk] = Y_array @ B

    # Tikhonov regularization
    alpha = 10 ** (amp_threshold / 20)
    beta = 1 / (2 * alpha)

    H_filt = np.zeros(((order_sht + 1) ** 2, Nmic, len(f)), dtype=complex)

    for kk in range(len(f)):
        tempH = H_array[:, :, kk]
        tempH_trunc = tempH[:, : (order_sht + 1) ** 2]

        H_filt[:, :, kk] = (
            tempH_trunc.T
            @ np.linalg.inv(tempH @ tempH.T + (beta ** 2) * np.eye(Nmic))
        )

    # impulse response
    h_filt = None
    if True:
        h = H_filt.copy()

        # enforce Hermitian symmetry
        h[:, :, -1] = np.abs(h[:, :, -1])
        h = np.concatenate(
            [h, np.conj(h[:, :, -2:0:-1])],
            axis=2
        )

        h = np.real(np.fft.ifft(h, axis=2))
        h = np.fft.fftshift(h, axes=2)

        h_filt = h

    return H_filt, h_filt