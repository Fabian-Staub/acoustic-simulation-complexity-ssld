import numpy as np
from scipy.special import spherical_jn, spherical_yn
from spaudiopy.sph import sh_matrix


# ----------------------------
# Safe spherical Bessel helpers
# ----------------------------
def sph_besselj(n, x):
    return spherical_jn(n, x)


def dsph_besselj(n, x):
    return spherical_jn(n, x, derivative=True)


def sph_hankel2(n, x):
    x = np.asarray(x)
    x_safe = np.where(np.abs(x) < 1e-12, 1e-12, x)
    return spherical_jn(n, x_safe) - 1j * spherical_yn(n, x_safe)


def dsph_hankel2(n, x):
    x = np.asarray(x)
    x_safe = np.where(np.abs(x) < 1e-12, 1e-12, x)
    return dsph_besselj(n, x_safe) - 1j * spherical_yn(n_safe, x_safe, derivative=True)


# ----------------------------
# replicatePerOrder
# ----------------------------
def replicate_per_order(x):
    N = len(x) - 1
    out = [np.repeat(x[n], 2 * n + 1) for n in range(N + 1)]
    return np.concatenate(out, axis=0)


# ----------------------------
# SH modal coefficients (stable)
# ----------------------------
def sph_modal_coeffs(N, kr, array_type="rigid", dir_coeff=0.0):

    kr = np.asarray(kr)
    kr_safe = np.where(np.abs(kr) < 1e-12, 1e-12, kr)

    b_N = np.zeros((len(kr_safe), N + 1), dtype=complex)

    for n in range(N + 1):

        jn = sph_besselj(n, kr_safe)

        if array_type == "open":
            b_N[:, n] = 4 * np.pi * (1j ** n) * jn

        elif array_type == "rigid":

            jnprime = dsph_besselj(n, kr_safe)
            hn = sph_hankel2(n, kr_safe)
            hnprime = dsph_hankel2(n, kr_safe)

            eps = 1e-12
            hnprime_safe = np.where(np.abs(hnprime) < eps, eps + 0j, hnprime)

            ratio = np.divide(
                jnprime,
                hnprime_safe,
                out=np.zeros_like(jnprime),
                where=np.abs(hnprime_safe) > eps
            )

            temp = 4 * np.pi * (1j ** n) * (jn - ratio * hn)

            # enforce k=0 behavior
            temp = np.where(kr == 0, 0.0 if n > 0 else 4 * np.pi, temp)

            b_N[:, n] = temp

        elif array_type == "directional":
            jnprime = dsph_besselj(n, kr_safe)
            temp = 4 * np.pi * (1j ** n) * (
                dir_coeff * jn - 1j * (1 - dir_coeff) * jnprime
            )
            b_N[:, n] = temp

        else:
            raise ValueError("Wrong array type")

    return np.nan_to_num(b_N, nan=0.0, posinf=0.0, neginf=0.0)


# ----------------------------
# SH matrix wrapper
# ----------------------------
def getSH(N, angles_azi_zen, sh_type="real"):
    azi = angles_azi_zen[:, 0]
    zen = angles_azi_zen[:, 1]
    return sh_matrix(N, azi, zen, sh_type=sh_type)


# ----------------------------
# Main function (STABLE VERSION)
# ----------------------------
def arraySHTfiltersTheory_regLS(
    R, mic_dirsAziElev, order_sht, Lfilt, fs, amp_threshold
):

    c = 343.0

    # ❗ remove DC (important fix)
    f = np.arange(1, Lfilt // 2 + 1) * fs / Lfilt

    kR = 2 * np.pi * f * R / c
    kR_max = kR[-1]

    order_array = min(30, int(np.floor(2 * kR_max)))

    mic_dirsAziElev = np.asarray(mic_dirsAziElev)
    Nmic = mic_dirsAziElev.shape[0]

    if order_sht > np.sqrt(Nmic) - 1:
        print("Warning: order too high for mic count")
        order_sht = int(np.floor(np.sqrt(Nmic) - 1))

    # convert elevation → inclination
    mic_dirsAziIncl = np.column_stack([
        mic_dirsAziElev[:, 0],
        np.pi / 2 - mic_dirsAziElev[:, 1]
    ])

    # SH encoding
    Y_array = getSH(order_array, mic_dirsAziIncl, "real") * np.sqrt(4 * np.pi)

    # modal responses
    bN = sph_modal_coeffs(order_array, kR, "rigid") / (4 * np.pi)

    H_array = np.zeros(
        (Nmic, (order_array + 1) ** 2, len(f)),
        dtype=complex
    )

    for kk in range(len(f)):

        temp_b = bN[kk, :]
        B = np.diag(replicate_per_order(temp_b))

        H_array[:, :, kk] = Y_array @ B

    # ----------------------------
    # Tikhonov regularization (stable solve)
    # ----------------------------
    alpha = 10 ** (amp_threshold / 20)
    beta = 1 / (2 * alpha)

    H_filt = np.zeros(
        ((order_sht + 1) ** 2, Nmic, len(f)),
        dtype=complex
    )

    I = np.eye(Nmic)

    for kk in range(len(f)):

        tempH = H_array[:, :, kk]
        tempH_trunc = tempH[:, : (order_sht + 1) ** 2]

        A = tempH @ tempH.T + (beta ** 2) * I

        # ❗ more stable than explicit inverse
        H_filt[:, :, kk] = np.linalg.solve(A, tempH_trunc).T

    # ----------------------------
    # impulse response reconstruction
    # ----------------------------
    h = H_filt.copy()

    # avoid forcing magnitude only (was unstable)
    h = np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)

    # Hermitian symmetry
    h = np.concatenate([h, np.conj(h[:, :, -2:0:-1])], axis=2)

    h = np.real(np.fft.ifft(h, axis=2))
    h = np.fft.fftshift(h, axes=2)

    return H_filt, h