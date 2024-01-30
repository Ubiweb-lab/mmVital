import numpy as np
from tqwt_radix2 import tqwt_radix2
from itqwt_radix2 import itqwt_radix2
from tqwt import tqwt
from itqwt import itqwt
from next import next_pow_of_2
def compute_wavelets(N, Q, r, J, radix_flag=None):
    # Compute the wavelets for a J-level transform
    if radix_flag == 'radix2':
        xform = tqwt_radix2
        inv_xform = itqwt_radix2
        C = N / next_pow_of_2(N)


    else:
        xform = tqwt
        inv_xform = itqwt
        C = 1

    z = np.zeros(N)  # All-zero signal
    wz = xform(z, Q, r, J)  # All-zero wavelet coefficients

    if wz is None:
        wlets = []
        return wlets, np.zeros(J + 1)

    wlets = [None] * (J + 1)

    for j in range(1, J + 2):
        w = wz.copy()  # Set w to all-zero coefficients
        m = round(C * len(w[j - 1]) / 2) + 1  # m: index of coefficient corresponding to the midpoint of the signal
        w[j - 1][m - 1] = 1  # Set a single wavelet coefficient to 1
        y = inv_xform(w, Q, r, N)  # Inverse TQWT
        wlets[j - 1] = y

    now = np.zeros(J + 1)
    for j in range(J + 1):
        now[j] = np.sqrt(np.sum(np.abs(wlets[j]) ** 2))  # L2 norm of wavelet

    return wlets, now


def next_power_of_two(n):
    return 2 ** int(np.ceil(np.log2(n)))


# Placeholder functions tqwt, tqwt_radix2, itqwt, and itqwt_radix2
# You should provide the actual implementations of these functions based on your requirements.

# Example usage
# Q = 1
# r = 3
# J = 10
# N = 2 ** 9
#
# wlets, now = compute_wavelets(N, Q, r, J, radix_flag='radix2')
# print("Wavelets:", wlets)
# print("Wavelet Norms:", now)
