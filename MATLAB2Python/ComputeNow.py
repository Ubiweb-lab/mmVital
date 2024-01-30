import numpy as np
from next import next_pow_of_2
from lps import lps
from sfb import sfb

def ComputeNow(N, Q, r, J, radix_flag=None):
    if radix_flag is not None and radix_flag != 'radix2':
        print('Invalid string')
        return None

    R2 = radix_flag == 'radix2'


    if R2 and np.log2(N) != np.round(np.log2(N)):
        print('N must be a power of 2 for radix-2 option for computing norm of wavelets')
        print('(otherwise, not all wavelets in each subband have equal norm).')
        return None

    beta = 2 / (Q + 1)
    alpha = 1 - beta / r

    # Create all-zero wavelet coefficients
    w = [None] * (J + 1)
    for j in range(1, J+1):
        N0 = 2 * np.round(alpha ** j * N / 2)
        # print('N0', N0)
        N1 = 2 * np.round(beta * alpha ** (j - 1) * N / 2)
        # print('N1', N1)
        if R2:
            w[j - 1] = np.zeros(next_pow_of_2(N1))


        else:
            w[j - 1] = np.zeros(int(N1))

    if R2:
        w[J] = np.zeros(next_pow_of_2(N0))
    else:
        w[J] = np.zeros(int(N0))

    wz = w.copy()

    now = np.zeros(J + 1)

    for i in range(1, J + 2):
        w = wz.copy()
        M = len(w[i-1])
        # print (M)
        w[i - 1][:M] = 1 / np.sqrt(M)
        Y = w[J]

        if R2:
            M = int(2 * np.round(alpha ** J * N / 2))
            Y = lps(Y, M)

        for j in range(J, 0, -1):
            W = w[j - 1]
            if R2:
                N1 = int(2 * np.round(beta * alpha ** (j - 1) * N / 2))
                W = lps(W, N1)
                # print('N1', N1)
            M = int(2 * np.round(alpha ** (j - 1) * N / 2))
            Y = sfb(Y, W, M)
            # print('Y', Y)
        now[i - 1] = np.sqrt(np.sum(np.abs(Y) ** 2))
        # print('now', now)

    return now


# # Example usage
# N = 1024
# Q = 4
# r = 3
# J = 22
# center_frequencies = ComputeNow(N, Q, r, J, radix_flag='radix2')
# print('shape of center_frequencies', center_frequencies.shape)
# print(center_frequencies)
