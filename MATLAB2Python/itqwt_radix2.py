import numpy as np
from check_params import check_params
from uDFT import uDFT
from lps import lps
from uDFTinv import uDFTinv
from sfb import sfb
from next import next_pow_of_2


def itqwt_radix2(w, Q, r, L):
    check_params(Q, r)

    beta = 2 / (Q + 1)
    # print('beta', beta)
    alpha = 1 - beta / r
    # print('alpha', alpha)
    J = len(w) - 1
    # print('J', J)

    N = next_pow_of_2(L)

    # print('N', N)

    Y = uDFT(w[J])
    # print('Y', Y)

    M = 2 * round(alpha**J * N / 2)  # *
    # print('M0', M)
    Y = lps(Y, M)  # *
    # print('Y2', Y)

    for j in range(J, 0, -1):
        # print('j', j)
        W = uDFT(w[j - 1])
        # print('W', W)
        N1 = 2 * round(beta * alpha**(j - 1) * N / 2)  # *
        # print('N1', N1)
        W = lps(W, N1)  # *
        M = 2 * round(alpha**(j - 1) * N / 2)
        # print('M', M)
        Y = sfb(Y, W, M)
        # print('Y', Y)

    y = uDFTinv(Y)
    # print('Y1', y)

    y = y[:L]  # *

    return y

# Define or implement uDFT, lps, sfb, and uDFTinv functions as needed
# Ensure that next_pow_of_2 is defined as well

# Example usage:
# Q, r, L = 4, 2, 16
#
# # Generate synthetic input for testing (replace with your specific input)
# w = [np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8]), np.array([9, 10, 11, 12]), np.array([13, 14, 15, 16])]
#
# # Call itqwt_radix2 function
# result = itqwt_radix2(w, Q, r, L)
#
# # Display the result
# print('Resulting signal:')
# print(result)
