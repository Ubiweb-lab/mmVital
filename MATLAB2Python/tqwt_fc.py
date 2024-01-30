import numpy as np
from check_params import check_params


def tqwt_fc(Q, r, J, fs):
    # Check parameters
    check_params(Q, r, J)

    beta = 2 / (Q + 1)

    alpha = 1 - beta / r


    # Calculate center frequencies
    fc = np.concatenate(([0.5], alpha ** (2 + np.arange(J-1)) * (2 - beta) / (4 * alpha))) * fs

    return fc


# Example usage
# Q = 10
# r = 2
# J = 5
# fs = 44100
#
# center_frequencies = tqwt_fc(Q, r, J, fs)
# print("Center Frequencies:", center_frequencies)
