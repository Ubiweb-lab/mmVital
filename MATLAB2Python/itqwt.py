import numpy as np
from check_params import check_params  # Assuming you have a function check_params defined
from uDFT import uDFT  # Assuming you have a function uDFT defined
from uDFTinv import uDFTinv  # Assuming you have a function uDFTinv defined
from sfb import sfb  # Assuming you have a function sfb defined

def itqwt(w, Q, r, N):
    check_params(Q, r)

    beta = 2 / (Q + 1)
    alpha = 1 - beta / r
    J = len(w) - 1

    Y = uDFT(w[J])

    for j in range(J-1, -1, -1):
        W = uDFT(w[j])
        M = 2 * round(alpha**(j) * N / 2)
        Y = sfb(Y, W, M)

    y = uDFTinv(Y)
    return y


