import numpy as np
from ComputeNow import ComputeNow
from dualQd import dualQd


def myemd_sec_rr(x, Q1, J1):
    # RSSD parameter initialization

    N = len(x)
    # print('N', N)
    r1 = 3
    if Q1 < 5:
        J1 = 22
    elif 5 <= Q1 <= 6:
        J1 = 25

    Q2 = 1.105263158
    r2 = 3
    J2 = 8

    # Set MCA parameters
    Nit = 100  # Number of iterations
    mu = 0.5  # SALSA parameter
    theta1 = 0.5  # Normalization parameter
    theta2 = 0.5

    # Perform decomposition

    now2 = ComputeNow(N, Q2, r2, J2, 'radix2')
    # print('now2', now2)
    now1 = ComputeNow(N, Q1, r1, J1, 'radix2')
    # print('now1', now1)

    lam1 = theta1 * now1

    lam2 = theta2 * now2
    # print('shape of lam2', lam2.shape)
    # print('lam2', lam2)
    # print('shape of x', x.shape)
    y1, y2, w1, _ = dualQd(x, Q1, r1, J1, Q2, r2, J2, lam1, lam2, mu, Nit)

    return y1, y2, w1, J1


