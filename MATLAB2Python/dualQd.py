import numpy as np
from tqwt_radix2 import tqwt_radix2
from itqwt_radix2 import itqwt_radix2
from next import next_pow_of_2
from ComputeNow import ComputeNow

def zeropad(x, N):
    return np.pad(x, (0, N - len(x)), mode='constant', constant_values=0)


def soft(x, T):
    y = np.zeros_like(x)
    k = np.where(x < -T)
    y[k] = x[k] + T
    k = np.where(x > T)
    y[k] = x[k] - T
    return y

def dualQd(x, Q1, r1, J1, Q2, r2, J2, lam1, lam2, mu, Nit):
    if Nit <= 0:
        raise ValueError("Number of iterations (Nit) should be a positive integer.")
    #
    # if plot_flag is not None and plot_flag == 'plots':
    #     GOPLOTS = True
    # else:
    #     GOPLOTS = False

    L = len(x)

    N = next_pow_of_2(L)

    if L < N:
        x = zeropad(x, N)

    # print('x0', x)
    # x = x.reshape((1, -1))  # row vector

    # Initialize
    # print('shape of x1', x.shape)
    w1 = tqwt_radix2(x, Q1, r1, J1)
    # num_columns_w1 = len(w1)
    # print(f'列表中的列数为dualQd_w1：{num_columns_w1}')
    w2 = tqwt_radix2(x, Q2, r2, J2)
    # num_columns_w2 = len(w2)
    # print(f'列表中的列数为dualQd_w2：{num_columns_w2}')
    d1 = tqwt_radix2(np.zeros_like(x), Q1, r1, J1)
    d2 = tqwt_radix2(np.zeros_like(x), Q2, r2, J2)

    T1 = lam1 / (2 * mu)
    T2 = lam2 / (2 * mu)

    u1 = [np.array([]) for _ in range(J1 + 1)]
    u2 = [np.array([]) for _ in range(J2 + 1)]

    # A = 1.1 * np.max(np.abs(x))

    # costfn = np.zeros(Nit) if Nit > 0 else []

    # print('J1:', J1)
    for k in range(Nit):
        for j in range(J1 + 1):
            # print('T1[j]', T1[j])
            # print('d1[j]', d1[j])
            # print('w1[j]', w1[j])

            u1[j] = soft(w1[j] + d1[j], T1[j]) - d1[j]

        for j in range(J2 + 1):
            u2[j] = soft(w2[j] + d2[j], T2[j]) - d2[j]

        c = x - itqwt_radix2(u1, Q1, r1, N) - itqwt_radix2(u2, Q2, r2, N)
        c = c / (mu + 2)

        d1 = tqwt_radix2(c, Q1, r1, J1)
        d2 = tqwt_radix2(c, Q2, r2, J2)

        for j in range(J1 + 1):
            w1[j] = d1[j] + u1[j]
        for j in range(J2 + 1):
            w2[j] = d2[j] + u2[j]

        # if Nit > 0:
        #     x1 = itqwt_radix2(w1, Q1, r1, N)
        #     x2 = itqwt_radix2(w2, Q2, r2, N)
        #
        #     res = x - x1 - x2
        #     costfn[k] = np.sum(np.abs(res)**2)
        #
        #     for j in range(J1 + 1):
        #         costfn[k] += lam1[j] * np.sum(np.abs(w1[j]))
        #
        #     for j in range(J2 + 1):
        #         costfn[k] += lam2[j] * np.sum(np.abs(w2[j]))

        # if GOPLOTS:
        #     # Add plotting code here if needed
        #     pass

    x1 = itqwt_radix2(w1, Q1, r1, L)
    x2 = itqwt_radix2(w2, Q2, r2, L)

    return x1, x2, w1, w2
    # return w1, w2


# Example

# Define an example input signal
# x = np.random.randn(1024)
#
# N = len(x)

# Define wavelet parameters
# Q1 = 10.88
# r1 = 3
# J1 = 27
# Q2 = 1.105263158
# r2 = 3
# J2 = 8
#
# theta1 = 0.5  # Normalization parameter
# theta2 = 0.5
# # Define regularization parameters
# now1 = ComputeNow(N, Q1, r1, J1, 'radix2')
# now2 = ComputeNow(N, Q2, r2, J2, 'radix2')
# lam1 = theta1 * now1
# print('shape of lam1', lam1.shape)
# print('lam1', lam1)
# lam2 = theta2 * now2
# print('shape of lam2', lam2.shape)
# print('lam2', lam2)
#
# # Other parameters
# Nit = 100  # Number of iterations
# mu = 0.5
#
# # Call the dualQd function
# x1, x2, w1, w2 = dualQd(x, Q1, r1, J1, Q2, r2, J2, lam1, lam2, mu, Nit)
#
# # Print or analyze the results as needed
# print("Signal 1:", x1)
# print("Signal 2:", x2)
# print("Wavelet coefficients 1:", w1)
# print("Wavelet coefficients 2:", w2)

