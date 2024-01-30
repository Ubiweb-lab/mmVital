import numpy as np


def lps(X, N0):
    # Low-pass scaling
    # Notes
    # - output Y will be length N0
    # - length(X) should be even

    N = len(X)
    Y = np.zeros(N0, dtype=X.dtype)

    if N0 <= N:
        k = np.arange(0, N0//2)
        # print('k1:', k)
        Y[k] = X[k]
        # print('X[k]:', X[k])
        Y[N0 // 2] = X[N // 2]
        k = np.arange(1, N0 // 2)
        # print('k2:', k)
        Y[N0 - k] = X[N - k]
    elif N0 >= N:
        k = np.arange(0, N//2)
        # print('k1:', k)
        Y[k] = X[k]
        # print('X[k]:', X[k])
        k = np.arange(N // 2, N0 // 2)
        # print('k2:', k)
        Y[k] = 0
        Y[N0 // 2] = X[N // 2]
        Y[N0 - k] = 0
        k = np.arange(1, N // 2)
        Y[N0 - k] = X[N - k]

    return Y


# Example usage
# N0 = 8
# X = np.array([29.+0.j, -1.+1.j, -1.+0.j, -1.-1.j])
# print('X', X)
# Y = lps(X, N0)
# print("Low-pass Scaling Result:", Y)
#
# # Test case 2: N0 >= N
# X_case2 = np.array([29.+0.j, -1.+1.j, -1.+0.j, -1.-1.j])
# N0_case2 = 12
# Y_case2 = lps(X_case2, N0_case2)
# print("Test Case 2:")
# print("Input X:")
# print(X_case2)
# print("Output Y:")
# print(Y_case2)