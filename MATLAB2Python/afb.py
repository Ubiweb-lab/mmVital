import numpy as np

def afb(X, N0, N1):
    X = X.flatten()  # Ensure X is a 1D array
    N = len(X)
    # print('shape of N', N)
    # print('shape of N1', N1)
    # print('shape of X', X)
    P, T, S = (N - N1) // 2, (N0 + N1 - N) // 2 - 1, (N - N0) // 2
    # print('shape of T',T)

    # Transition-band function
    v = np.arange(1, T+1) / (T + 1) * np.pi
    # print('np.arange(0, T)', np.arange(1, T+1))
    trans = (1 + np.cos(v)) * np.sqrt(2 - np.cos(v)) / 2
    # print('shape of trans', trans)
    # print('shape of X[P + 1:P + T + 2]', X[P + 1:P + T + 1])
    # print('shape of X[N - P:N - P - T:-1]', X[N - P:N - P - T:-1])
    #print('shape of X[N - P:N - P - T]', X[N - P:N - P - T])
    #print('shape of X[P + 1:P + T]', X[P + 1:P + T])
    # print('shape of trans[T::-1]', trans[T::-1])
    # print('shape of V0[P + 1:P + T + 1]', X[P + 1:P + T + 1])
    #print('shape of V1[1:T + 1]', V1[1:T + 1])



    # Low-pass subband
    N0 = int(N0)
    V0 = np.zeros(N0, dtype=X.dtype)

    # 直流分量
    V0[0] = X[0]

    V0[1:P+1] = X[1:P+1]
    # print('shape of X[1:P+1]', X[1:P+1].shape)


    # 过渡带分量的正频率部分
    V0[P + 1:P + T + 1] = X[P + 1:P + T + 1] * trans
    # print("X[P + 1:P + T + 1]:", X[P + 1:P + T + 1])
    # print("V0_1:", V0[P + 1:P + T + 1])

    # 奈奎斯特频率分量
    V0[N0 // 2] = 0

    # 过渡带分量的负频率部分
    V0[N0 - P - 1:N0 - P - T - 1:-1] = X[N - P - 1:N - P - T - 1:-1] * trans

    # print("X[N - P - 1:N - P - T - 1:-1]:", X[N - P - 1:N - P - T - 1:-1])

    # 通带分量的负频率部分
    V0[N0:N0 - P - 1:-1] = X[N:N - P - 1:-1]
    # print("X[N:N - P:-1]:", X[N:N - P - 1:-1])

    # High-pass subband
    V1 = np.zeros(N1, dtype=X.dtype)
    V1[0] = 0
    V1[1:T+1] = X[P + 1:P + T + 1] * trans[T::-1]  # trans-band (pos freq)
    # print("X[P + 1:P + T + 1]:", X[P + 1:P + T + 1])
    V1[T + 1:T + S + 1] = X[P + T + 1:P + T + S + 1]  # pass-band (pos freq)
    # print("X[P + T + 1:P + T + S + 1]:", X[P + T + 1:P + T + S + 1])
    if N % 2 == 0:
        V1[N1 // 2] = X[N // 2]  # Nyquist freq (if N even)
    V1[N1 - T - 1:N1 - T - S - 1:-1] = X[N - P - T - 1:N - P - T - S - 1:-1]
    # print("X[N - P - T:N - P - T - S:-1]:", X[N - P - T - 1:N - P - T - S - 1:-1])
    V1[N1:N1 - T - 1:-1] = X[N - P - 1:N - P - T - 1:-1] * trans[T::-1]  # trans-band (neg freq)
    # print("X[N - P:N - P - T:-1]:", X[N - P - 1:N - P - T - 1:-1])
    return V0, V1



# Example usage
# X_example = np.array([0.3188, 0.4242, 0.5079, 0.0855, 0.2625, 0.8010, 0.2434, 0.8308, 0.1878, 0.7081,
#                      0.7503, 0.4505, 0.0838, 0.2290, 0.9133, 0.1524, 0.8258, 0.5383, 0.9961, 0.0782])
# N0_example = 16
# N1_example = 12
# V0_example, V1_example = afb(X_example, N0_example, N1_example)
# print("shape of V0:", V0_example.shape)
# print("V0:", V0_example)
# print("shape of V1:", V1_example.shape)
# print("V1:", V1_example)
