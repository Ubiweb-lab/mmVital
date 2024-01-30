import numpy as np

def sfb(V0, V1, N):
    N0 = len(V0)
    N1 = len(V1)


    # print('shape of N', N)
    # print('shape of N1', N1)
    # print('shape of X', X)
    P, T, S = (N - N1) // 2, (N0 + N1 - N) // 2 - 1, (N - N0) // 2
    # print('shape of T',T)

    # Transition-band function
    v = np.arange(1, T+1) / (T + 1) * np.pi
    # print('np.arange(0, T)', np.arange(1, T+1))
    trans = (1 + np.cos(v)) * np.sqrt(2 - np.cos(v)) / 2

    # print('shape of V0[N0 - P - 1:N - P - T - 1:-1]', V0[N0 - P - 1:N - P - T - 1:-1].shape)
    # print('shape of V0[N0 - P - 1:N - P - T - 1:-1]', V0[N0 - P - 1:N - P - T - 1:-1])
    # print('shape of trans', trans)
    # print('shape of X[P + 1:P + T + 2]', X[P + 1:P + T + 1])
    # print('shape of X[N - P:N - P - T:-1]', X[N - P:N - P - T:-1])
    #print('shape of X[N - P:N - P - T]', X[N - P:N - P - T])
    #print('shape of X[P + 1:P + T]', X[P + 1:P + T])
    # print('shape of trans[T::-1]', trans[T::-1])
    # print('shape of V0[P + 1:P + T + 1]', X[P + 1:P + T + 1])
    #print('shape of V1[1:T + 1]', V1[1:T + 1])



    # Low-pass subband

    # low-pass subband
    Y0 = np.zeros(N, dtype=V0.dtype)
    Y0[0] = V0[0]  # dc term
    Y0[1:P + 1] = V0[1:P + 1]  # pass-band (pos freq)
    Y0[P + 1:P + T + 1] = V0[P + 1:P + T + 1] * trans
    Y0[P + T + 1:P + T + S + 1] = 0
    if N % 2 == 0:
        Y0[N // 2] = 0  # Nyquist freq (if N even)
    Y0[N - P - T - 1:N - P - T - S - 1:-1] = 0

    Y0[N - P - 1:N - P - T - 1:-1] = V0[N0 - P - 1:N0 - P - T - 1:-1] * trans

    Y0[N:N - P - 1:-1] = V0[N0:N0 - P - 1:-1]

    # high-pass subband
    Y1 = np.zeros(N, dtype=V1.dtype)
    Y1[0] = 0  # dc term
    Y1[1:P + 1] = 0  # stop-band (pos freq)
    Y1[P + 1:P + T + 1] = V1[1:T + 1] * trans[T::-1]
    Y1[P + T + 1:P + T + S + 1] = V1[T + 1:T + S + 1]
    if N % 2 == 0:
        Y1[N // 2] = V1[N1 // 2]  # Nyquist freq (if N even)
    Y1[N - P - T - 1:N - P - T - S - 1:-1] = V1[N1 - T - 1:N1 - T - S - 1:-1]
    Y1[N - P - 1:N - P - T - 1:-1] = V1[N1:N1 - T - 1:-1] * trans[T::-1]   # trans-band (neg freq)
    Y1[N:N - P - 1:-1] = 0

    Y = Y0 + Y1
    return Y



# Sample input values
# V0_example = np.array([0.814723686393179, 0.905791937075619, 0.126986816293506, 0.913375856139019, 0.632359246225410, 0.0946665554577639, 0.196927979109265, 0.131772644590213, 0, 0.116952631613659, 0.565883746402421, 0.137705917300976, 0.421761282626275, 0.915735525189067, 0.792207329559554, 0.959492426392903])
# V1_example = np.array([0, 0.0235026357077444, 0.196927979109265, 0.530768655993977, 0.957506835434298, 0.964888535199277, 0.157613081677548, 0.970592781760616, 0.957166948242946, 0.471074943434438, 0.565883746402421, 0.0341879134978094])
# N_example = 20
#
# # Call the Python sfb function with the example input
# Y_example = sfb(V0_example, V1_example, N_example)
#
# # Display the resulting subband Y
# print('Synthesized Subband Y:')
# print(Y_example)

