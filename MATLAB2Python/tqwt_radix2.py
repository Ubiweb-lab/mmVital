import numpy as np
from check_params import check_params
from lps import lps
from uDFTinv import uDFTinv
from afb import afb
from next import next_pow_of_2


def tqwt_radix2(x, Q, r, J):
    # Radix-2 tunable Q-factor wavelet transform (Radix-2 TQWT)
    # TQWT implemented using radix-2 FFTs
    check_params(Q, r, J)

    beta = 2 / (Q + 1)
    alpha = 1 - beta / r
    L = len(x)
    N = next_pow_of_2(L)


    # ---------------------------------------try discarding Jmax-------------------
    Jmax = int(np.floor(np.log(beta * N / 8) / np.log(1 / alpha)))
    # print('J:', J)
    # print('Jmax:', Jmax)
    if J > Jmax:
        if Jmax > 0:
            print(f"Reduce levels to {Jmax}")
        else:
            print("Increase signal length.")
        return None
    # ---------------------------------------end-------------------
    X = np.fft.fft(x, N) / np.sqrt(N)

    w = [None] * (J + 1)

    for j in range(1, J + 1):
        N0 = 2 * round(alpha**j * N / 2)
        N1 = 2 * round(beta * alpha**(j - 1) * N / 2)
        # print('——————————————————————————————————————————————————————————————————-')
        # print('shape of x2', X.shape)
        X, W = afb(X, N0, N1)
        W = lps(W, next_pow_of_2(N1))

        w[j - 1] = uDFTinv(W)
        # print('shape of x in w[j - 1]', w[j - 1].shape)
        # num_columns = len(w[j - 1])
        # print(f'列表中的列数为_tqwt：{num_columns}')

    X = lps(X, next_pow_of_2(N0));
    w[J] = uDFTinv(X)
    # print('shape of w', w.shape)
    return w



# Example usage
# Example usage of tqwt_radix2 function

# Define specific input for testing
# x_example = np.array([0.814723686393179, 0.905791937075619, 0.126986816293506, 0.913375856139019, 0.632359246225410, 0.0975404049994095, 0.278498218867048, 0.546881519204984, 0.957506835434298, 0.964888535199277, 0.157613081677548, 0.970592781760616, 0.957166948242946, 0.485375648722841, 0.800280468888800, 0.141886338627215, 0.421761282626275, 0.915735525189067, 0.792207329559554, 0.959492426392903])
# Q_example = 2
# r_example = 4
# J_example = 3
#
# # Call the tqwt_radix2 function with the example input
# w_example = tqwt_radix2(x_example, Q_example, r_example, J_example)
#
# # Display or inspect the resulting wavelet coefficients
# print('Wavelet coefficients (w):')
# print(w_example)

# You can also visualize the coefficients or perform further analysis as needed


#example_real
# x_example = np.array([0.814723686393179, 0.905791937075619, 0.126986816293506, 0.913375856139019, 0.632359246225410, 0.0975404049994095, 0.278498218867048, 0.546881519204984, 0.957506835434298, 0.964888535199277, 0.157613081677548, 0.970592781760616, 0.957166948242946, 0.485375648722841, 0.800280468888800, 0.141886338627215, 0.421761282626275, 0.915735525189067, 0.792207329559554, 0.959492426392903])
# x_example = np.random.randn(1024)
# Q_example = 10.88
# r_example = 3
# J_example = 27
#
# # Call the tqwt_radix2 function with the example input
# w_example = tqwt_radix2(x_example, Q_example, r_example, J_example)
#
# # Display or inspect the resulting wavelet coefficients
# print('Wavelet coefficients (w):')
# print(w_example)