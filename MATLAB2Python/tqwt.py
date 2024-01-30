import numpy as np
from check_params import check_params
from uDFT import uDFT
from uDFTinv import uDFTinv
from afb import afb


def tqwt(x, Q, r, J):
    check_params(Q, r, J)

    beta = 2 / (Q + 1)
    alpha = 1 - beta / r
    N = len(x)

    Jmax = int(np.floor(np.log(beta * N / 8) / np.log(1 / alpha)))
    print('Jmax', Jmax)
    if J > Jmax:
        J = Jmax
        print('Note: too many levels specified.')
        if Jmax > 0:
            print('Reduce levels to', Jmax)
        else:
            print('Increase signal length.')
        return []

    X = uDFT(x)
    w = [None] * (J + 1)

    # J stages:
    for j in range(1, J + 1):
        N0 = 2 * round(alpha ** j * N / 2)
        N1 = 2 * round(beta * alpha ** (j - 1) * N / 2)
        X, W = afb(X, N0, N1)
        w[j - 1] = uDFTinv(W)

    w[J] = uDFTinv(X)

    return w


# Define a fixed input signal
x = np.array([0.814723686393179, 0.905791937075619, 0.126986816293506, 0.913375856139019, 0.632359246225410, 0.0975404049994095, 0.278498218867048, 0.546881519204984, 0.957506835434298, 0.964888535199277, 0.157613081677548, 0.970592781760616, 0.957166948242946, 0.485375648722841, 0.800280468888800, 0.141886338627215, 0.421761282626275, 0.915735525189067, 0.792207329559554, 0.959492426392903])

# Set parameters
Q = 4
r = 2
J = 3

# Call the tqwt function with the fixed input signal
result = tqwt(x, Q, r, J)

# Print or inspect the output
print('Output Result:')
for j, w_j in enumerate(result):
    print(f'Subband {j+1}: {w_j}')

