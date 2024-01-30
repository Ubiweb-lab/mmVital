import numpy as np

def uDFTinv(X):
    N = len(X)

    x = np.sqrt(N) * np.fft.ifft(X)
    # print('shape of x in uDFTinv', x.shape)
    return x

