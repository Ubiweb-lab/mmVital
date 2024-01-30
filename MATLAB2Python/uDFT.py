import numpy as np

def uDFT(x):
    N = len(x)
    X = np.fft.fft(x) / np.sqrt(N)
    return X
