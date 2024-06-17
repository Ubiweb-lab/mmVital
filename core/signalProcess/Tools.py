import copy
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

class Tools:

    @staticmethod
    def reshape(data, rows=8):
        """
        example:\n
        data = [3504, -623, 1636, 4461, -30, 33, -1156, 2962, 3610, -1096, 2462, 3824, 2913, 1277, 2229, 6153]\n
        rows = 8\n
        then, the output will be:\n
        [[3504, 3610],
        [-623, -1096],
        [1636, 2462],
        [4461, 3824],
        [-30, 2913],
        [33, 1277],
        [-1156, 2229],
        [2962, 6153]]
        :param data: one dimensional array
        :param rows:
        :return:
        """
        cols = int(len(data) / rows)
        b = []

        for i, e in enumerate(data):
            if i >= rows:
                break
            row_list = []
            for j in range(cols):
                row_list.append(data[i + j*rows])
            b.append(row_list)

        return b

    @staticmethod
    def complex_convert(data):
        real_part = np.array(data[0:4])
        imaginary_part = np.array(data[4:8])
        return real_part + 1j * imaginary_part

    @staticmethod
    def complex_two(real_part, imaginary_part):
        return real_part + 1j * imaginary_part

    @staticmethod
    def ULA(num_elements=4, element_spacing=0.5):
        ex_value = element_spacing/2
        array = np.zeros((3, 4), dtype=float)
        row_2 = np.arange(-element_spacing-ex_value, (element_spacing + ex_value + 1e-10), element_spacing)
        array[1] = row_2

        return array

    @staticmethod
    def lps(X, N0):
        # Low-pass scaling

        N = len(X)
        N0 = int(N0)
        Y = np.zeros(N0).astype(np.complex128)

        # Add 1 to indices because Python indexing starts at 0

        if N0 <= N:
            k = np.arange(N0 // 2)
            Y[k] = X[k]

            Y[N0 // 2] = X[N // 2]

            k = np.arange(1, N0 // 2)
            Y[N0 - k] = X[N - k]

        elif N0 >= N:
            k = np.arange(N // 2)
            Y[k] = X[k]

            k = np.arange(N // 2, N0 // 2)
            Y[k] = 0

            Y[N0 // 2] = X[N // 2]

            k = np.arange(1, N // 2)
            Y[N0 - k] = X[N - k]

        return Y

    @staticmethod
    def sfb(V0, V1, N):
        # sfb: synthesis filter bank

        N0 = len(V0)
        N1 = len(V1)

        S = (N - N0) // 2
        P = (N - N1) // 2
        T = (N0 + N1 - N) // 2 - 1

        # transition-band function
        v = np.arange(1, T + 1) / (T + 1) * np.pi
        trans = (1 + np.cos(v)) * np.sqrt(2 - np.cos(v)) / 2

        # low-pass subband
        Y0 = np.zeros(N).astype(np.complex128)
        Y0[0] = V0[0]  # dc term
        Y0[0 + 1:P + 1] = V0[0 + 1:P + 1]  # pass-band (pos freq)
        Y0[P + 0 + 1:P + T + 1] = V0[P + 0 + 1:P + T + 1] * trans  # trans-band (pos freq)
        Y0[P + T + 0 + 1:P + T + S + 1] = 0  # stop-band (pos freq)
        if N % 2 == 0:
            Y0[N // 2] = 0  # Nyquist freq (if N even)
        Y0[N - P - T - S:N - P - T] = 0  # stop-band (neg freq)
        Y0[N - P - T:N - P] = (V0[N0 - P - T:N0 - P][::-1] * trans)[::-1]  # trans-band (neg freq)
        Y0[N - P:] = V0[N0 - P:]

        Y0_SUM = np.sum(Y0)

        # high-pass subband
        Y1 = np.zeros(N).astype(np.complex128)
        Y1[0] = 0  # dc term
        Y1[1:P + 1] = 0  # stop-band (pos freq)
        Y1[P + 1:P + T + 1] = V1[1:T + 1] * trans[::-1]  # trans-band (pos freq)
        Y1[P + T + 1:P + T + S + 1] = V1[T + 1:T + S + 1]  # pass-band (pos freq)
        if N % 2 == 0:
            Y1[N // 2] = V1[N1 // 2]  # Nyquist freq (if N even)
        Y1[N - P - T - S:N - P - T] = V1[N1 - T - S:N1 - T]  # pass-band (neg freq)
        Y1[N - P - T:N - P] = V1[N1 - T:N1] * trans  # trans-band (neg freq)
        Y1[N - P:] = 0  # stop-band (neg freq)

        Y1_SUM = np.sum(Y1)

        Y = Y0 + Y1
        return Y

    @staticmethod
    def next(k):
        return 2 ** np.ceil(np.log2(k))

    @staticmethod
    def compute_now(N, Q, r, J, radix_flag=None):
        if radix_flag == 'radix2':
            R2 = True
        elif radix_flag is None:
            R2 = False
        else:
            print('Invalid string')
            return None

        if R2:
            if np.log2(N) != round(np.log2(N)):
                print('N must be a power of 2 for radix-2 option for computing the norm of wavelets')
                print('(otherwise, not all wavelets in each subband have equal norm).')
                return None

        beta = 2 / (Q + 1)
        alpha = 1 - beta / r

        w = []
        for i in range(J):
            j = i + 1
            N0 = 2 * round(alpha ** j * N / 2)
            N1 = 2 * round(beta * alpha ** (j - 1) * N / 2)
            if R2 is True:
                dim1 = int(Tools.next(N1))
                w.append(np.zeros(dim1))
                # print("i = " + str(i) + ", dim1 = " + str(dim1))
            else:
                w.append(np.zeros(N1))

        if R2 is True:
            dim1 = int(Tools.next(N0))
            w.append(np.zeros(dim1))
        else:
            w.append(np.zeros(N0))

        wz = copy.deepcopy(w)

        now = np.zeros(J + 1)

        for i in range(0, J + 1):
            w = copy.deepcopy(wz)

            M = len(w[i])
            w[i][:M] = 1 / np.sqrt(M)

            Y = w[J]
            if R2:
                M = 2 * round(alpha ** J * N / 2)
                Y = Tools.lps(Y, M)

            for j in range(J, 0, -1):
                W = w[j - 1]
                if R2:
                    N1 = 2 * round(beta * alpha ** (j - 1) * N / 2)
                    W = Tools.lps(W, N1)

                M = 2 * round(alpha ** (j - 1) * N / 2)
                Y = Tools.sfb(Y, W, M)
                Y_SUM = np.sum(Y)

            now[i] = np.sqrt(np.sum(np.abs(Y) ** 2))

        return now


    @staticmethod
    def myemd(x, fs):

        N = len(x)

        Q1 = 10.88
        r1 = 3
        J1 = 27

        Q2 = 1.105263158
        r2 = 3
        J2 = 8

        Nit = 100

        mu = 0.5
        theta1 = 0.5
        theta2 = 0.5

        fs = fs;

        now1 = Tools.compute_now(N, Q1, r1, J1, 'radix2')
        now2 = Tools.compute_now(N, Q2, r2, J2, 'radix2')

        lam1 = theta1 * now1
        lam2 = theta2 * now2

        _, _, w1, _ = Tools.dualQd(x, Q1, r1, J1, Q2, r2, J2, lam1, lam2, mu, Nit)

        e = Tools.PlotEnergy(w1);
        fc = Tools.tqwt_fc(Q1, r1, J1, fs);
        hr_test = np.zeros((2, J1))
        hr_test[0, :] = fc
        hr_test[1, :] = e[0:J1]

        hr_test_enrgy = hr_test[:, hr_test[0, :] >= 1]
        hr_test_enrgy_1 = hr_test_enrgy[:, hr_test_enrgy[0, :] < 1.25]
        hr_test_enrgy_2 = hr_test_enrgy[:, hr_test_enrgy[0, :] < 1.5]
        hr_test_enrgy_3 = hr_test_enrgy[:, hr_test_enrgy[0, :] < 1.75]
        hr_test_enrgy_4 = hr_test_enrgy[:, hr_test_enrgy[0, :] < 2]

        # Calculating energy values
        enrgy_1 = np.sum(hr_test_enrgy_1[1, :])
        enrgy_2 = np.sum(hr_test_enrgy_2[1, :]) - enrgy_1
        enrgy_3 = np.sum(hr_test_enrgy_3[1, :]) - enrgy_2 - enrgy_1
        enrgy_4 = np.sum(hr_test_enrgy_4[1, :]) - enrgy_3 - enrgy_2 - enrgy_1

        hr_energy = np.zeros((1, 4))

        hr_energy[0, 0] = enrgy_1 * 100 / (enrgy_1 + enrgy_2 + enrgy_3 + enrgy_4)
        hr_energy[0, 1] = enrgy_2 * 100 / (enrgy_1 + enrgy_2 + enrgy_3 + enrgy_4)
        hr_energy[0, 2] = enrgy_3 * 100 / (enrgy_1 + enrgy_2 + enrgy_3 + enrgy_4)
        hr_energy[0, 3] = enrgy_4 * 100 / (enrgy_1 + enrgy_2 + enrgy_3 + enrgy_4)

        hr_energy = np.round(hr_energy)

        root_directory = os.path.dirname(os.path.abspath(__file__))
        q = loadmat(os.path.join(root_directory, 'Qfile.mat'))
        q_test = q['q_heart_all']
        q_test1 = q_test[:, :4]

        # Calculate distances and find the index with the minimum distance
        d = np.linalg.norm(q_test1[:, :4] - hr_energy, axis=1)
        indB = np.argmin(d)

        # Extract Q1 and set J1 accordingly
        Q1 = q_test[indB, 4]
        J1 = 22 if Q1 < 5 else J1  # Set your desired value if Q1 is not less than 5

        y1, y2, w1, w2 = Tools.dualQd(x, Q1, r1, J1, Q2, r2, J2, lam1, lam2, mu, Nit);

        return y1, y2, w1, Q1

    @staticmethod
    def myemd_sec(x, Q1):
        N = len(x)

        # RSSD parameter initialization
        r1 = 3
        J1 = 27 if Q1 >= 5 else 22
        Q2 = 1.105263158
        r2 = 3
        J2 = 8

        # MCA parameters
        Nit = 100  # Number of iterations
        mu = 0.5  # SALSA parameter
        theta1 = 0.5  # normalization parameter
        theta2 = 0.5

        # Perform decomposition
        now1 = Tools.compute_now(N, Q1, r1, J1, 'radix2')
        now2 = Tools.compute_now(N, Q2, r2, J2, 'radix2')
        lam1 = theta1 * now1
        lam2 = theta2 * now2

        y1, y2, w1, w2 = Tools.dualQd(x, Q1, r1, J1, Q2, r2, J2, lam1, lam2, mu, Nit)

        return y1, y2, w1


    @staticmethod
    def tqwt_fc(Q, r, J, fs):
        # check_params(Q, r, J)

        beta = 2 / (Q + 1)
        alpha = 1 - beta / r
        fc = np.concatenate([[0.5], alpha ** (np.arange(2, J + 1)) * (2 - beta) / (4 * alpha)]) * fs

        return fc

    @staticmethod
    def PlotEnergy(w, Q=None, r=None, fs=None):
        J = len(w) - 1

        e = np.zeros(J + 1)
        for j in range(1, J + 2):
            e[j - 1] = np.sum(np.abs(w[j - 1]) ** 2)

        return e


    @staticmethod
    def zeropad(x, N):
        """
        Zero pads vector x to length N. (Appends zeros to the end of x.)

        Parameters:
        - x: numpy array, vector
        - N: int, scalar

        Returns:
        - y: numpy array, zero-padded vector
        """
        if not np.isscalar(N):
            raise ValueError('Error in zeropad: N must be a scalar.')

        L = len(x)

        if L > N:
            raise ValueError('Warning in zeropad: N must be >= length(x).')
        elif L == N:
            y = x.copy()
        else:
            y = np.append(x, np.zeros(N - L))  # perform zero-padding.

        return y


    @staticmethod
    def uDFTinv(X):
        """
        Inverse unitary DFT.

        Parameters:
        - X: numpy array, input signal in the frequency domain

        Returns:
        - x: numpy array, time-domain signal
        """
        N = len(X)
        x = np.sqrt(N) * np.fft.ifft(X)
        return x.astype(np.float64)

    @staticmethod
    def uDFT(x):
        """
        Unitary Discrete Fourier Transform.

        Parameters:
        - x: numpy array, input signal

        Returns:
        - X: numpy array, frequency domain representation
        """
        N = len(x)
        X = np.fft.fft(x) / np.sqrt(N)
        return X


    @staticmethod
    def afb(X, N0, N1):
        """
        Analysis Filter Bank.

        Converts a vector X into two vectors: V0 of length N0, V1 of length N1.

        Parameters:
        - X: numpy array, input vector
        - N0: int, length of V0
        - N1: int, length of V1

        Returns:
        - V0: numpy array, low-pass subband
        - V1: numpy array, high-pass subband
        """

        X = X.reshape(1, -1).squeeze()  # Ensure X is a row vector
        N = len(X)

        P = (N - N1) // 2
        T = (N0 + N1 - N) // 2 - 1
        S = (N - N0) // 2

        # Transition-band function
        v = (np.arange(1, T + 1) / (T + 1)) * np.pi
        trans = (1 + np.cos(v)) * np.sqrt(2 - np.cos(v)) / 2

        # Add 1 to indices because Python indexing starts at 0

        # Low-pass subband
        V0 = np.zeros(N0).astype(np.complex128)
        V0[0] = X[0]  # dc term
        V0[1:P + 1] = X[1:P + 1]  # pass-band (pos freq)
        V0[P + 1:P + T + 1] = X[P + 1:P + T + 1] * trans  # trans-band (pos freq)
        V0[N0 // 2] = 0  # Nyquist freq
        V0[N0 - P - T:N0 - P] = X[N - P - T:N - P] * trans[::-1]  # trans-band (neg freq)
        V0[N0 - P:N0] = X[N - P:N]  # pass-band (neg freq)

        # High-pass subband
        V1 = np.zeros(N1).astype(np.complex128)
        V1[0] = 0  # dc term
        V1[1:T + 1] = X[P + 1:P + T + 1] * trans[::-1]  # trans-band (pos freq)
        V1[T + 1:T + S + 1] = X[P + T + 1:P + T + S + 1]  # pass-band (pos freq)
        if N % 2 == 0:
            V1[N1 // 2] = X[N // 2]  # Nyquist freq (if N even)
        V1[N1 - T - S:N1 - T] = X[N - P - T - S:N - P - T]  # pass-band (neg freq)
        V1[N1 - T:N1] = ((X[N - P - T:N - P][::-1]) * trans[::-1])[::-1]  # trans-band (neg freq)

        return V0, V1


    @staticmethod
    def tqwt_radix2(x, Q, r, J):
        # check_params(Q, r, J)

        beta = 2 / (Q + 1)
        alpha = 1 - beta / r
        L = len(x.squeeze())
        N = Tools.next(L)

        Jmax = int(np.floor(np.log(beta * N / 8) / np.log(1 / alpha)))
        if J > Jmax:
            if Jmax > 0:
                print(f"Reduce levels to {Jmax}")
            else:
                print("Increase signal length.")
            return []

        X = np.fft.fft(x, N) / np.sqrt(N)

        w = [None] * (J + 1)

        for j in range(1, J + 1):
            N0 = 2 * np.round(alpha ** j * N / 2).astype(int)
            N1 = 2 * np.round(beta * alpha ** (j - 1) * N / 2).astype(int)
            X, W = Tools.afb(X, N0, N1)
            W = Tools.lps(W, Tools.next(N1))
            w[j - 1] = Tools.uDFTinv(W)

        X = Tools.lps(X, Tools.next(N0))
        w[J] = Tools.uDFTinv(X)

        return w


    @staticmethod
    def itqwt_radix2(w, Q, r, L):
        """
        Inverse radix-2 TQWT.

        Parameters:
        - w: list of numpy arrays, wavelet coefficients
        - Q: float, Q-factor
        - r: float, oversampling rate (redundancy)
        - L: int, length of the output signal

        Returns:
        - y: numpy array, inverse TQWT result
        """
        # check_params(Q, r)

        beta = 2 / (Q + 1)
        alpha = 1 - beta / r
        J = len(w) - 1

        N = Tools.next(L)  # *

        Y = Tools.uDFT(w[J])

        M = 2 * round(alpha ** J * N / 2)  # *
        Y = Tools.lps(Y, M)  # *

        for j in range(J, 0, -1):
            W = Tools.uDFT(w[j-1])
            N1 = 2 * round(beta * alpha ** (j - 1) * N / 2)  # *
            W = Tools.lps(W, N1)  # *
            M = 2 * round(alpha ** (j - 1) * N / 2)
            Y = Tools.sfb(Y, W, M)

        y = Tools.uDFTinv(Y)
        y = y[:L]  # *

        return y

    @staticmethod
    def dualQd(x, Q1, r1, J1, Q2, r2, J2, lam1, lam2, mu, Nit, plot_flag=None):

        def soft(x, T):
            if np.isreal(x).all():
                y = np.zeros_like(x)
                k = (x < -T)
                y[k] = x[k] + T
                k = (x > T)
                y[k] = x[k] - T
            else:
                # following alternative definition works for real and complex data:
                y = np.maximum(np.abs(x) - T, 0)
                y = y / (y + T) * x

            return y

        # By default do not compute cost function (to reduce computation)
        if False:
            COST = True
            costfn = np.zeros(Nit)  # cost function
        else:
            COST = False
            costfn = []

        GOPLOTS = False
        if plot_flag == 'plots':
            GOPLOTS = True

        L = len(x)
        N = Tools.next(L)
        if L < N:
            x = Tools.zeropad(x, N)

        x = x.reshape((1, -1))  # row vector

        # Initialize:
        w1 = Tools.tqwt_radix2(x, Q1, r1, J1)
        w2 = Tools.tqwt_radix2(x, Q2, r2, J2)
        d1 = Tools.tqwt_radix2(np.zeros_like(x), Q1, r1, J1)
        d2 = Tools.tqwt_radix2(np.zeros_like(x), Q2, r2, J2)

        T1 = lam1 / (2 * mu)
        T2 = lam2 / (2 * mu)

        u1 = [np.zeros_like(w) for w in w1]
        u2 = [np.zeros_like(w) for w in w2]

        N = len(x.squeeze())
        A = 1.1 * np.max(np.abs(x))

        for k in range(Nit):
            for j in range(J1 + 1):
                u1[j] = soft(w1[j] + d1[j], T1[j]) - d1[j]
            for j in range(J2 + 1):
                u2[j] = soft(w2[j] + d2[j], T2[j]) - d2[j]

            c = x - Tools.itqwt_radix2(u1, Q1, r1, N) - Tools.itqwt_radix2(u2, Q2, r2, N)
            c = c / (mu + 2)

            d1 = Tools.tqwt_radix2(c, Q1, r1, J1)
            d2 = Tools.tqwt_radix2(c, Q2, r2, J2)

            for j in range(J1 + 1):
                w1[j] = d1[j] + u1[j]

            for j in range(J2 + 1):
                w2[j] = d2[j] + u2[j]

            if COST or GOPLOTS:
                x1 = Tools.itqwt_radix2(w1, Q1, r1, N)
                x2 = Tools.itqwt_radix2(w2, Q2, r2, N)

                res = x - x1 - x2
                costfn[k] = np.sum(np.abs(res) ** 2)
                for j in range(J1 + 1):
                    costfn[k] += lam1[j] * np.sum(np.abs(w1[j]))
                for j in range(J2 + 1):
                    costfn[k] += lam2[j] * np.sum(np.abs(w2[j]))

            if GOPLOTS:
                plt.figure()
                plt.subplot(3, 1, 1)
                plt.plot(x1[0])
                plt.xlim([0, N])
                plt.ylim([-A, A])
                plt.title(f"ITERATION {k}\nCOMPONENT 1")
                plt.box(False)
                plt.subplot(3, 1, 2)
                plt.plot(x2[0])
                plt.xlim([0, N])
                plt.ylim([-A, A])
                plt.box(False)
                plt.title('COMPONENT 2')
                plt.subplot(3, 1, 3)
                plt.plot(res[0])
                plt.xlim([0, N])
                plt.ylim([-A, A])
                plt.title('RESIDUAL')
                plt.box(False)
                plt.show()

        x1 = Tools.itqwt_radix2(w1, Q1, r1, L)
        x2 = Tools.itqwt_radix2(w2, Q2, r2, L)

        if COST:
            return x1, x2, w1, w2, costfn
        else:
            return x1, x2, w1, w2

    @staticmethod
    def plot_energy(w, Q=None, r=None, fs=None):
        J = len(w) - 1

        e = np.zeros(J + 1)
        for j in range(1, J + 2):
            e[j - 1] = np.sum(np.abs(w[j - 1]) ** 2)

        return e

    @staticmethod
    def tqwt_fc(Q, r, J, fs):
        # Check parameters
        # check_params(Q, r, J)

        beta = 2 / (Q + 1)
        alpha = 1 - beta / r
        fc = np.concatenate(([0.5], alpha ** (np.arange(2, J + 1)) * (2 - beta) / (4 * alpha))) * fs

        return fc

    @staticmethod
    def histogram(array, bin_width=5):
        min = np.min(array)
        max = np.max(array)

        S_BinEdges = 0
        E_BinEdges = 0

        for i in np.arange(min, min-bin_width, -1):
            if i % bin_width == 0:
                S_BinEdges = i
                break

        for i in np.arange(max, max+bin_width, 1):
            if i % bin_width == 0:
                E_BinEdges = i
                break

        BinEdges = np.arange(S_BinEdges, E_BinEdges+1, bin_width)

        h, _ = np.histogram(array, bins=BinEdges)
        idxh = np.argsort(-h)

        return h, BinEdges, idxh


    @staticmethod
    def myemd_rr(x, J1, fs):
        N = len(x)

        # RSSD parameter initialization
        Q1 = 4
        r1 = 3

        Q2 = 1.105263158
        r2 = 3
        J2 = 8

        # Set MCA parameters
        Nit = 100  # Number of iterations
        mu = 0.5  # SALSA parameter
        theta1 = 0.5  # normalization parameter
        theta2 = 0.5

        # Peform decomposition
        now1 = Tools.compute_now(N, Q1, r1, J1, 'radix2')
        now2 = Tools.compute_now(N, Q2, r2, J2, 'radix2')
        lam1 = theta1 * now1
        lam2 = theta2 * now2

        _, _, w1, _ = Tools.dualQd(x, Q1, r1, J1, Q2, r2, J2, lam1, lam2, mu, Nit)

        # Energy distribution in levels
        e = Tools.plot_energy(w1)

        # Center frequency of levels
        fc = Tools.tqwt_fc(Q1, r1, J1, fs)

        rr_test = np.zeros((2, J1))
        rr_test[0, :] = fc
        rr_test[1, :] = e[0:J1]

        # Energy distribution in different frequency HR ranges
        rr_test_energy = rr_test[:, (rr_test[0, :] >= 0.1)]
        rr_test_energy_1 = rr_test_energy[:, (rr_test_energy[0, :] < 0.25)]
        enrgy_1 = np.sum(rr_test_energy_1[1, :])
        rr_test_energy_2 = rr_test_energy[:, (rr_test_energy[0, :] < 0.33)]
        enrgy_2 = np.sum(rr_test_energy_2[1, :]) - enrgy_1
        rr_test_energy_3 = rr_test_energy[:, (rr_test_energy[0, :] < 0.42)]
        enrgy_3 = np.sum(rr_test_energy_3[1, :]) - enrgy_2 - enrgy_1
        rr_test_energy_4 = rr_test_energy[:, (rr_test_energy[0, :] < 0.6)]
        enrgy_4 = np.sum(rr_test_energy_4[1, :]) - enrgy_3 - enrgy_2 - enrgy_1

        # Energy distribution in different frequency HR ranges
        # according to which final RSSD decomposition shall be done
        rr_energy = np.zeros((1, 4))
        rr_energy[0, 0] = enrgy_1 * 100 / (enrgy_1 + enrgy_2 + enrgy_3 + enrgy_4)
        rr_energy[0, 1] = enrgy_2 * 100 / (enrgy_1 + enrgy_2 + enrgy_3 + enrgy_4)
        rr_energy[0, 2] = enrgy_3 * 100 / (enrgy_1 + enrgy_2 + enrgy_3 + enrgy_4)
        rr_energy[0, 3] = enrgy_4 * 100 / (enrgy_1 + enrgy_2 + enrgy_3 + enrgy_4)
        rr_energy = np.round(rr_energy)

        # Load Qfile_RR.mat
        root_directory = os.path.dirname(os.path.abspath(__file__))
        q_data = loadmat(os.path.join(root_directory, 'Qfile_RR.mat'))
        q_test = q_data['q_RR_all']
        q_test1 = q_test[:, 0:4]

        d = np.linalg.norm(q_test1 - rr_energy, axis=1)
        indB = np.argmin(d)
        Q1 = q_test[indB, 4]

        y1, y2, w1, _ = Tools.dualQd(x, Q1, r1, J1, Q2, r2, J2, lam1, lam2, mu, Nit)

        return y1, y2, w1, Q1


    @staticmethod
    def myemd_sec_rr(x, Q1, J1):
        N = len(x)

        # RSSD parameter initialization
        r1 = 3
        if Q1 < 5:
            J1 = 22
        elif Q1 >= 5 and Q1 <= 6:
            J1 = 25

        Q2 = 1.105263158
        r2 = 3
        J2 = 8

        # Set MCA parameters
        Nit = 100  # Number of iterations
        mu = 0.5  # SALSA parameter
        theta1 = 0.5  # normalization parameter
        theta2 = 0.5

        # Perform decomposition
        now1 = Tools.compute_now(N, Q1, r1, J1, 'radix2')
        now2 = Tools.compute_now(N, Q2, r2, J2, 'radix2')
        lam1 = theta1 * now1
        lam2 = theta2 * now2

        y1, y2, w1, _ = Tools.dualQd(x, Q1, r1, J1, Q2, r2, J2, lam1, lam2, mu, Nit)

        return y1, y2, w1, J1

# Tools.myemd(4, 0.5)