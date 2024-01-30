import numpy as np
from ComputeNow import ComputeNow
from PlotEnergy import PlotEnergy
from tqwt_fc import tqwt_fc
from dualQd import dualQd
from scipy.io import loadmat

def myemd(x, fs):
    N = len(x)

    # RSSD parameter initialization
    Q1 = 10.88
    r1 = 3
    J1 = 27
    Q2 = 1.105263158
    r2 = 3
    J2 = 8

    # Set MCA parameters
    Nit = 100          # Number of iterations
    mu = 0.5           # SALSA parameter
    theta1 = 0.5       # Normalization parameter
    theta2 = 0.5
    fs = fs

    # Perform decomposition
    print('N_start', N)
    now1 = ComputeNow(N, Q1, r1, J1, 'radix2')
    now2 = ComputeNow(N, Q2, r2, J2, 'radix2')
    lam1 = theta1 * now1
    # print('shape of lam1', lam1.shape)
    # print('lam1', lam1)
    lam2 = theta2 * now2
    # print('shape of lam2', lam2.shape)
    # print('lam2', lam2)
    # num_columns = len(x)
    # print(f'列表中的列数为befor_dualQd：{num_columns}')
    _, _, w1, _ = dualQd(x, Q1, r1, J1, Q2, r2, J2, lam1, lam2, mu, Nit)
    # num_columns_w1 = len(w1)
    # print(f'列表中的列数为after_dualQd：{num_columns_w1}')
    e = PlotEnergy(w1,Q1,r1,fs)  # energy distribution in levels
    fc = tqwt_fc(Q1, r1, J1, fs)  # center frequency of levels

    hr_test = np.zeros((2, J1))
    hr_test[0, :] = fc
    # print('shape of fc', fc.shape)
    # print('shape of hr_test[0, :]', hr_test[0, :].shape)
    # print('shape of e[0, 0:J1]', e.shape)
    # print('shape of hr_test[1, :]', hr_test[1, :].shape)
    # hr_test[1, :] = e[0, 0:J1]
    hr_test[1, :J1] = e[:J1]


    hr_test_energy = hr_test[:, (hr_test[0, :] >= 1)]
    hr_test_energy_1 = hr_test_energy[:, (hr_test_energy[0, :] < 1.25)]
    enrgy_1 = np.sum(hr_test_energy_1[1, :])

    hr_test_energy_2 = hr_test_energy[:, (hr_test_energy[0, :] < 1.5)]
    enrgy_2 = np.sum(hr_test_energy_2[1, :]) - enrgy_1

    hr_test_energy_3 = hr_test_energy[:, (hr_test_energy[0, :] < 1.75)]
    enrgy_3 = np.sum(hr_test_energy_3[1, :]) - enrgy_2 - enrgy_1

    hr_test_energy_4 = hr_test_energy[:, (hr_test_energy[0, :] < 2)]
    enrgy_4 = np.sum(hr_test_energy_4[1, :]) - enrgy_3 - enrgy_2 - enrgy_1

    # Energy distribution in different frequency HR ranges
    # according to which final RSSD decomposition shall be done
    hr_energy = np.zeros((1, 4))
    hr_energy[0, 0] = enrgy_1 * 100 / (enrgy_1 + enrgy_2 + enrgy_3 + enrgy_4)
    hr_energy[0, 1] = enrgy_2 * 100 / (enrgy_1 + enrgy_2 + enrgy_3 + enrgy_4)
    hr_energy[0, 2] = enrgy_3 * 100 / (enrgy_1 + enrgy_2 + enrgy_3 + enrgy_4)
    hr_energy[0, 3] = enrgy_4 * 100 / (enrgy_1 + enrgy_2 + enrgy_3 + enrgy_4)
    hr_energy = np.round(hr_energy)

    mat_data = loadmat('Qfile.mat')

    # call the value
    q_test = mat_data['q_heart_all']
    # print('q_test ', q_test)
    q_test1 = q_test[:, 0:4]
    d = np.zeros(664)

    for m in range(664):
        d[m] = np.linalg.norm(q_test1[m, 0:4] - hr_energy)

    indB = np.argmin(d)
    Q1 = q_test[indB, 4]

    if Q1 < 5:
        J1 = 22

    y1, y2, w1, _ = dualQd(x, Q1, r1, J1, Q2, r2, J2, lam1, lam2, mu, Nit)

    return y1, y2, w1, Q1


# Implement the missing functions ComputeNow, dualQd, PlotEnergy, tqwt_fc based on your original MATLAB code.
# Also, make sure you have Qfile.npy available for loading in the code.

