import os
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import pandas as pd

from core.signalProcess.Tools import *
from datetime import datetime

def signal_process():

    time_begin = datetime.now()

    root_path = os.path.dirname(os.path.abspath(__file__))
    path_bin = root_path + "Data1/adc_data.bin"
    path_bin = os.path.join(root_path, "Data1/adc_data.bin")
    path_HR_ref = root_path + "/Hex_HR_ref.xlsx"
    path_RR_ref = root_path + "/Hex_RR_ref.xlsx"

    x_HR = pd.read_excel(path_HR_ref, header=[1]).loc[:, "Data2"].values
    x_HR = x_HR.reshape(len(x_HR), 1)

    x_RR = pd.read_excel(path_RR_ref, header=[1]).loc[:, "Data2"].values
    x_RR = x_RR.reshape(len(x_RR), 1)

    # (1)
    # Read file and convert to signed number
    numADCBits = 16
    numLanes = 4
    isReal = 0

    with open(path_bin, 'rb') as file:
        adc_data = np.fromfile(file, dtype=np.int16)

    time_load_bin = datetime.now()
    print(f'file loading time = {time_load_bin - time_begin}')

    if numADCBits != 16:
        l_max = 2 ** (numADCBits - 1) - 1
        adc_data[adc_data > l_max] = adc_data[adc_data > l_max] - 2 ** numADCBits
    # adc_data = np.array([3504, -623, 1636, 4461, -30, 33, -1156, 2962, 3610, -1096, 2462, 3824, 2913, 1277, 2229, 6153])

    # Organize data by LVDS lane
    if isReal:
        adc_data_reshaped = np.reshape(adc_data, (numLanes, -1))
    else:
        adc_data_reshaped = Tools.reshape(adc_data, numLanes * 2)
        adc_data = Tools.complex_convert(adc_data_reshaped)
    # print(adc_data)
    time_complex_convert = datetime.now()
    print(f'complex conversion time = {time_complex_convert - time_load_bin}')

    # adc_data = adc_data[:, 0:200]
    retVal = adc_data

    # (2) ###########  Beamforming at receiver (PASS this part now) ################
    c = 3e8
    fc = 77e9
    lambda_ = c / fc
    Ne = 4

    # rxArray = Tools.ULA(4, lambda_ / 2)

    w = np.array([0.25, 0.25, 0.25, 0.25])
    retVal_transposed = retVal.T
    conj_w = np.conj(w)
    Xbf = np.dot(retVal_transposed, conj_w)
    bf_data = Xbf.T

    ##################### (3) DC Offset correction ##########################

    # bf_data = 1.0e+03 * np.array([3.5040 - 0.0300j, 3.6100 + 2.9130j, 1.9930 + 3.4640j, -1.6380 + 3.6710j])

    Bi = np.real(bf_data)
    Bq = np.imag(bf_data)
    Ar = np.sqrt(np.sum(np.power(Bi, 2) + np.power(Bq, 2)))


    def objective_function(dc):
        return np.abs(np.sqrt((Bi - dc[0]) ** 2 + (Bq - dc[1]) ** 2) - dc[2])


    dc0 = np.array([0, 0, Ar])
    options = {'method': 'lm'}
    result = least_squares(objective_function, dc0, method='lm')
    dcnew = result.x
    Bin = Bi - dcnew[0]
    Bqn = Bq - dcnew[1]

    # plt.figure(1)
    # plt.clf()
    # plt.plot(Bq, Bi, 'o', color='y', label='Original Data')

    # plt.plot(Bqn, Bin, '*', label='Offset Corrected')

    # plt.title('DC Offset Correction')
    # plt.legend()
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.axis('equal')

    # plt.show()

    bf_signal_dccorrected = Tools.complex_two(Bin, Bqn)
    # print(bf_signal_dccorrected.shape)

    time_dc_offset = datetime.now()
    print(f'(3) DC offset time = {time_dc_offset - time_complex_convert}')

    ################### (4) Data Cube Formation ######################
    Nsample = 128
    Nrow = round(len(bf_data) / Nsample)
    fs = 200
    end_val = 0
    data_cube = np.zeros((Nrow, Nsample), dtype=complex)
    data_cube_complete = np.zeros((Nrow, Nsample), dtype=complex)
    for i in range(Nrow):
        init_val = end_val + 1
        end_val = Nsample * (i + 1)
        data_cube[i, :] = bf_data[init_val - 1:end_val]
        data_cube_complete[i, :] = bf_signal_dccorrected[init_val - 1:end_val]

    data_cube_locs = np.diff(data_cube_complete, axis=0)

    time_data_cube = datetime.now()
    print(f'(4) Data Cube time = {time_data_cube - time_dc_offset}')

    ################ (5) PHASE EXTRACTION: VITAL SIGNAL ####################
    range_slow_mat_locs = np.fft.fft(data_cube_locs, axis=1, n=Nsample)
    magnitude_spectrum = np.abs(range_slow_mat_locs)
    magnitude_spectrum_in_db = 20 * np.log10(magnitude_spectrum)

    plt.imshow(magnitude_spectrum_in_db, aspect='auto', cmap='viridis')
    plt.colorbar(label='Magnitude (dB)')
    plt.title('Magnitude Spectrum in dB')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    # plt.show()

    pow_range = np.abs(range_slow_mat_locs) ** 2
    mean_pow_range = np.mean(pow_range, axis=0)
    subset_range = mean_pow_range[0:50]
    pks1 = np.max(subset_range)
    locs1 = np.argmax(subset_range)
    range_slow_mat = np.fft.fft(data_cube_complete, axis=1, n=Nsample)
    signal_target = range_slow_mat[:, locs1]
    sig_i = np.real(signal_target);
    sig_q = np.imag(signal_target);
    nmer = np.zeros((Nrow, 1));
    dnmer = np.zeros((Nrow, 1));
    ph_term = np.zeros((Nrow, 1));
    x_phase = np.zeros((Nrow, 1));

    for k in range(1, Nrow):
        nmer[k] = sig_i[k] * (sig_q[k] - sig_q[k - 1]) - (sig_i[k] - sig_i[k - 1]) * sig_q[k]
        dnmer[k] = sig_q[k] ** 2 + sig_i[k] ** 2
        ph_term[k] = nmer[k] / dnmer[k]

    for n in range(1, Nrow):
        x_phase[n] = np.sum(ph_term[1:n + 1])

    frame_duration = 5
    fs = 1000 / frame_duration
    down_sampling_factor = 20
    fs = fs / down_sampling_factor

    x_phase = x_phase[::down_sampling_factor]

    N = len(x_phase)
    t = np.arange(0, N) / fs
    # plt.figure(5)
    # plt.clf()

    # plt.plot(t, x_phase, 'b')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')
    # plt.title('Downsampled Signal')
    # plt.show()

    time_phase_extraction = datetime.now()
    print(f'(5) PHASE EXTRACTION time = {time_phase_extraction - time_data_cube}')

    ############## (6) HR Harmonic index save #################
    r1 = 3;
    J1 = 27;
    x_sig = x_phase[:, 0]
    x_sig.reshape(len(x_sig), 1)
    _, _, w1, Q1 = Tools.myemd(x_sig, fs)
    e = Tools.plot_energy(w1);

    if Q1 < 5:
        J1 = 22;

    fc = Tools.tqwt_fc(Q1, r1, J1, fs);

    hr_test = np.zeros((2, J1))
    hr_test[0, :] = fc
    hr_test[1, :] = e[0:J1]

    # Find out the HR indices for the fundamental
    hr_test_1 = hr_test[:, (hr_test[0, :] > 1) & (hr_test[0, :] < 1.65)]
    esort = np.argsort(hr_test_1[1, :])[::-1]
    fc_hr_1 = hr_test_1[0, esort]

    # First harmonic range check index
    hr_test_2 = hr_test[:, (hr_test[0, :] > 2.015) & (hr_test[0, :] < 3.3)]
    esort = np.argsort(hr_test_2[1, :])[::-1]
    fc_hr_2 = hr_test_2[0, esort]

    if len(fc_hr_1) == 2:
        fc_hr_1 = np.append(fc_hr_1, fc_hr_1[0])
    elif len(fc_hr_1) == 1:
        fc_hr_1 = np.append(fc_hr_1, fc_hr_1[0])
        fc_hr_1 = np.append(fc_hr_1, fc_hr_1[0])

    if len(fc_hr_2) == 2:
        fc_hr_2 = np.append(fc_hr_2, fc_hr_2[0])
    elif len(fc_hr_2) == 1:
        fc_hr_2 = np.append(fc_hr_2, fc_hr_2[0])
        fc_hr_2 = np.append(fc_hr_2, fc_hr_2[0])

    if len(fc_hr_1) == 0:
        fc_hr_1 = fc_hr_2

    x_heart_radar_1 = np.round(fc_hr_1[0:3], 2)
    x_heart_radar_index1 = np.round(x_heart_radar_1 * (512 / fs))

    x_heart_radar_2 = np.round(fc_hr_2[0:3], 2)
    x_heart_radar_index2 = np.round(x_heart_radar_2 * (512 / (fs * 2)))

    index_hr = np.zeros((1, 6))
    index_hr[0, 0:3] = x_heart_radar_index1
    index_hr[0, 3:6] = x_heart_radar_index2

    # Find the best representative of frequency in the basic and
    # first harmonic top three energy values
    h, bins, idxh = Tools.histogram(index_hr[0, :], bin_width=5)

    test_index_check = bins[idxh[0]] + 3  # center of histogram width
    index_radar = test_index_check

    # HR harmonic values id estimation through RSSD
    ix0 = round(60 * 512 / (fs * 60)) + 3
    Nr = len(x_phase)
    Nchirp = 512
    init_chirp = 1
    est_window = fs
    noofestimates = round((Nr - Nchirp) / est_window)
    ihh = np.zeros((noofestimates, 6))

    time_loop_s = datetime.now()

    for p in range(noofestimates):

        end_chirp = init_chirp + Nchirp - 1
        x_sig = x_phase[int(init_chirp - 1): int(end_chirp)]
        init_chirp = (p + 1) * est_window + 1
        _, _, w1 = Tools.myemd_sec(x_sig, Q1)
        e = Tools.plot_energy(w1)
        fc = Tools.tqwt_fc(Q1, r1, J1, fs)
        hr_test = np.zeros((2, J1))
        hr_test[0, :] = fc
        hr_test[1, :] = e[0:J1]

        # Find out the HR indices for the fundamental
        hr_test_1 = hr_test[:, (hr_test[0, :] > 1) & (hr_test[0, :] < 1.65)]
        esort = np.argsort(hr_test_1[1, :])[::-1]

        fc_hr_1 = hr_test_1[0, esort]

        # First harmonic range check index
        hr_test_2 = hr_test[:, (hr_test[0, :] > 2.015) & (hr_test[0, :] < 3.3)]
        esort = np.argsort(hr_test_2[1, :])[::-1]
        fc_hr_2 = hr_test_2[0, esort]

        if len(fc_hr_1) == 2:
            fc_hr_1 = np.append(fc_hr_1, fc_hr_1[0])
        elif len(fc_hr_1) == 1:
            fc_hr_1 = np.append(fc_hr_1, fc_hr_1[0])
            fc_hr_1 = np.append(fc_hr_1, fc_hr_1[0])

        if len(fc_hr_2) == 2:
            fc_hr_2 = np.append(fc_hr_2, fc_hr_2[0])
        elif len(fc_hr_2) == 1:
            fc_hr_2 = np.append(fc_hr_2, fc_hr_2[0])
            fc_hr_2 = np.append(fc_hr_2, fc_hr_2[0])

        if len(fc_hr_1) == 0:
            fc_hr_1 = fc_hr_2

        x_heart_radar_1 = np.round(fc_hr_1[0:3], 2)
        x_heart_radar_index1 = np.round(x_heart_radar_1 * (512 / fs))

        x_heart_radar_2 = np.round(fc_hr_2[0:3], 2)
        x_heart_radar_index2 = np.round(x_heart_radar_2 * (512 / (fs * 2)))

        ihh[p, 0:3] = x_heart_radar_index1
        ihh[p, 3:6] = x_heart_radar_index2

    time_loop_e = datetime.now()
    print(f'loop time = {time_loop_e - time_loop_s}')

    time_hr_harmonic_index_save = datetime.now()
    print(f'(6) HR Harmonic index save time = {time_hr_harmonic_index_save - time_phase_extraction}')

    ############### (7) Save all hr index  ################
    hr_hex = x_HR[int(round(Nchirp / fs)): int(round(Nchirp / fs) + noofestimates), 0]
    hr_hex_ind = np.round(hr_hex * 512 / (60 * fs)).astype(int)

    hr_ind_cpm = ihh[:len(hr_hex), :]
    hr_ind_cpm = np.column_stack((hr_ind_cpm, np.full_like(hr_ind_cpm[:, 0], index_radar)))
    hr_ind_cpm = np.hstack((hr_ind_cpm, hr_hex_ind.reshape(-1, 1))).astype(int)

    time_all_hr_index_save = datetime.now()
    print(f'(7) Save all hr index time = {time_all_hr_index_save - time_hr_harmonic_index_save}')

    ############# (8) RR Harmonic Index Save  ##################
    r1 = 3
    J1 = 27

    # Find index for complete signal
    y1, y2, w1, Q1 = Tools.myemd_rr(x_phase, J1, fs)
    e = Tools.plot_energy(w1)
    fc = Tools.tqwt_fc(Q1, r1, J1, fs)
    rr_test = np.zeros((2, J1))
    rr_test[0, :] = fc
    rr_test[1, :] = e[0:J1]

    # Find RR indices
    rr_test_1 = rr_test[:, (rr_test[0, :] > 0.1) & (rr_test[0, :] < 0.6)]
    esort = np.argsort(rr_test_1[1, :])[::-1]
    fc_rr_1 = rr_test_1[0, esort]

    if len(fc_rr_1) == 2:
        fc_rr_1 = np.append(fc_rr_1, fc_rr_1[0])
    elif len(fc_rr_1) == 1:
        fc_rr_1 = np.append(fc_rr_1, fc_rr_1[0])
        fc_rr_1 = np.append(fc_rr_1, fc_rr_1[0])

    x_RR_radar_1 = np.round(fc_rr_1[0:3], 2)
    x_RR_radar_index1 = np.round(x_RR_radar_1 * (512 / fs))
    index_rr = np.zeros((1, 3), dtype=int)
    index_rr[0, :] = x_RR_radar_index1

    # Find the best representative of frequency
    h, BinEdges, idxh = Tools.histogram(index_rr[0, :], bin_width=3)
    test_index_check = BinEdges[idxh[0]] + 1  # mean of histogram width
    index_radar = int(test_index_check)

    # RR harmonic values id estimation through RSSD
    init_chirp = 1
    irr = np.zeros((noofestimates, 5), dtype=int)

    for p in range(noofestimates):
        end_chirp = init_chirp + Nchirp - 1
        x_sig = x_phase[init_chirp - 1:end_chirp]  # Python uses 0-based indexing
        init_chirp = int((p + 1) * est_window + 1)

        y1, y2, w1, J1 = Tools.myemd_sec_rr(x_sig, Q1, J1)
        e = Tools.plot_energy(w1)
        fc = Tools.tqwt_fc(Q1, r1, J1, fs)
        rr_test = np.zeros((2, J1))
        rr_test[0, :] = fc
        rr_test[1, :] = e[0:J1]

        # Find HR indices
        rr_test_1 = rr_test[:, (rr_test[0, :] > 0.08) & (rr_test[0, :] < 0.6)]
        esort = np.argsort(rr_test_1[1, :])[::-1]
        fc_rr_1 = rr_test_1[0, esort]

        if len(fc_rr_1) == 2:
            fc_rr_1 = np.append(fc_rr_1, fc_rr_1[0])
        elif len(fc_rr_1) == 1:
            fc_rr_1 = np.append(fc_rr_1, fc_rr_1[0])
            fc_rr_1 = np.append(fc_rr_1, fc_rr_1[0])

        x_RR_radar_1 = np.round(fc_rr_1[0:3], 2)
        x_RR_radar_index1 = np.round(x_RR_radar_1 * (512 / fs))
        irr[p, 0:3] = x_RR_radar_index1

    time_rr_harmonic_index_save = datetime.now()
    print(f'(8) RR Harmonic index save time = {time_rr_harmonic_index_save - time_all_hr_index_save}')

    ################# (9) Save all RR index #################
    start_index = round(Nchirp / fs)
    end_index = round(Nchirp / fs) + noofestimates

    rr_hex = x_RR[start_index: end_index, 0]
    rr_hex_ind = np.round(rr_hex * 512 / (fs * 60))
    rr_ind_cpm = irr[:len(rr_hex), :]
    rr_ind_cpm[:, 3] = index_radar
    rr_ind_cpm[:, 4] = rr_hex_ind

    time_all_rr_index_save = datetime.now()
    print(f'(9) Save all rr index time = {time_all_rr_index_save - time_rr_harmonic_index_save}')

    ################ (10) HR & RR Estimation ####################
    index_hr = np.copy(hr_ind_cpm)
    nh = index_hr[0, 6]
    hr_ind_val = np.zeros((len(index_hr), 1))

    for k in range(len(index_hr)):
        idx = np.argmin(np.abs(index_hr[k, :6] - nh))
        hr_ind_val[k, 0] = index_hr[k, idx]

    index_rr = np.copy(rr_ind_cpm)
    nr = index_rr[0, 3]
    rr_ind_val = np.zeros((len(index_rr), 1))

    for k in range(len(index_rr)):
        idx = np.argmin(np.abs(index_rr[k, :3] - nr))
        rr_ind_val[k, 0] = index_rr[k, idx]

    hr_radar = np.round(hr_ind_val * (60 * fs / 512))
    hr_hex = np.round(index_hr[:, 7] * (60 * fs / 512))
    hr_radar_avg = np.floor(np.mean(hr_radar))
    hr_hex_avg = np.floor(np.mean(hr_hex))

    rr_radar = np.round(rr_ind_val * (60 * fs / 512))
    rr_hex = np.round(index_rr[:, 4] * (60 * fs / 512))
    rr_radar_avg = np.floor(np.mean(rr_radar))
    rr_hex_avg = np.floor(np.mean(rr_hex))

    print(f"hr_radar_avg = {hr_radar_avg}, hr_hex_avg = {hr_hex_avg}")
    print(f"rr_radar_avg = {rr_radar_avg}, rr_hex_avg = {rr_hex_avg}")

    time_estimation = datetime.now()
    print(f'(10) HR & RR Estimation time = {time_estimation - time_all_rr_index_save}')
    print(f'total time usage = {time_estimation - time_begin}')
    # Save into a csv file;
    # hr_radar = hr_radar.squeeze()
    # rr_radar = rr_radar.squeeze()
    #
    # values = np.column_stack((hr_radar, rr_radar))
    # columns = ["HR", "RR"]
    # df = pd.DataFrame(values, columns=columns)
    # df.to_csv("HR_RR.csv", index=False)

    print("Done....")

    return hr_radar, rr_radar
