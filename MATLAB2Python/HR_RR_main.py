import numpy as np
from myemd import myemd
from PlotEnergy import PlotEnergy
from tqwt_fc import tqwt_fc
from myemd_sec import myemd_sec
import pandas as pd
import scipy.io
from myemd_rr import myemd_rr
from myemd_sec_rr import myemd_sec_rr
import time

start_time = time.time()  # 记录开始时间

filename_hex = f"Hex_HR_ref.xlsx"

# Read Excel file
x_HR_all = pd.read_excel(filename_hex)
# x_HR = x_HR_all.iloc[:, 1] # Assuming you want the second column (index 1)

x_HR = x_HR_all.iloc[1:, 1].values
print('shape of x_HR', x_HR.shape)
print('x_HR', x_HR)

filename_hex_rr = f"Hex_RR_ref.xlsx"
x_RR_all = pd.read_excel(filename_hex_rr)
x_RR = x_RR_all.iloc[1:, 1].values
print('shape of x_RR', x_RR.shape)
print('x_RR', x_RR)


import matplotlib.pyplot as plt
# Replace this function with the actual implementation or equivalent in Python
# Example usage of the provided code
fs = 10  # replace with the actual sampling frequency
# x_phase = np.random.randn(1024, 1)
# x_phase = np.random.randn(1024)

# # 加载MAT文件
# mat_data_1 = scipy.io.loadmat('data_cube.mat')
#
# # 获取变量x_phase的值
# x_phase = mat_data_1['x_phase'].flatten()

# read .csv

x_phase_read = pd.read_csv('x_phase_downsampled.csv', header=None)
x_phase = x_phase_read.values.flatten()
# replace with the actual signal
print('shape of x_phase', x_phase.shape)
print('x_phase', x_phase)
# Process complete signal to check for sustained oscillation dominated frequency for complete test duration
r1 = 3
J1 = 27

# Find index for the complete signal
# x_sig = x_phase[:len(x_phase), 0]
x_sig = x_phase[:len(x_phase)]

# print('len(x_sig_1)', x_sig.shape)
_, _, w1, Q1 = myemd(x_sig, fs)
e = PlotEnergy(w1, Q1, r1, fs)

if Q1 < 5:
    J1 = 22

fc = tqwt_fc(Q1, r1, J1, fs)
hr_test = np.zeros((2, J1))
hr_test[0, :] = fc
# hr_test[1, :] = e[0, 0:J1]
hr_test[1, :J1] = e[:J1]

# Find out the HR indices fundamental
hr_test_1 = hr_test[:, (hr_test[0, :] > 1)]
hr_test_1 = hr_test_1[:, (hr_test_1[0, :] < 1.65)]
esort = np.argsort(hr_test_1[1, :])[::-1]
fc_hr_1 = hr_test_1[0, esort]

# First harmonic range check index
hr_test_2 = hr_test[:, (hr_test[0, :] > 2.015)]
hr_test_2 = hr_test_2[:, (hr_test_2[0, :] < 3.3)]
esort = np.argsort(hr_test_2[1, :])[::-1]
fc_hr_2 = hr_test_2[0, esort]

if len(fc_hr_1) == 2:
    fc_hr_1 = np.append(fc_hr_1, fc_hr_1[0])
elif len(fc_hr_1) == 1:
    fc_hr_1 = np.append(fc_hr_1, [fc_hr_1[0], fc_hr_1[0]])

if len(fc_hr_2) == 2:
    fc_hr_2 = np.append(fc_hr_2, fc_hr_2[0])
elif len(fc_hr_2) == 1:
    fc_hr_2 = np.append(fc_hr_2, [fc_hr_2[0], fc_hr_2[0]])

if not np.any(fc_hr_1):
    fc_hr_1 = fc_hr_2


x_heart_radar_1 = np.round(fc_hr_1[0:3], 2)
x_heart_radar_index1 = np.round(x_heart_radar_1 * (512 / fs))
x_heart_radar_2 = np.round(fc_hr_2[0:3], 2)
x_heart_radar_index2 = np.round(x_heart_radar_2 * (512 / (fs * 2)))
index_hr = np.zeros((1, 6))
index_hr[0, 0:3] = x_heart_radar_index1
index_hr[0, 3:6] = x_heart_radar_index2
# index_hr = index_hr.flatten()
print('index_hr', index_hr)
# Find the best representative of frequency in the basic and first harmonic top three energy values
# Compute the histogram
h_values, h_edges = np.histogram(index_hr[0, :], bins=np.arange(min(index_hr[0, :]), max(index_hr[0, :]) + 6, 5))

# Sort the histogram values in descending order
idxh = np.argsort(h_values)[::-1]

# Compute the test_index_check
test_index_check = h_edges[idxh[0]] + 3

# Assign the result to index_radar
index_radar = test_index_check
# h, _ = np.histogram(index_hr[0, :], bins=range(0, int(max(index_hr[0, :])) + 6, 5))
# idxh = np.argsort(h)[::-1]
# test_index_check = h[idxh[0]] + 3  # center of histogram width
# index_radar = test_index_check
print('index_radar', index_radar)

# HR harmonic values id estimation through RSSD
ix0 = round(60 * 512 / (fs * 60)) + 3
Nr = len(x_phase)
Nchirp = 512
init_chirp = 0
est_window = fs
print('Nr', Nr)
noofestimates = round((Nr - Nchirp) / est_window)
ihh = np.zeros((noofestimates, 6))



for p in range(noofestimates):
    end_chirp = init_chirp + Nchirp - 1
    # print('x_phase', x_phase.shape)
    # print('x_phase', len(x_phase))
    x_sig = x_phase[init_chirp:end_chirp+1]
    # print('shape of len(x_sig_1)', x_sig.shape)
    # print('len(x_sig)_1', len(x_sig))
    init_chirp = (p + 1) * est_window + 1
    # print('len(x_sig)', x_sig.shape)
    # print('len(x_sig)', len(x_sig))
    _, _, w1 = myemd_sec(x_sig, Q1)
    e = PlotEnergy(w1, Q1, r1, fs)
    fc = tqwt_fc(Q1, r1, J1, fs)
    hr_test = np.zeros((2, J1))
    hr_test[0, :] = fc
    # hr_test[1, :] = e[0, 0:J1]
    hr_test[1, :J1] = e[:J1]

    # Find out the HR indices fundamental
    hr_test_1 = hr_test[:, (hr_test[0, :] > 1)]
    hr_test_1 = hr_test_1[:, (hr_test_1[0, :] < 1.65)]
    esort = np.argsort(hr_test_1[1, :])[::-1]
    fc_hr_1 = hr_test_1[0, esort]

    # First harmonic range check index
    hr_test_2 = hr_test[:, (hr_test[0, :] > 2.015)]
    hr_test_2 = hr_test_2[:, (hr_test_2[0, :] < 3.3)]
    esort = np.argsort(hr_test_2[1, :])[::-1]
    fc_hr_2 = hr_test_2[0, esort]

    if len(fc_hr_1) == 2:
        fc_hr_1 = np.append(fc_hr_1, fc_hr_1[0])
    elif len(fc_hr_1) == 1:
        fc_hr_1 = np.append(fc_hr_1, [fc_hr_1[0], fc_hr_1[0]])

    if len(fc_hr_2) == 2:
        fc_hr_2 = np.append(fc_hr_2, fc_hr_2[0])
    elif len(fc_hr_2) == 1:
        fc_hr_2 = np.append(fc_hr_2, [fc_hr_2[0], fc_hr_2[0]])

    if not np.any(fc_hr_1):
        fc_hr_1 = fc_hr_2

    x_heart_radar_1 = np.round(fc_hr_1[0:3], 2)
    x_heart_radar_index1 = np.round(x_heart_radar_1 * (512 / fs))
    x_heart_radar_2 = np.round(fc_hr_2[0:3], 2)
    x_heart_radar_index2 = np.round(x_heart_radar_2 * (512 / (fs * 2)))
    ihh[p, 0:3] = x_heart_radar_index1
    ihh[p, 3:6] = x_heart_radar_index2

print('shape of ihh', ihh.shape)
# print('ihh', ihh)

# Save all hr index
# hr_hex = np.round(x_HR[int(Nchirp / fs) + 1:int(Nchirp / fs) + noofestimates, 0])
# hr_hex = x_HR[round(Nchirp/fs):round(Nchirp/fs)+noofestimates]
hr_hex = x_HR[round(Nchirp / fs + 1):round(Nchirp / fs) + noofestimates + 1]

print('shape of hr_hex', hr_hex.shape)
print('hr_hex', hr_hex)

scaling_factor = 512 / (60 * fs)
hr_hex_ind = hr_hex * scaling_factor
# hr_hex_ind = np.around(hr_hex_ind).astype(int)
# print('hr_hex_ind', hr_hex_ind)
hr_ind_cpm = ihh[0:len(hr_hex), :]
# print('hr_ind_cpm', hr_ind_cpm)
# hr_ind_cpm[:, 6] = index_radar
if hr_ind_cpm.shape[1] < 7:
    hr_ind_cpm = np.pad(hr_ind_cpm, ((0, 0), (0, 7 - hr_ind_cpm.shape[1])), mode='constant')

# 将所有行的第7列（索引为6）赋值为 index_radar
hr_ind_cpm[:, 6] = index_radar
if hr_ind_cpm.shape[1] < 8:
    hr_ind_cpm = np.pad(hr_ind_cpm, ((0, 0), (0, 8 - hr_ind_cpm.shape[1])), mode='constant')
hr_ind_cpm[:, 7] = hr_hex_ind

# print('hr_ind_cpm:', hr_ind_cpm)
# _________________________________________________________________________
# Find index for complete signal
y1, y2, w1, Q1 = myemd_rr(x_phase, J1, fs)
e = PlotEnergy(w1, Q1, r1, fs)
fc = tqwt_fc(Q1, r1, J1, fs)
rr_test = np.zeros((2, J1))
rr_test[0, :] = fc
# rr_test[1, :] = e[0, 0:J1]
rr_test[1, :J1] = e[:J1]

# Find RR indices
rr_test_1 = rr_test[:, (rr_test[0, :] > 0.1)]
rr_test_1 = rr_test_1[:, (rr_test_1[0, :] < 0.6)]
esort = np.argsort(rr_test_1[1, :])[::-1]
fc_rr_1 = rr_test_1[0, esort]

if len(fc_rr_1) == 2:
    fc_rr_1 = np.append(fc_rr_1, fc_rr_1[0])
elif len(fc_rr_1) == 1:
    fc_rr_1 = np.append(fc_rr_1, fc_rr_1[0])
    fc_rr_1 = np.append(fc_rr_1, fc_rr_1[0])

x_RR_radar_1 = np.round(fc_rr_1[0:3], 2)
x_RR_radar_index1 = np.round(x_RR_radar_1 * (512 / fs))
index_rr = np.zeros((1, 3))
index_rr[0, :] = x_RR_radar_index1

# Find the best representative of frequency
h, _ = np.histogram(index_rr[0, :], bins=3)
idxh = np.argsort(h)[::-1]
test_index_check = h[idxh[0]] + 1  # mean of histogram width
index_radar = test_index_check

# RR harmonic values ID estimation through RSSD
init_chirp = 0
irr = np.zeros((noofestimates, 5))


for p in range(noofestimates):
    end_chirp = init_chirp + Nchirp - 1
    # x_sig = x_phase[init_chirp:end_chirp]
    x_sig = x_phase[init_chirp:end_chirp+1]

    # init_chirp = p * est_window + 1
    # print('len(x_sig)', x_sig.shape)
    # print('len(x_sig)', len(x_sig))
    init_chirp = (p + 1) * est_window + 1


    y1, y2, w1, J1 = myemd_sec_rr(x_sig, Q1, J1)
    e = PlotEnergy(w1, Q1, r1, fs)
    fc = tqwt_fc(Q1, r1, J1, fs)
    rr_test = np.zeros((2, J1))
    rr_test[0, :] = fc
    # rr_test[1, :] = e[0, 0:J1]
    rr_test[1, :J1] = e[:J1]

    # Find RR indices
    # rr_test_1 = rr_test[:, (rr_test[0, :] > 0.08)]
    rr_test_1 = rr_test[:, (rr_test[0, :] > 0.08)]
    # rr_test_1 = rr_test_1[:, (rr_test_1[0, :] < 0.6)]
    rr_test_1 = rr_test_1[:, (rr_test_1[0, :] < 1)]
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

    rr_hex = x_RR[round(Nchirp / fs + 1):round(Nchirp / fs) + noofestimates + 1]

    # print('shape of rr_hex', rr_hex.shape)
    # print('rr_hex', rr_hex)

    scaling_factor_rr = 512 / (60 * fs)
    rr_hex_ind = rr_hex * scaling_factor_rr
    # hr_hex_ind = np.around(hr_hex_ind).astype(int)
    # print('rr_hex_ind', rr_hex_ind)
    rr_ind_cpm = irr[0:len(rr_hex), :]
    # print('hr_ind_cpm', rr_ind_cpm)
    # hr_ind_cpm[:, 6] = index_radar
    if rr_ind_cpm.shape[1] < 4:
        rr_ind_cpm = np.pad(rr_ind_cpm, ((0, 0), (0, 7 - rr_ind_cpm.shape[1])), mode='constant')

    # 将所有行的第7列（索引为6）赋值为 index_radar
    rr_ind_cpm[:, 3] = index_radar
    if rr_ind_cpm.shape[1] < 5:
        rr_ind_cpm = np.pad(rr_ind_cpm, ((0, 0), (0, 8 - rr_ind_cpm.shape[1])), mode='constant')
    rr_ind_cpm[:, 4] = rr_hex_ind

# _________________________________________________________________________
index_hr = hr_ind_cpm
nh = index_hr[0, 6]

# Initialize hr_ind_val array
hr_ind_val = np.zeros((len(index_hr), 1))

# Iterate through each row of index_hr
for k in range(len(index_hr)):
    # Find the index of the minimum absolute difference
    idx = np.argmin(np.abs(index_hr[k, :6] - nh))
    hr_ind_val[k, 0] = index_hr[k, idx]

# Calculate hr_radar using the given formula
hr_radar = np.round(hr_ind_val * (60 * fs / 512))

# Calculate hr_hex using the given formula
hr_hex = np.round(index_hr[:, 7] * (60 * fs / 512))

# Calculate average heart rates
hr_radar_avg = np.floor(np.mean(hr_radar))
hr_hex_avg = np.floor(np.mean(hr_hex))

# Print the results
print("hr_radar:", hr_radar)
print("hr_hex:", hr_hex)
print("hr_radar_avg:", hr_radar_avg)
print("hr_hex_avg:", hr_hex_avg)


index_rr = rr_ind_cpm
nr = index_rr[0, 3]

# Initialize hr_ind_val array
rr_ind_val = np.zeros((len(index_rr), 1))

# Iterate through each row of index_hr
for k in range(len(index_rr)):
    # Find the index of the minimum absolute difference
    idx = np.argmin(np.abs(index_rr[k, :3] - nr))
    rr_ind_val[k, 0] = index_rr[k, idx]


# Calculate hr_radar using the given formula
rr_radar = np.round(rr_ind_val * (60 * fs / 512))

# Calculate hr_hex using the given formula
rr_hex = np.round(index_rr[:, 4] * (60 * fs / 512))

# Calculate average heart rates
rr_radar_avg = np.floor(np.mean(rr_radar))
rr_hex_avg = np.floor(np.mean(rr_hex))

# Print the results
print("rr_radar:", rr_radar)
print("rr_hex:", rr_hex)
print("rr_radar_avg:", rr_radar_avg)
print("rr_hex_avg:", rr_hex_avg)

end_time = time.time()  # 记录结束时间

execution_time = end_time - start_time  # 计算执行时间
print(f"Execution time: {execution_time} seconds")
