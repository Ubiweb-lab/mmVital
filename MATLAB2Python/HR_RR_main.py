
import numpy as np
from myemd import myemd
from PlotEnergy import PlotEnergy
from tqwt_fc import tqwt_fc
from myemd_sec import myemd_sec
import pandas as pd
from myemd_rr import myemd_rr
from myemd_sec_rr import myemd_sec_rr
import csv
import time

start_time = time.time()  # 记录开始时间

c = 3e8
fc = 77e9
lambda_ = c / fc

# Specify the path to your binary file
file_path = 'adc_data.bin'

# Read the binary file as int16 data
adc_data = np.fromfile(file_path, dtype=np.int16)
# print('shape of adc_data', adc_data.shape)
if len(adc_data) > 20971520:
    adc_data  = adc_data[:20971520]

# print('shape of adc_data', adc_data.shape)

# Reshape the data to separate real and imaginary parts
num_antennas = 4
num_samples = len(adc_data) // (2 * num_antennas)
adc_data = adc_data.reshape((num_samples, num_antennas, 2))

# Define array geometry
ula_spacing = lambda_ / 2
array_geometry = np.arange(0, num_antennas) * ula_spacing

# Calculate steering vector
beamforming_direction = np.array([0, 0])
steering_vector = np.exp(1j * 2 * np.pi * fc / c * array_geometry * np.sin(np.radians(beamforming_direction[0])))

# Reshape steering vectors to match dimensions of adc_data
sv_reshaped = steering_vector.reshape((1, num_antennas, 1))

# Apply steering vectors to ADC data
beamformed_signal = adc_data * sv_reshaped
#
# print("beamformed_signal:")
# print(beamformed_signal)

# Sum across antennas
combined_signal = np.sum(beamformed_signal, axis=1)

# print("combined_signal:")
# print(combined_signal)

# Convert combined_signal to 1D array
combined_signal_1d = np.squeeze(combined_signal)



# Reshape to (1, n)
combined_signal_reshaped = np.reshape(combined_signal_1d, (1, -1))



# print('shaple of combined_signal_reshaped', combined_signal_reshaped.shape)
# print("combined_signal_reshaped:")
print(combined_signal_reshaped)

bf_data = combined_signal_reshaped.ravel()
# print('shaple of bf_data ', bf_data.shape)

# Optional: If you want to visualize the results, you can plot the beamformed signal in time and frequency domains
# For example, you can use matplotlib for plotting

import matplotlib.pyplot as plt

# Plot time-domain signal
# plt.figure(figsize=(10, 4))
# plt.plot(np.real(combined_signal))
# plt.title('Beamformed Signal (Time Domain)')
# plt.xlabel('Sample')
# plt.ylabel('Amplitude')
# plt.grid(True)
# plt.show()
#
# # Plot frequency-domain signal (optional)
# beamformed_signal_fft = np.fft.fft(combined_signal, axis=0)
# frequency_axis = np.fft.fftfreq(num_samples)
# plt.figure(figsize=(10, 4))
# plt.plot(frequency_axis, 20 * np.log10(np.abs(beamformed_signal_fft)))
# plt.title('Beamformed Signal (Frequency Domain)')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Magnitude (dB)')
# plt.grid(True)
# plt.show()

#-----------------------------------DC Offset correction----------------------------------
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

Bi = np.real(bf_data)
Bq = np.imag(bf_data)
Ar = np.sqrt(np.sum((Bi**2) + (Bq**2)))

def fun(dc):
    return np.abs(np.sqrt((Bi - dc[0])**2 + (Bq - dc[1])**2) - dc[2])

dc0 = [0, 0, Ar]
options = {'method': 'lm'}  # Levenberg-Marquardt algorithm

result = least_squares(fun, dc0, **options)
dcnew = result.x

Bin = Bi - dcnew[0]
Bqn = Bq - dcnew[1]

# plt.figure(1)
# plt.clf()
# plt.plot(Bq, Bi, 'o', color='y', label='Original')
# plt.plot(Bqn, Bin, '*', label='Offset Corrected')
# plt.title('DC Offset Correction')
# plt.legend()
# plt.xlabel('x')
# plt.ylabel('y')
# plt.axis('equal')

# bf_signal_dccorrected = complex(Bin, Bqn)
bf_signal_dccorrected = Bin + 1j * Bqn

# plt.show()

#-----------------------------------Data cube formation----------------------------------

Nsample = 128
Nrow = round(len(bf_data) / Nsample)
end_val = 0
data_cube = np.zeros((Nrow, Nsample), dtype=complex)
data_cube_complete = np.zeros((Nrow, Nsample), dtype=complex)



for i in range(1, Nrow+1):
    init_val = end_val + 1
    end_val = Nsample * i
    data_cube[i-1, :] = bf_data[init_val-1:end_val]
    data_cube_complete[i-1, :] = bf_signal_dccorrected[init_val-1:end_val]

data_cube_locs = np.diff(data_cube_complete)

# print('shape of data_cube_locs', data_cube_locs.shape)
# print('data_cube_locs', data_cube_locs)
# np.save('data_cube_locs.npy', data_cube_locs)
# np.save('data_cube_complete.npy', data_cube_complete)




def is_power_of_two(num):
    return num > 0 and (num & (num - 1)) == 0
# mat_data_1 = scipy.io.loadmat('data_cube.mat', variable_names='data_cube_locs')
#
# # extract the variable
# data_cube_locs = mat_data_1['data_cube_locs']
#
# # read variable from MATLAB '.mat' file
# mat_data = scipy.io.loadmat('data_cube.mat', variable_names='data_cube_complete')
#
# # extract the variable
# data_cube_complete = mat_data['data_cube_complete']

# 加载保存的数组文件
# data_cube_locs = np.load('data_cube_locs.npy')
# data_cube_complete = np.load('data_cube_complete.npy')

# Print the shape of the reshaped data
# print("Shape of data_cube_locs:", data_cube_locs.shape)
# print("data_cube_locs:", data_cube_locs)
# print("Shape of data_cube_complete:", data_cube_complete.shape)
# print("data_cube_complete:", data_cube_complete)


# Print the shape of the reshaped data
# print("Shape of data_cube_complete:", data_cube_complete.shape)
# print("data_cube_complete:", data_cube_complete)

# Assuming data_cube_locs_complex is a NumPy array
Nsample = data_cube_locs.shape[1]
Nsample_1 = data_cube_complete.shape[1]
#Nrow = data_cube_locs.shape[0]

# Range bin finding through FFT on fast time samples
range_slow_mat_locs = np.fft.fft(data_cube_locs, Nsample, axis=1)
# print('range_slow_mat_locs:', range_slow_mat_locs.shape)
# print('range_slow_mat_locs:', range_slow_mat_locs)
# plt.figure(10)
# plt.imshow(20 * np.log10(np.abs(range_slow_mat_locs)), cmap='viridis', aspect='auto')  # Equivalent to mag2db
# plt.colorbar(label='Magnitude (dB)')

# Power in the range domain
pow_range = np.abs(range_slow_mat_locs)**2
#print('pow_range:', pow_range.shape)
#print('pow_range:', pow_range)
mean_pow_range = np.mean(pow_range, axis=0)
#print('mean_pow_range:', mean_pow_range.shape)
#print('mean_pow_range:', mean_pow_range)
pks1, locs1 = np.max(mean_pow_range[:50]), np.argmax(mean_pow_range[:50])
#print('pks1:', pks1) #6.62005237e+08
# print('locs1:', locs1) #11 in Python



# Range bin finding through FFT on fast time samples
range_slow_mat = np.fft.fft(data_cube_complete, Nsample_1, axis=1)  # Use data_cube_locs_complex here
# print('range_slow_mat:', range_slow_mat.shape)
# print('range_slow_mat:', range_slow_mat)
signal_target = range_slow_mat[:, locs1]
sig_i, sig_q = np.real(signal_target), np.imag(signal_target)
# print('sig_i:', sig_i)
# print('sig_q:', sig_q)


nmer = np.zeros_like(sig_i)
dnmer = np.zeros_like(sig_i)

for k in range(1, len(sig_i)):
    nmer_k = sig_i[k] * (sig_q[k] - sig_q[k-1]) - (sig_i[k] - sig_i[k-1]) * sig_q[k]
    nmer[k] = nmer_k
    dnmer[k] = sig_q[k] ** 2 + sig_i[k] ** 2

ph_term = np.zeros_like(nmer)

# Check for zero values in dnmer
nonzero_mask = dnmer != 0

# Perform division only for non-zero values
ph_term[nonzero_mask] = nmer[nonzero_mask] / dnmer[nonzero_mask]

# Compute x_phase
#nmer = sig_i * (np.roll(sig_q, 1) - sig_q) - (np.roll(sig_i, 1) - sig_i) * sig_q
# print('nmer:', nmer)
# print('shape of nmer:', nmer.shape)
#dnmer = sig_q**2 + sig_i**2
# print('dnmer:', dnmer)
# print('shape of dnmer:', dnmer.shape)
#ph_term = nmer / dnmer
# print('ph_term:', ph_term)
x_phase = np.cumsum(ph_term[1:])  # Skip the first element
# print('x_phase:', x_phase.shape)


# Sampling frequency of the signal: slow time sampling rate: frame rate
# For 1443: frame duration is 5 msec, 1 chirp per frame hence slow time sampling rate is 1/5 msec
frame_duration = 5  # in msec
fs = 1000 / frame_duration
down_sampling_factor = 20  # For 1443 downsample by 20, fs=10 Hz, for 1843, fs=10 Hz after downsampling

# Now the frame rate = 100 msec
fs = fs / down_sampling_factor  # fs = 10 Hz for 1443, fs = 10 Hz for 1843
# print('fs', fs)




# # -----------------------------------try to cancel down sampling and reshape the x_phase----------------------------
x_phase = x_phase[::down_sampling_factor]

# example
# length_x_phase = len(x_phase)
#
# # if len(x_phase) is not power of 2
# while not is_power_of_two(length_x_phase):
#     x_phase = x_phase[:-1]  # delete last element
#     length_x_phase = len(x_phase)


# -----------------------------------try  end ----------------------------
# print('x_phase:', x_phase.shape)
#print('x_phase:', x_phase)
# print('x_phase (first 10 elements):', x_phase[:10])  #the value will be little bit diference after MATLAB downsample
N = len(x_phase)
t = np.arange(0, N) / fs  # Time axis

# Save x_phase to CSV file
# np.savetxt('x_phase_downsampled.csv', np.column_stack((t, x_phase.real)), delimiter=',', header='Time,Phase', comments='')
# np.savetxt('x_phase_downsampled.csv', x_phase, delimiter=',', fmt='%f')
# plt.figure(5)
# plt.plot(t, x_phase.real, 'b')  # Plot the real part of x_phase
# plt.xlabel('Time (s)')
# plt.ylabel('Phase (Real Part)')
# plt.title('Downsampled Phase')
# plt.show()


# --------------------HR_RR--------------------------

filename_hex = f"Hex_HR_ref.xlsx"

# Read Excel file
x_HR_all = pd.read_excel(filename_hex)
# x_HR = x_HR_all.iloc[:, 1] # Assuming you want the second column (index 1)

x_HR = x_HR_all.iloc[1:, 1].values
# print('shape of x_HR', x_HR.shape)
# print('x_HR', x_HR)

filename_hex_rr = f"Hex_RR_ref.xlsx"
x_RR_all = pd.read_excel(filename_hex_rr)
x_RR = x_RR_all.iloc[1:, 1].values
# print('shape of x_RR', x_RR.shape)
# print('x_RR', x_RR)


import matplotlib.pyplot as plt
# Replace this function with the actual implementation or equivalent in Python
# Example usage of the provided code
fs = 10  # replace with the actual sampling frequency
# x_phase = np.random.randn(1024, 1)
# x_phase = np.random.randn(1024)


# mat_data_1 = scipy.io.loadmat('data_cube.mat')
# x_phase = mat_data_1['x_phase'].flatten()

# read .csv

# x_phase_read = pd.read_csv('x_phase_downsampled.csv', header=None)
# x_phase = x_phase_read.values.flatten()
# replace with the actual signal
# print('shape of x_phase', x_phase.shape)
# print('x_phase', x_phase)
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
# print('index_hr', index_hr)
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
# print('index_radar', index_radar)

# HR harmonic values id estimation through RSSD
ix0 = round(60 * 512 / (fs * 60)) + 3
Nr = len(x_phase)
Nchirp = 512
init_chirp = 0
est_window = fs
# print('Nr', Nr)
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

# print('shape of ihh', ihh.shape)
# print('ihh', ihh)

# Save all hr index
# hr_hex = np.round(x_HR[int(Nchirp / fs) + 1:int(Nchirp / fs) + noofestimates, 0])
# hr_hex = x_HR[round(Nchirp/fs):round(Nchirp/fs)+noofestimates]
hr_hex = x_HR[round(Nchirp / fs + 1):round(Nchirp / fs) + noofestimates + 1]

# print('shape of hr_hex', hr_hex.shape)
# print('hr_hex', hr_hex)

scaling_factor = 512 / (60 * fs)
hr_hex_ind = hr_hex * scaling_factor
# hr_hex_ind = np.around(hr_hex_ind).astype(int)
# print('hr_hex_ind', hr_hex_ind)
hr_ind_cpm = ihh[0:len(hr_hex), :]
# print('hr_ind_cpm', hr_ind_cpm)
# hr_ind_cpm[:, 6] = index_radar
if hr_ind_cpm.shape[1] < 7:
    hr_ind_cpm = np.pad(hr_ind_cpm, ((0, 0), (0, 7 - hr_ind_cpm.shape[1])), mode='constant')

#  index_radar
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

# 将 rr_radar 和 hr_radar 中的元素逐行写入 CSV 文件
with open('rr_hr_data.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # 写入数据
    for rr, hr in zip(rr_radar, hr_radar):
        #writer.writerow([rr, hr])
        writer.writerow([str(rr).strip('[]'), str(hr).strip('[]')])

end_time = time.time()

execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")




