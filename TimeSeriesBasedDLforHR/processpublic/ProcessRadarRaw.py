
import biosppy.signals.ecg as ecg
import matplotlib.pyplot as plt
import os
import scipy
from pathlib import Path
import re
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from datetime import datetime, timedelta

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

data_splits = {
    "train": 22,
    "val": 4,
    "test": 3
}

def plot_radar_raw_signal(participant):
    fs = 2000
    radar_phase = __get_radar_phase(participant)

    x_seconds = len(radar_phase) / fs
    x_times = np.array([i for i in np.arange(0, x_seconds, x_seconds / len(radar_phase))])

    plt.figure(figsize=(12, 6))
    plt.plot(x_times, radar_phase, color='blue')
    plt.title('Radar Raw Signal Data')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

def get_measure_time(participant):
    '''
    get the start time of measurement for the participant.
    :param participant:
    :return:
    '''
    mat_data = __get_mat_data(participant)
    measurement_info = "measurement_info"
    measure_time = mat_data[measurement_info][0][0][0]
    return measure_time

def __get_radar_phase(participant):
    mat_data = __get_mat_data(participant)
    radar_i = None
    radar_q = None
    measure_time = None
    # Access variables in the loaded data
    name_i = 'radar_i'
    name_q = "radar_q"
    measurement_info = "measurement_info"
    if name_i in mat_data and name_q in mat_data:
        radar_i = mat_data[name_i]
        radar_q = mat_data[name_q]
        measure_time = mat_data[measurement_info][0][0][0]
    radar_i = radar_i.squeeze()
    radar_q = radar_q.squeeze()
    radar_complex = radar_i + 1j * radar_q
    radar_phase = radar_complex

    return radar_phase, measure_time


def __get_ecg_phase(participant):
    ecg_phase = None
    mat_data = __get_mat_data(participant)

    # Access variables in the loaded data
    variable_name = 'tfm_ecg1'
    if variable_name in mat_data:
        ecg_phase = mat_data[variable_name]

    ecg_phase = ecg_phase.squeeze()
    return ecg_phase


def __get_mat_data(participant):
    folder = "GDN000" if int(participant) < 10 else "GDN00"
    mat_file_path = os.path.join(root_path, "publicdata/" + folder + str(participant) + "/" + folder + str(
        participant) + "_1_Resting.mat")
    mat_data = scipy.io.loadmat(mat_file_path)
    return mat_data


def process_radar_raw(participant):
    fs = 2000
    window_size = 5

    radar_phase, measure_time = __get_radar_phase(participant)
    ecg_phase = __get_ecg_phase(participant)

    x_seconds = len(radar_phase) / fs
    pd_columns = ["fs_" + str(i) for i in range(1, fs * window_size + 1)]
    pd_columns.append("hr")
    pd_X_values = []

    for i in range(int(x_seconds) - window_size + 1):
        X_ecg_phase = ecg_phase[i * fs: (fs * window_size + i * fs)]
        ecg_result = ecg.ecg(signal=X_ecg_phase, sampling_rate=fs, show=False)  # show=True, plot the img.

        heart_rate = ecg_result['heart_rate']
        y_hr = round(np.mean(heart_rate))

        X_radar_phase = radar_phase[i * fs: (fs * window_size + i * fs)]
        row_data = X_radar_phase

        row_data = np.append(row_data, y_hr)

        pd_X_values.append(row_data)

    saved_directory = os.path.join("../publicdata/dataset/raw", "window_size_" + str(window_size))
    if not os.path.exists(saved_directory):
        os.makedirs(saved_directory)

    saved_file_path = os.path.join(saved_directory, "Train_raw_000" + str(participant) + "_Resting.csv")
    df = pd.DataFrame(pd_X_values, columns=pd_columns)

    df.to_csv(saved_file_path)

def process_radar_raw_with_date(participant):
    '''
    generate date for the Dataset
    :param participant:
    :return:
    '''
    fs = 2000
    window_size = 5

    radar_phase, measure_time = __get_radar_phase(participant)
    ecg_phase = __get_ecg_phase(participant)

    x_seconds = len(radar_phase) / fs
    X_pd_columns = ["fs_" + str(i) for i in range(1, fs + 1)]
    X_pd_columns.insert(0, "date")
    X_pd_columns.insert(0, "id")
    y_pd_columns = ["id", "hr"]

    X_pd_values = []
    y_pd_values = []

    dt = datetime.strptime(measure_time, '%Y-%m-%d_%H-%M-%S')

    for i in range(int(x_seconds) - window_size + 1):
        X_ecg_phase = ecg_phase[i * fs: (fs * window_size + i * fs)]
        ecg_result = ecg.ecg(signal=X_ecg_phase, sampling_rate=fs, show=False)  # show=True, plot the img.

        heart_rate = ecg_result['heart_rate']
        y_hr = round(np.mean(heart_rate))

        X_radar_phase = radar_phase[i * fs: (fs * window_size + i * fs)]
        real_part = X_radar_phase.real
        imag_part = X_radar_phase.imag
        row_data = np.column_stack((real_part, imag_part))

        # Initialize PCA with 1 component
        pca = PCA(n_components=1)
        # Fit and transform the data
        reduced_row_data = pca.fit_transform(row_data)
        reduced_row_data = reduced_row_data.squeeze()

        reduced_row_data = reduced_row_data.reshape(window_size, fs)

        # X_assembled_data = []
        # y_assembled_data = []
        for index, row in enumerate(reduced_row_data):
            # Add one second
            dt += timedelta(seconds=1)
            # Convert the datetime object back into a string
            latest_datetime = dt.strftime('%Y-%m-%d %H:%M:%S')

            new_row = [i] + [latest_datetime] + row.tolist()
            X_pd_values.append(new_row)

        y_pd_values.append([i] + [y_hr])

    saved_directory = os.path.join("../publicdata/dataset", "cross_train_date")
    if not os.path.exists(saved_directory):
        os.makedirs(saved_directory)

    X_saved_file_path = os.path.join(saved_directory, "X_raw_" + str(participant) + "_Resting.csv")
    y_saved_file_path = os.path.join(saved_directory, "y_raw_" + str(participant) + "_Resting.csv")

    X_df = pd.DataFrame(X_pd_values, columns=X_pd_columns)
    y_df = pd.DataFrame(y_pd_values, columns=y_pd_columns)

    X_df.to_csv(X_saved_file_path)
    y_df.to_csv(y_saved_file_path)

def prepare_train_val_data():
    parent_dir = str(Path.cwd().parent)
    dataset_directory = parent_dir + "/publicdata/Dataset/"
    raw_dir = dataset_directory + "raw/window_size_5/"
    train_dir = dataset_directory + "train"
    val_dir = dataset_directory + "val"
    test_dir = dataset_directory + "test"

    file_names = [file_name for file_name in os.listdir(raw_dir) if file_name.startswith('Train_raw_000')]
    file_names = sorted(file_names, key=lambda x: int(re.findall(r'\d+', x)[0]))
    num_val = data_splits["val"]
    num_test = data_splits["test"]

    # create and save data for validation
    for i in range(1, num_val+1):
        last_file_name = file_names.pop()
        _pca_to_csv(last_file_name, val_dir)
        print(f"validation done: {last_file_name}")

    # create and save data for test
    for i in range(1, num_test + 1):
        last_file_name = file_names.pop()
        _pca_to_csv(last_file_name, test_dir)
        print(f"test done: {last_file_name}")

    # create and save data for training
    for index, file_name in enumerate(file_names):
        _pca_to_csv(file_name, train_dir)
        print(f"training done: {file_name}")


def _pca_to_csv(file_name, save_dir):
    parent_dir = str(Path.cwd().parent)
    dataset_directory = parent_dir + "/publicdata/Dataset/"
    raw_dir = dataset_directory + "raw/window_size_5/"

    df_dataset = pd.read_csv(raw_dir + file_name)
    last_column = "fs_10000"
    df_X = df_dataset.loc[:, "fs_1":last_column]
    X = df_X.to_numpy()
    X_comp = np.complex64(X)
    X_train_r = X_comp.real
    X_train_i = X_comp.imag
    X = np.dstack((X_train_r, X_train_i)).reshape(X_train_r.shape[0], X_train_r.shape[1], 2)
    y = np.complex64(df_dataset["hr"]).real
    # Data dimensionality reduction
    X_flattened = X.reshape(len(X), -1)
    X_length = 590
    pca = PCA(n_components=X_length)
    X = pca.fit_transform(X_flattened)
    df_values = np.concatenate((X, y.reshape(len(y), 1)), axis=1)
    pd_columns = ["f_" + str(i) for i in range(1, X_length + 1)]
    pd_columns.append("hr")
    num_participant = int(re.findall(r'\d+', file_name)[0])
    saved_file_path = os.path.join(save_dir, "raw_" + str(num_participant) + "_Resting.csv")
    df = pd.DataFrame(df_values, columns=pd_columns)
    df.to_csv(saved_file_path)


def plot_ecg_radar_raw_signal(participant):
    fs = 2000
    window_size = 5

    radar_phase = np.array(__get_radar_phase(participant))
    normalized_radar_phase = (radar_phase - np.min(radar_phase)) / (np.max(radar_phase) - np.min(radar_phase))
    ecg_phase = __get_ecg_phase(participant)

    x_seconds = len(ecg_phase) / fs

    x_times = np.array([i for i in np.arange(0, x_seconds, x_seconds / len(ecg_phase))])
    ecg_phase = ecg_phase.squeeze()
    x_start = 20 * fs
    x_end = 30 * fs

    x_value = x_times[x_start:x_end]
    x_value = x_value - x_times[x_start]
    y_value_ECG = ecg_phase[x_start:x_end]
    y_value_Radar = normalized_radar_phase[x_start:x_end]

    plt.plot(x_value, y_value_ECG, color='#E09302', linewidth=0.9, label='ECG Signal')
    plt.plot(x_value, y_value_Radar, color='blue', linewidth=0.9, label='Radar Raw Signal')
    plt.xlabel("Time (Seconds)")
    # plt.ylabel("Amplitude")
    plt.yticks([])
    plt.xticks([i for i in range(11)])
    # plt.grid()
    plt.legend()
    # plt.savefig(model_path + '.png')
    plt.show()

# plot_ecg_radar_raw_signal(2)
# process_radar_raw_with_date(2)


# for i in range(4, 31):
#     process_radar_raw_with_date(i)
#     print(f'{i} done...')

# prepare_train_val_data()