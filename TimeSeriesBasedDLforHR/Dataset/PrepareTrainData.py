import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

from pathlib import Path
import re
from datetime import datetime, timedelta

from processpublic.ProcessRadarRaw import get_measure_time
from utils.timefeatures import time_features


class RadarDataset(Dataset):
    def __init__(self, X, y, X_date_stamp=None, y_date_stamp=None):
        self.X = X
        self.y = y
        self.X_date_stamp = X_date_stamp
        self.y_date_stamp = y_date_stamp

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        if self.X_date_stamp is None:
            return self.X[index], self.y[index]
        else:
            X_mask = self.X_date_stamp[index]
            y_mask = self.y_date_stamp[index][(len(self.y_date_stamp[index])-2):]
            # y_mask = y_mask.view(1, len(y_mask))

            X_seq = self.X[index]
            y_seq = self.y[index]
            y_seq = torch.cat((y_seq, y_seq)).view(2,1)
            return X_seq, y_seq, X_mask, y_mask


class PrepareTrainData:
    def __init__(self, is_date=False, seq_length=5, n_features=118, batch_size=16, is_shuffle=False):
        self.seq_length = seq_length
        self.n_features = n_features
        self.batch_size = batch_size
        self.is_shuffle = is_shuffle
        self.is_date = is_date

    def get_trainval_data(self, is_train=True):
        parent_dir = str(Path.cwd().parent.parent)

        df_dataset = None
        sub_folder = "train/" if is_train is True else "val/"
        dataset_directory = parent_dir + "/publicdata/Dataset/" + sub_folder
        file_names = [file_name for file_name in os.listdir(dataset_directory) if file_name.startswith("raw_")]

        for i, file_name in enumerate(file_names):
            file_path = os.path.join(dataset_directory, file_name)
            df = pd.read_csv(file_path)
            df_dataset = pd.concat([df_dataset, df], ignore_index=True)
            print(f'file_name-{i} = {file_path}')

            if i > 3:
                break

        last_column = "f_590"
        df_X= df_dataset.loc[:, "f_1":last_column]

        X_data = df_X.to_numpy()
        y_data = df_dataset["hr"].to_numpy()
        print(f'x_data length = {X_data.shape}')

        return X_data, y_data

    def get_val_data(self, participant):
        # parent_dir = str(Path.cwd().parent)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        dataset_directory = parent_dir + "/publicdata/Dataset/cross_train/"
        file_name = "raw_" + str(participant) + "_Resting.csv"
        file_path = os.path.join(dataset_directory, file_name)
        df = pd.read_csv(file_path)
        last_column = "f_590"
        df_X_val = df.loc[:, "f_1":last_column]

        X_val = df_X_val.to_numpy()
        y_val = df["hr"].to_numpy()

        return X_val, y_val

    def get_test_data(self, participant):
        parent_dir = str(Path.cwd().parent)
        dataset_directory = parent_dir + "/publicdata/Dataset/test/"
        file_name = "raw_" + str(participant) + "_Resting.csv"
        file_path = os.path.join(dataset_directory, file_name)
        df = pd.read_csv(file_path)
        last_column = "f_590"
        df_X_val = df.loc[:, "f_1":last_column]

        X_val = df_X_val.to_numpy()
        y_val = df["hr"].to_numpy()

        return X_val, y_val

    def train_dataloader(self):
        X_train, y_train = self.get_trainval_data()
        return self.get_dataloader(X_train, y_train)

    def val_dataloader(self, participant):
        X_val, y_val = self.get_val_data(participant)
        return self.get_dataloader(X_val, y_val)

    def test_dataloader(self, participant):
        X_test, y_test = self.get_test_data(participant)
        return self.get_dataloader(X_test, y_test)

    def get_dataloader(self, X_data, y_data):
        X_data = X_data.reshape(len(X_data), self.seq_length, self.n_features)
        y_data = y_data.reshape(len(y_data), 1)
        X_data = torch.tensor(X_data).float()
        y_data = torch.tensor(y_data).float()
        dataset = RadarDataset(X_data, y_data)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.is_shuffle)
        return dataloader

    def get_cross_dataloaders(self, participant_id):
        '''
        Prepare dataloader for cross validation.
        :param participant_id:
        :return:
        '''
        parent_dir = str(Path.cwd().parent)
        dataset_directory = parent_dir + "/publicdata/Dataset/cross_train/"
        val_file_name = "raw_" + str(participant_id) + "_Resting.csv"
        df_dataset = None

        # prepare train dataloader
        file_names = [file_name for file_name in os.listdir(dataset_directory) if file_name.startswith("raw_") and file_name != val_file_name]
        file_names = sorted(file_names, key=lambda x: int(re.findall(r'\d+', x)[0]))
        for i, file_name in enumerate(file_names):
            file_path = os.path.join(dataset_directory, file_name)
            # print(f'training file_name-{i} = {file_path}')
            df = pd.read_csv(file_path)
            df_dataset = pd.concat([df_dataset, df], ignore_index=True)

            if i>1:
                break

        last_column = "f_590"
        df_train_X = df_dataset.loc[:, "f_1":last_column]

        Train_X_data = df_train_X.to_numpy()
        Train_y_data = df_dataset["hr"].to_numpy()
        # print(f'Train_X_data shape = {Train_X_data.shape}, Train_y_data shape = {Train_y_data.shape}')
        train_X = Train_X_data.reshape(len(Train_X_data), self.seq_length, self.n_features)
        train_y = Train_y_data.reshape(len(Train_y_data), 1)
        train_X = torch.tensor(train_X).float()
        train_y = torch.tensor(train_y).float()
        train_dataset = RadarDataset(train_X, train_y)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.is_shuffle)

        # prepare validation train dataloader
        df_val = pd.read_csv(os.path.join(dataset_directory, val_file_name))

        df_val_X = df_val.loc[:, "f_1":last_column]
        val_X_data = df_val_X.to_numpy()
        val_y_data = df_val["hr"].to_numpy()
        # print(f'val_X_data shape = {val_X_data.shape}, val_y_data shape = {val_y_data.shape}')
        val_X = val_X_data.reshape(len(val_X_data), self.seq_length, self.n_features)
        val_y = val_y_data.reshape(len(val_y_data), 1)
        val_X = torch.tensor(val_X).float()
        val_y = torch.tensor(val_y).float()
        val_dataset = RadarDataset(val_X, val_y)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=self.is_shuffle)

        return train_dataloader, val_dataloader

    def get_cross_dataloaders_add_date(self, participant_id):
        '''
                Prepare dataloader for cross validation using manually add date into data (118 features).
                :param participant_id:
                :return:
        '''
        parent_dir = str(Path.cwd().parent)
        dataset_directory = parent_dir + "/publicdata/dataset/cross_train/"
        val_file_name = "raw_" + str(participant_id) + "_Resting.csv"

        train_X = []
        train_y = []

        X_date_stamp = []
        y_date_stamp = []

        last_column = "f_590"

        # prepare train dataloader
        file_names = [file_name for file_name in os.listdir(dataset_directory) if
                      file_name.startswith("raw_") and file_name != val_file_name]
        file_names = sorted(file_names, key=lambda x: int(re.findall(r'\d+', x)[0]))
        for i, file_name in enumerate(file_names):
            pt_pattern = r'\d+'
            pt_id = re.search(pt_pattern, file_name).group()
            measure_time = get_measure_time(pt_id)
            dt = datetime.strptime(measure_time, '%Y-%m-%d_%H-%M-%S')

            file_path = os.path.join(dataset_directory, file_name)
            # print(f'training file_name-{i} = {file_path}')
            df = pd.read_csv(file_path)
            X_data_all = df.loc[:, "f_1":last_column].to_numpy()
            y_data_all = df["hr"].to_numpy()
            for idx in range(X_data_all.shape[0]):
                X_data = X_data_all[idx].reshape(self.seq_length, self.n_features)
                train_X.append(X_data)
                train_y.append([y_data_all[idx]])

                # X_date  = []
                for j in range(self.seq_length):
                    dt += timedelta(seconds=1)
                    # Convert the datetime object back into a string
                    latest_datetime = dt.strftime('%Y-%m-%d %H:%M:%S')
                    X_date_stamp.append(latest_datetime)

                    dt_day = dt
                    dt_day += timedelta(days=1)
                    latest_datetime_d = dt_day.strftime('%Y-%m-%d %H:%M:%S')
                    y_date_stamp.append(latest_datetime_d)


            if i > 1:
                break

        train_X_date_stamp = self._refactor_date_stamp(X_date_stamp)
        train_y_date_stamp = self._refactor_date_stamp(y_date_stamp)

        train_X = torch.tensor(train_X).float()
        train_y = torch.tensor(train_y).float()
        train_X_date_stamp = torch.tensor(train_X_date_stamp).float()
        train_y_date_stamp = torch.tensor(train_y_date_stamp).float()
        train_dataset = RadarDataset(train_X, train_y, X_date_stamp=train_X_date_stamp, y_date_stamp=train_y_date_stamp)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.is_shuffle)

        # prepare validation train dataloader
        val_X = []
        val_y = []
        X_date_stamp = []
        y_date_stamp = []

        measure_time_val = get_measure_time(participant_id)
        dt_val = datetime.strptime(measure_time_val, '%Y-%m-%d_%H-%M-%S')

        df_val = pd.read_csv(os.path.join(dataset_directory, val_file_name))
        X_data_val_all = df_val.loc[:, "f_1":last_column].to_numpy()
        y_data_val_all = df_val["hr"].to_numpy()
        for idx in range(X_data_val_all.shape[0]):
            X_data_val = X_data_val_all[idx].reshape(self.seq_length, self.n_features)
            val_X.append(X_data_val)
            val_y.append([y_data_val_all[idx]])

            # X_date  = []
            for j in range(self.seq_length):
                dt_val += timedelta(seconds=1)
                # Convert the datetime object back into a string
                latest_datetime = dt_val.strftime('%Y-%m-%d %H:%M:%S')
                X_date_stamp.append(latest_datetime)

                dt_val_d = dt_val
                dt_val_d += timedelta(days=1)
                # Convert the datetime object back into a string
                latest_datetime_d = dt_val_d.strftime('%Y-%m-%d %H:%M:%S')
                y_date_stamp.append(latest_datetime_d)

        val_X_date_stamp = self._refactor_date_stamp(X_date_stamp)
        val_y_date_stamp = self._refactor_date_stamp(y_date_stamp)

        val_X = torch.tensor(val_X).float()
        val_y = torch.tensor(val_y).float()
        val_X_date_stamp = torch.tensor(val_X_date_stamp).float()
        val_y_date_stamp = torch.tensor(val_y_date_stamp).float()

        val_dataset = RadarDataset(val_X, val_y, X_date_stamp=val_X_date_stamp, y_date_stamp=val_y_date_stamp)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=self.is_shuffle)

        return train_dataloader, val_dataloader

    def _refactor_date_stamp(self, data_stamp):
        train_date_stamp = []
        data_stamp = time_features(pd.to_datetime(data_stamp), freq='s')
        data_stamp = data_stamp.transpose(1, 0)
        for i in range(int(len(data_stamp) / self.seq_length)):
            train_date_stamp.append(data_stamp[i * self.seq_length: self.seq_length * (i + 1)])
        return train_date_stamp

    def get_cross_with_date_dataloaders(self, participant_id):
        '''
            Prepare dataloader for cross validation using the raw data with date (2000 features).
            :param participant_id:
            :return:
        '''
        train_X = []
        train_y = []
        val_X = []
        val_y = []

        parent_dir = str(Path.cwd().parent)
        dataset_directory = parent_dir + "/publicdata/dataset/cross_train_date/"
        val_file_name_X = "X_raw_" + str(participant_id) + "_Resting.csv"
        val_file_name_y = "y_raw_" + str(participant_id) + "_Resting.csv"
        df_dataset = None

        # prepare train dataloader
        file_names_X = [file_name for file_name in os.listdir(dataset_directory) if
                      file_name.startswith("X_raw_") and file_name != val_file_name_X]
        file_names_y = [file_name for file_name in os.listdir(dataset_directory) if
                        file_name.startswith("y_raw_") and file_name != val_file_name_y]
        file_names_X = sorted(file_names_X, key=lambda x: int(re.findall(r'\d+', x)[0]))
        file_names_y = sorted(file_names_y, key=lambda x: int(re.findall(r'\d+', x)[0]))

        for i, (file_X, file_y) in enumerate(zip(file_names_X, file_names_y)):
            if i>1:
                break
            file_path_X = os.path.join(dataset_directory, file_X)
            file_path_y = os.path.join(dataset_directory, file_y)
            # print(f'training file_name-{i} = {file_path}')
            df_X = pd.read_csv(file_path_X)
            df_y = pd.read_csv(file_path_y)
            for id in range(df_y.shape[0]):
                filter_X = (df_X["id"] == id)
                filter_y = (df_y["id"] == id)
                if self.is_date is False:
                    X_data = df_X.loc[filter_X, "fs_1":"fs_2000"].to_numpy().tolist()
                else:
                    X_data = df_X.loc[filter_X, "date":"fs_2000"].to_numpy().tolist()
                y_data = df_y.loc[filter_y, "hr"].to_numpy().tolist()
                train_X.append(X_data)
                train_y.append(y_data)

            print(f'loaded training file_name-{i} = {file_X}')

        train_X = torch.tensor(train_X).float()
        train_y = torch.tensor(train_y).float()
        if self.is_date is True:
            train_y = train_y.reshape(train_y.size(0), 1, 1)
        train_dataset = RadarDataset(train_X, train_y)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.is_shuffle)

        # prepare validation train dataloader
        df_val_X = pd.read_csv(os.path.join(dataset_directory, val_file_name_X))
        df_val_y = pd.read_csv(os.path.join(dataset_directory, val_file_name_y))

        for id in range(df_val_y.shape[0]):
            filter_val_X = (df_val_X["id"] == id)
            filter_val_y = (df_val_y["id"] == id)

            if self.is_date is False:
                X_val_data = df_val_X.loc[filter_val_X, "fs_1":"fs_2000"].to_numpy().tolist()
            else:
                X_val_data = df_val_X.loc[filter_val_X, "date":"fs_2000"].to_numpy().tolist()
            y_val_data = df_val_y.loc[filter_val_y, "hr"].to_numpy().tolist()
            val_X.append(X_val_data)
            val_y.append(y_val_data)

        val_X = torch.tensor(val_X).float()
        val_y = torch.tensor(val_y).float()
        if self.is_date is True:
            val_y = val_y.reshape(val_y.size(0), 1, 1)
        val_dataset = RadarDataset(val_X, val_y)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=self.is_shuffle)

        return train_dataloader, val_dataloader


# PrepareTrainData().get_cross_with_date_dataloaders(1)
# PrepareTrainData().get_cross_dataloaders_add_date(1)
