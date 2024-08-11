import os
from pathlib import Path

import numpy as np
import pandas as pd

import sys



sys.path.append('/home/syt0722/Weichun/60pts')
from Models.ExecuteTrainModels import ExecuteTrainModels
from Models.ModelNames import ModelNames
from Models.LSTM.RadarLSTM import RadarLSTM
from Models.BiLSTM.RadarBiLSTM import RadarBiLSTM
from Models.GRU.RadarGRU import RadarGRU
from Models.TPALSTM.RadarTpaLSTM import RadarTpaLSTM
from Models.NBEATS.NBeats import NBeats
from Models.RNNED.NetGRU import NetGRU
from Models.Informer.Informer import Informer
from Models.Transformer.Transformer import Transformer
from Models.Reformer.Reformer import Reformer
from Models.MTGNN.MTGNN import MTGNN

def get_model(model_name):
    model = None
    if model_name == ModelNames.LSTM.value:
        model = RadarLSTM()
    elif model_name == ModelNames.BiLSTM.value:
        model = RadarBiLSTM()
    elif model_name == ModelNames.GRU.value:
        model = RadarGRU()
    elif model_name == ModelNames.DILATE.value:
        model = NetGRU()
    elif model_name == ModelNames.TPALSTM.value:
        model = RadarTpaLSTM()
    elif model_name == ModelNames.NBEATS.value:
        model = NBeats()
    # elif model_name == ModelNames.Trans.value:
    #     model = Informer(118, 118, 118, 5, 1, 1)
    elif model_name == ModelNames.Transformer.value:
        model = Transformer()
    elif model_name == ModelNames.Informer.value:
        model = Informer()
    elif model_name == ModelNames.Reformer.value:
        model = Reformer()
    elif model_name == ModelNames.MTGNN.value:
        model = MTGNN()

    return model


def start_training(model_name, epochs, participant=-1):
    pd_columns = ["t_loss", "v_loss", "p_id"]
    pd_values = None
    current_dir = str(Path.cwd())
    save_path = os.path.join(current_dir, model_name, "trained_models")

    # all participants ID
    if participant == -1:
        participant_ids = [i for i in range(1, 31) if i != 3]
        saved_file_path = os.path.join(save_path, "loss_" + model_name + ".csv")
    else:
        participant_ids = [participant]
        saved_file_path = os.path.join(save_path, "loss_" + model_name + "_" + str(participant) + ".csv")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for participant_id in participant_ids:
        ids = [participant_id for i in range(epochs)]
        my_model = get_model(model_name)
        et = ExecuteTrainModels(my_model, model_name, participant_id, epochs=epochs)
        t_loss, v_loss, _ = et.start_training()

        if pd_values is None:
            pd_values = np.column_stack((t_loss, v_loss, ids))
        else:
            pd_values = np.vstack([pd_values, np.column_stack((t_loss, v_loss, ids))])
        print(f'participant {participant_id} done.')
    df = pd.DataFrame(pd_values, columns=pd_columns)
    df.to_csv(saved_file_path)


epochs = 50

# model_name = ModelNames.GRU.value
# start_training(model_name, epochs)
#
# model_name = ModelNames.BiLSTM.value
# start_training(model_name, epochs)
#
# model_name = ModelNames.TPALSTM.value
# start_training(model_name, epochs)

# model_name = ModelNames.TSFM.value
# start_training(model_name, epochs)

model_name = ModelNames.MTGNN.value
start_training(model_name, epochs)

# model_name = ModelNames.RNNED.value
# start_training(model_name, epochs, participant=2)
# start_training(model_name, epochs)