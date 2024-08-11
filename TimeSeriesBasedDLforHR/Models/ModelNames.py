from enum import Enum


class ModelNames(Enum):
    NBEATS = "NBEATS"
    LSTM = "LSTM"
    BiLSTM = "BiLSTM"
    GRU = "GRU"
    CnnLSTM = "CNNLSTM"
    DILATE = "RNNED"
    TPALSTM = "TPALSTM"
    Transformer = "Transformer"
    Informer = "Informer"
    Reformer = "Reformer"
    MTGNN = "MTGNN"


class PlotMarker(Enum):
    TPALSTM = "o"
    LSTM = "s"
    BiLSTM = "*"
    GRU = "h"
    RNNED = "d"
    CNNLSTM = "8"
    NBEATS = "p"
    Transformer = "+"
    Informer = "H"
    Reformer = "x"

# all_v = list(PlotMarker)
# s = ModelNames.NBEATS.value
# print(PlotMarker[s].value)