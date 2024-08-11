import torch
import torch.nn as nn


class RadarGRU(nn.Module):
    def __init__(self, n_features=118, n_hidden=1024, n_layers=3, dropout=0.01):
        super(RadarGRU, self).__init__()

        self.lr = 0.001
        self.loss_fun = nn.MSELoss()

        self.n_hidden = n_hidden  # 隐层大小
        self.n_layers = n_layers  # gru层数
        self.dropout = dropout

        self.gru = nn.GRU(n_features, n_hidden, n_layers, batch_first=True, dropout=self.dropout)
        self.linear = nn.Linear(n_hidden, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.n_layers, batch_size, self.n_hidden)

        if torch.cuda.is_available():
            h0 = h0.cuda()

        output, h0 = self.gru(x, h0)

        y_preds = self.linear(output[:, -1, :])

        return y_preds
