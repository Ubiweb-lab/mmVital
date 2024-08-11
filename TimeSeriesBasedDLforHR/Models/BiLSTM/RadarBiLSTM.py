import torch
import torch.nn as nn


class RadarBiLSTM(nn.Module):
    def __init__(self, n_features=118, n_hidden=1024, n_layers=3, dropout=0.01):
        super().__init__()

        self.lr = 0.001
        self.loss_fun = nn.MSELoss()

        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.num_directions = 2
        self.dropout = dropout

        self.lstm = nn.LSTM(self.n_features, self.n_hidden, self.n_layers, batch_first=True, bidirectional=True, dropout=self.dropout)
        self.linear = nn.Linear(self.n_hidden, 1)

    def forward(self, x):
        batch_size = x.size(0)
        sequence_length = x.size(1)

        h0 = torch.zeros(self.num_directions * self.n_layers, batch_size, self.n_hidden)
        c0 = torch.zeros(self.num_directions * self.n_layers, batch_size, self.n_hidden)

        if torch.cuda.is_available():
            h0 = h0.cuda()
            c0 = c0.cuda()

        bi_lstm_input = x.view(batch_size, sequence_length, self.n_features)

        lstm_out, (h0_update, c0_update) = self.lstm(bi_lstm_input, (h0, c0))
        lstm_out = lstm_out.contiguous().view(batch_size, sequence_length, self.num_directions, self.n_hidden)
        lstm_out = torch.mean(lstm_out, dim=2)

        y_preds = self.linear(lstm_out[:, -1, :])

        return y_preds