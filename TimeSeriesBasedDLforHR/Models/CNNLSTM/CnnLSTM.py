from torch import nn
import torch

class CnnLSTM(nn.Module):

    def __init__(self, n_features=118, out_channels=54, kernel_size=1, n_hidden=1024, n_layers=3, dropout=0.01):
        super(CnnLSTM, self).__init__()

        self.lr = 0.001
        self.loss_fun = nn.MSELoss()

        self.n_layers = n_layers
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.dropout = dropout

        self.relu = nn.ReLU(inplace=True)

        self.conv = nn.Sequential(
            nn.Conv1d( in_channels=self.n_features, out_channels=out_channels, kernel_size=kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=kernel_size, stride=1)
        )
        self.lstm = nn.LSTM(input_size=out_channels, hidden_size=n_hidden,
                            num_layers=n_layers, batch_first=True, dropout=self.dropout)

        self.linear = nn.Linear(n_hidden, out_features=1)

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)

        conv_input = x.permute(0, 2, 1)
        conv_out = self.conv(conv_input)

        h0 = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        c0 = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        if torch.cuda.is_available():
            h0 = h0.cuda()
            c0 = c0.cuda()

        lstm_input = conv_out.permute(0, 2, 1)

        lstm_out, (h_out, c_out) = self.lstm(lstm_input, (h0, c0))

        y_preds = self.linear(lstm_out[:, -1, :])

        return y_preds