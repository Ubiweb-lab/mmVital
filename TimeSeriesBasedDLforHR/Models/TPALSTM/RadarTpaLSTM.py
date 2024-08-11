import torch
from torch import nn


class RadarTpaLSTM(nn.Module):

    def __init__(self, n_features=118, num_filters=3, n_hidden=1024, obs_len=5, n_layers=3, dropout=0.):
        super(RadarTpaLSTM, self).__init__()

        self.lr = 0.001
        self.loss_fun = nn.MSELoss()
        
        self.dropout = dropout
        self.hidden = nn.Linear(n_features, n_hidden)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(n_features, n_hidden, n_layers, bias=True,
                            batch_first=True, dropout=self.dropout)  # output (batch_size, obs_len, hidden_size)
        self.hidden_size = n_hidden
        self.filter_num = num_filters
        self.filter_size = 1  # Don't change this - otherwise CNN filters no longer 1D
        self.attention = TemporalPatternAttention(self.filter_size, self.filter_num, obs_len - 1, n_hidden)
        
        self.linear = nn.Linear(n_hidden, 1)
        self.n_layers = n_layers


    def forward(self, x):
        batch_size, obs_len, f_dim = x.size()

        H = torch.zeros(batch_size, obs_len - 1, self.hidden_size)
        ht = torch.zeros(self.n_layers, batch_size, self.hidden_size)

        if torch.cuda.is_available():
            H = H.cuda()
            ht = ht.cuda()

        ct = ht.clone()

        for t in range(obs_len):
            xt = x[:, t, :].view(batch_size, 1, -1)
            out, (ht, ct) = self.lstm(xt, (ht, ct))
            htt = ht.permute(1, 0, 2)
            htt = htt[:, -1, :]
            if t != obs_len - 1:
                H[:, t, :] = htt
        H = self.relu(H)

        # reshape hidden states H
        H = H.view(-1, 1, obs_len - 1, self.hidden_size)
        new_ht = self.attention(H, htt)
        ypred = self.linear(new_ht)

        return ypred



class TemporalPatternAttention(nn.Module):

    def __init__(self, filter_size, filter_num, attn_len, attn_size):
        super(TemporalPatternAttention, self).__init__()
        self.filter_size = filter_size
        self.filter_num = filter_num
        self.feat_size = attn_size - self.filter_size + 1
        self.conv = nn.Conv2d(1, filter_num, (attn_len, filter_size))
        self.linear1 = nn.Linear(attn_size, filter_num)
        self.linear2 = nn.Linear(attn_size + self.filter_num, attn_size)
        self.relu = nn.ReLU()

    def forward(self, H, ht):
        _, channels, _, attn_size = H.size()
        new_ht = ht.view(-1, 1, attn_size)
        w = self.linear1(new_ht)  # batch_size, 1, filter_num
        conv_vecs = self.conv(H)

        conv_vecs = conv_vecs.view(-1, self.feat_size, self.filter_num)
        conv_vecs = self.relu(conv_vecs)

        # score function
        w = w.expand(-1, self.feat_size, self.filter_num)
        s = torch.mul(conv_vecs, w).sum(dim=2)
        alpha = torch.sigmoid(s)
        new_alpha = alpha.view(-1, self.feat_size, 1).expand(-1, self.feat_size, self.filter_num)
        v = torch.mul(new_alpha, conv_vecs).sum(dim=1).view(-1, self.filter_num)

        concat = torch.cat([ht, v], dim=1)
        new_ht = self.linear2(concat)
        return new_ht