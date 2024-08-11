import torch
from torch import nn


class HARHN(nn.Module):
    def __init__(self, n_conv_layers=3, seq_length=5, n_features=118, target_feats=1, n_units_enc=32, n_units_dec=32, enc_input_size=32,
                 rec_depth=3,
                 out_feats=1, n_filters=32, filter_size=5):
        super(HARHN, self).__init__()

        self.lr = 0.001
        self.loss_fun = nn.MSELoss()

        assert n_conv_layers > 0
        self.n_convs = n_conv_layers
        self.n_units_enc = n_units_enc
        self.n_units_dec = n_units_dec
        self.rec_depth = rec_depth
        self.seq_length = seq_length
        self.convs = nn.ModuleList([ConvBlock(seq_length, n_features, n_filters=n_filters,
                                              filter_size=filter_size) if i == 0 else ConvBlock(seq_length, n_filters,
                                                                                                n_filters=n_filters,
                                                                                                filter_size=filter_size)
                                    for i in range(n_conv_layers)])
        self.conv_to_enc = nn.Linear(n_filters, enc_input_size)
        self.RHNEncoder = RHN(enc_input_size, out_feats=n_units_enc, n_units=n_units_enc, rec_depth=rec_depth)
        self.RHNDecoder = RHNCell(target_feats, n_units_dec, rec_depth=rec_depth)
        self.T_k = nn.ModuleList([nn.Linear(n_units_dec, n_units_enc, bias=False) for i in range(self.rec_depth)])
        self.U_k = nn.ModuleList([nn.Linear(n_units_enc, n_units_enc) for i in range(self.rec_depth)])
        self.v_k = nn.ModuleList([nn.Linear(n_units_enc, 1) for i in range(self.rec_depth)])
        self.W_tilda = nn.Linear(target_feats, target_feats, bias=False)
        self.V_tilda = nn.Linear(rec_depth * n_units_enc, target_feats)
        self.W = nn.Linear(n_units_dec, target_feats)
        self.V = nn.Linear(rec_depth * n_units_enc, target_feats)

    def forward(self, x):
        for l in range(self.n_convs):
            x = self.convs[l](x)
        x = self.conv_to_enc(x)
        x, h_T_L = self.RHNEncoder(x)  # h_T_L.shape = (batch_size, T, n_units_enc, rec_depth)
        s = torch.zeros(x.shape[0], self.n_units_dec)
        if torch.cuda.is_available():
            s = s.cuda()
        for t in range(self.seq_length):
            s_rep = s.unsqueeze(1)
            s_rep = s_rep.repeat(1, self.seq_length, 1)
            d_t = []
            for k in range(self.rec_depth):
                h_T_k = h_T_L[..., k]
                a = self.U_k[k](h_T_k)
                b = self.T_k[k](s_rep)
                e_t_k = self.v_k[k](torch.tanh(self.T_k[k](s_rep) + self.U_k[k](h_T_k)))
                alpha_t_k = torch.softmax(e_t_k, 1)
                d_t_k = torch.sum(h_T_k * alpha_t_k, dim=1)
                d_t.append(d_t_k)
            d_t = torch.cat(d_t, dim=1)
            # y_tilda_t = self.W_tilda(y[:, t, :]) + self.V_tilda(d_t)
            y_tilda_t = self.V_tilda(d_t)
            s, _ = self.RHNDecoder(y_tilda_t, s)

        y_T = self.W(s) + self.V(d_t)
        return y_T


class HSGLayer(nn.Module):
    def __init__(self, n_units, init_gates_closed):
        super(HSGLayer, self).__init__()
        self.W_R = nn.Linear(n_units, n_units, bias=False)
        self.W_F = nn.Linear(n_units, n_units)
        if init_gates_closed:
            if torch.cuda.is_available():
                self.W_F.bias = nn.Parameter(torch.Tensor([-2.5] * n_units).cuda())
            else:
                self.W_F.bias = nn.Parameter(torch.Tensor([-2.5] * n_units))

    def forward(self, s_L_t, s_prime_tm1):
        g = torch.sigmoid(self.W_R(s_prime_tm1) + self.W_F(s_L_t))
        s_prime_t = g * s_prime_tm1 + (1 - g) * s_L_t
        return s_prime_t


class RHNCell(nn.Module):
    def __init__(self, in_feats, n_units, rec_depth=3, couple_gates=True,
                 use_HSG=False, init_gates_closed=False):
        super(RHNCell, self).__init__()
        self.rec_depth = rec_depth
        self.in_feats = in_feats
        self.n_units = n_units
        self.couple_gates = couple_gates
        self.use_HSG = use_HSG
        self.W_H = nn.Linear(in_feats, n_units, bias=False)
        self.W_T = nn.Linear(in_feats, n_units, bias=False)
        if not couple_gates:
            self.W_C = nn.Linear(in_feats, n_units, bias=False)
        self.R_H = nn.ModuleList([nn.Linear(n_units, n_units) for _ in range(rec_depth)])
        self.R_T = nn.ModuleList([nn.Linear(n_units, n_units) for _ in range(rec_depth)])
        if not couple_gates:
            self.R_C = nn.ModuleList([nn.Linear(n_units, n_units) for _ in range(rec_depth)])

        if use_HSG:
            self.HSG = HSGLayer(n_units, init_gates_closed)

        if init_gates_closed:
            for l in range(rec_depth):
                if torch.cuda.is_available():
                    self.R_T[l].bias = nn.Parameter(torch.Tensor([-2.5] * n_units).cuda())
                else:
                    self.R_T[l].bias = nn.Parameter(torch.Tensor([-2.5] * n_units))
                if not couple_gates:
                    if torch.cuda.is_available():
                        self.R_C[l].bias = nn.Parameter(torch.Tensor([-2.5] * n_units).cuda())
                    else:
                        self.R_C[l].bias = nn.Parameter(torch.Tensor([-2.5] * n_units))

    def forward(self, x, s):
        if self.use_HSG:
            s_prime_tm1 = s
        preds = []
        for l in range(self.rec_depth):
            if l == 0:
                h_l_t = torch.tanh(self.W_H(x) + self.R_H[l](s))
                t_l_t = torch.sigmoid(self.W_T(x) + self.R_T[l](s))
                if not self.couple_gates:
                    c_l_t = torch.sigmoid(self.W_C(x) + self.R_C[l](s))
            else:
                h_l_t = torch.tanh(self.R_H[l](s))
                t_l_t = torch.sigmoid(self.R_T[l](s))
                if not self.couple_gates:
                    c_l_t = torch.sigmoid(self.R_C[l](s))

            if not self.couple_gates:
                s = h_l_t * t_l_t + c_l_t * s
            else:
                s = h_l_t * t_l_t + (1 - t_l_t) * s
            preds.append(s)

        if self.use_HSG:
            s = self.HSG(s, s_prime_tm1)
            preds.pop()
            preds.append(s)
        preds = torch.stack(preds)
        return s, preds


class RHN(nn.Module):
    def __init__(self, in_feats, out_feats, n_units=32, rec_depth=3, couple_gates=True, use_HSG=False,
                 init_gates_closed=False, use_batch_norm=False):
        super(RHN, self).__init__()
        assert rec_depth > 0
        self.rec_depth = rec_depth
        self.in_feats = in_feats
        self.n_units = n_units
        self.init_gates_closed = init_gates_closed
        self.couple_gates = couple_gates
        self.use_HSG = use_HSG
        self.use_batch_norm = use_batch_norm
        self.RHNCell = RHNCell(in_feats, n_units, rec_depth, couple_gates=couple_gates,
                               use_HSG=use_HSG, init_gates_closed=init_gates_closed)
        if use_batch_norm:
            self.bn_x = nn.BatchNorm1d(in_feats)
            self.bn_s = nn.BatchNorm1d(n_units)

    def forward(self, x):
        s = torch.zeros(x.shape[0], self.n_units)
        if torch.cuda.is_available():
            s = s.cuda()
        preds = []
        highway_states = []
        for t in range(x.shape[1]):
            if self.use_batch_norm:
                x_inp = self.bn_x(x[:, t, :])
                s = self.bn_s(s)
            else:
                x_inp = x[:, t, :]
            s, all_s = self.RHNCell(x_inp, s)
            preds.append(s)
            highway_states.append(all_s)
        preds = torch.stack(preds)
        preds = preds.permute(1, 0, 2)
        highway_states = torch.stack(highway_states)
        highway_states = highway_states.permute(2, 0, 3, 1)
        out = preds

        return out, highway_states


class ConvBlock(nn.Module):
    def __init__(self, T, in_channels, n_filters=32, filter_size=5):
        super(ConvBlock, self).__init__()
        padding1 = self._calc_padding(T, filter_size)
        self.conv = nn.Conv1d(in_channels, n_filters, filter_size, padding=padding1)
        self.relu = nn.ReLU()
        self.maxpool = nn.AdaptiveMaxPool1d(T)
        self.zp = nn.ConstantPad1d((1, 0), 0)

    def _calc_padding(self, Lin, kernel, stride=1, dilation=1):
        p = int(((Lin - 1) * stride + 1 + dilation * (kernel - 1) - Lin) / 2)
        return p

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.permute(0, 2, 1)
        return x
    # HARHN(3, X_train.shape[1], X_train.shape[2], 1, n_units_enc=64, n_units_dec=64).cuda()


