import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from Models.layers.SelfAttention_Family import ReformerLayer
from Models.layers.Embed import DataEmbedding
import numpy as np


class Reformer(nn.Module):


    def __init__(self, pred_len=1, enc_in=118, dropout=0.01):
        super(Reformer, self).__init__()

        self.lr = 0.001
        self.loss_fun = nn.MSELoss()

        self.label_len = 1
        self.pred_len = pred_len
        self.output_attention = False
        self.d_model = 512
        self.embed = 'timeF'
        self.freq = 's'
        self.enc_in = enc_in
        self.dec_in = 1
        self.dropout = dropout
        self.factor = 3
        self.n_heads = 8
        self.d_ff = 2048
        self.activation = 'gelu'
        self.e_layers = 2
        self.d_layers = 1
        self.c_out = 118
        self.features = 'MS'  # M : multivariate predict multivariate, S : univariate predict univariate, MS : multivariate predict univariate
        self.distil = True
        self.bucket_size = 4
        self.n_hashes = 4

        # Embedding
        self.enc_embedding = DataEmbedding(self.enc_in, self.d_model, self.embed, self.freq,
                                           self.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    ReformerLayer(None, self.d_model, self.n_heads, bucket_size=self.bucket_size,
                                  n_hashes=self.n_hashes),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                ) for l in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )
        self.projection = nn.Linear(self.d_model, self.c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # add placeholder
        # x_enc = torch.cat([x_enc, x_dec[:, -self.pred_len:, :]], dim=1)
        x_enc = torch.cat([x_enc, torch.zeros_like(x_enc[:, -self.pred_len:, :]).float()], dim=1)
        x_mark_enc = torch.cat([x_mark_enc, x_mark_dec[:, -self.pred_len:, :]], dim=1)
        # Reformer: encoder only
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        enc_out = self.projection(enc_out)

        if self.output_attention:
            return enc_out[:, -self.pred_len:, :], attns
        else:
            return enc_out[:, -self.pred_len:, :]  # [B, L, D]
