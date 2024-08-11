import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.masking import TriangularCausalMask, ProbMask
from Models.layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from Models.layers.SelfAttention_Family import FullAttention, ProbAttention, AttentionLayer
from Models.layers.Embed import DataEmbedding
import numpy as np


class Informer(nn.Module):

    def __init__(self, pred_len=1, enc_in=118, dropout=0.01):
        super(Informer, self).__init__()

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

        # Embedding
        self.enc_embedding = DataEmbedding(self.enc_in, self.d_model, self.embed, self.freq,
                                           self.dropout)
        self.dec_embedding = DataEmbedding(self.dec_in, self.d_model, self.embed, self.freq,
                                           self.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, self.factor, attention_dropout=self.dropout,
                                      output_attention=self.output_attention),
                        self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                ) for l in range(self.e_layers)
            ],
            [
                ConvLayer(
                    self.d_model
                ) for l in range(self.e_layers - 1)
            ] if self.distil else None,
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, self.factor, attention_dropout=self.dropout, output_attention=False),
                        self.d_model, self.n_heads),
                    AttentionLayer(
                        ProbAttention(False, self.factor, attention_dropout=self.dropout, output_attention=False),
                        self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation,
                )
                for l in range(self.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model),
            projection=nn.Linear(self.d_model, self.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
