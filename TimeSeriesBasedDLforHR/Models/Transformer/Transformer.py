import torch
import torch.nn as nn
from Models.layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from Models.layers.SelfAttention_Family import FullAttention, AttentionLayer
from Models.layers.Embed import DataEmbedding


class Transformer(nn.Module):

    def __init__(self, pred_len=1, enc_in=118, dropout=0.05):
        super(Transformer, self).__init__()

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
        self.features = 'MS'   # M : multivariate predict multivariate, S : univariate predict univariate, MS : multivariate predict univariate


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
                        FullAttention(False, self.factor, attention_dropout=self.dropout,
                                      output_attention=self.output_attention), self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                ) for l in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, self.factor, attention_dropout=self.dropout, output_attention=False),
                        self.d_model, self.n_heads),
                    AttentionLayer(
                        FullAttention(False, self.factor, attention_dropout=self.dropout, output_attention=False),
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
            outputs = dec_out[:, -self.pred_len:, :]  # [B, L, D]
            return outputs
