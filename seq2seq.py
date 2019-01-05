import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from encoder import Encoder

class Seq2Seq(nn.Module):

    def __init__(self, embedding, hidden_size, num_layers, bidirectional, dropout=0.0):
        super(Seq2Seq, self).__init__()
        # [embedding setting]
        self._embedding = embedding
        self._embed_size = embedding.embed_size

        # [encoder setting]
        self._encoder = Encoder(
            embed_size=self._embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout
        )
        encoder_output_size = hidden_size * (2 if bidirectional else 1)


    def forward(self, src, src_lens, trg):
        src = self._embedding(src)
        encoder_output, (final_encoder_hidden, final_encoder_cell) = self._encoder(src, src_lens)
