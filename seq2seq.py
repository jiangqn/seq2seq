import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from encoder import Encoder
from bridge import Bridge
from decoder import Decoder
from utils import len_mask

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
        # [bridge setting]
        self._bridge = Bridge(
            hidden_size=hidden_size,
            bidirectional=bidirectional
        )
        # [decoder setting]
        self._decoder = Decoder()


    def forward(self, src, src_lens, trg):
        src_embedding = self._embedding(src)
        src_memory, (init_decoder_hidden, init_decoder_cell) = self._bridge(self._encoder(src_embedding, src_lens))
        src_mask = len_mask(src_lens)
        trg_embedding = self._embedding(trg)
        logit = self._decoder(src_memory, src_mask, (init_decoder_hidden, init_decoder_cell), trg_embedding)