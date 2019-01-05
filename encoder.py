import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.rnn import pack_padded_sequence, pad_packed_sequence
from utils import INIT

class Encoder(nn.Module):

    def __init__(self, embed_size, hidden_size, num_layers, bidirectional, dropout):
        self._lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True
        )
        state_layers = num_layers * (2 if bidirectional else 1)
        self._init_encoder_hidden = nn.Parameter(
            torch.Tensor(state_layers, hidden_size)
        )
        self._init_encoder_cell = nn.Parameter(
            torch.Tensor(state_layers, hidden_size)
        )
        init.uniform_(self._init_encoder_hidden, -INIT, INIT)
        init.uniform_(self._init_encoder_cell, -INIT, INIT)

    def forward(self, src_embedding, src_lens):
        # src_embedding: Tensor (batch_size, time_step, embed_size)
        # src_lens: list (batch_size,)
        state_layers = self._init_encoder_hidden.size(0)
        batch_size = len(src_lens)
        hidden_size = self._init_encoder_hidden.size(1)
        size = (state_layers, batch_size, hidden_size)
        init_encoder_hidden = self._init_encoder_hidden.expand(*size)
        init_encoder_cell = self._init_encoder_cell.expand(*size)
        sort_index = sorted(range(len(src_lens)), key=lambda i: src_lens[i], reverse=True)
        init_encoder_hidden = init_encoder_hidden.contiguous()
        init_encoder_cell = init_encoder_cell.contiguous()
        packed_src = pack_padded_sequence(src_embedding, src_lens)
        packed_output, (final_encoder_hidden, final_encoder_cell) = self._lstm(packed_src, src_lens)
        encoder_output, _ = pad_packed_sequence(packed_output)