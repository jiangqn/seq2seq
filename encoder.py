import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.rnn import pack_padded_sequence, pad_packed_sequence
from utils import INIT, reorder_sequence, reorder_lstm_states

class Encoder(nn.Module):

    def __init__(self, embed_size, hidden_size, num_layers, bidirectional, dropout):
        super(Encoder, self).__init__()
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
        assert src_embedding.size(0) == len(src_lens)
        # initialize encoder hidden states and cell states
        state_layers = self._init_encoder_hidden.size(0)
        batch_size = len(src_lens)
        hidden_size = self._init_encoder_hidden.size(1)
        size = (state_layers, batch_size, hidden_size)
        init_encoder_hidden = self._init_encoder_hidden.expand(*size)
        init_encoder_cell = self._init_encoder_cell.expand(*size)
        init_encoder_hidden = init_encoder_hidden.contiguous()
        init_encoder_cell = init_encoder_cell.contiguous()
        # sort the src_embedding and src_lens
        sort_index = sorted(range(batch_size), key=lambda i: src_lens[i], reverse=True)
        src_lens = [src_lens[i] for i in sort_index]
        src_embedding = reorder_sequence(src_embedding, sort_index)
        # pack the sorted src_embedding and src_lens
        packed_src = pack_padded_sequence(src_embedding, src_lens)
        packed_output, (final_encoder_hidden, final_encoder_cell) = self._lstm(packed_src, (init_encoder_hidden, init_encoder_cell))
        # unpack the packed encoder output
        encoder_output, _ = pad_packed_sequence(packed_output)
        # unsort the encoder output
        back_map = {index: i for i, index in enumerate(sort_index)}
        reorder_index = [back_map[i] for i in range(batch_size)]
        encoder_output = reorder_sequence(encoder_output, reorder_index)
        (final_encoder_hidden, final_encoder_cell) = reorder_lstm_states((final_encoder_hidden, final_encoder_cell), reorder_index)
        return encoder_output, (final_encoder_hidden, final_encoder_cell)