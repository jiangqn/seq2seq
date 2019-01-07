import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from model.utils import INIT, reorder_sequence, reorder_lstm_states

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
        batch_size = src_embedding.size(0)
        assert batch_size == len(src_lens)
        init_encoder_states = self._get_init_states(batch_size)
        packed_src, sort_index = self._pack_padded_sequence(src_embedding, src_lens)
        packed_output, final_encoder_states = self._lstm(packed_src, init_encoder_states)
        return self._pad_packed_sequence(packed_output, final_encoder_states, sort_index)

    def _get_init_states(self, batch_size):
        state_layers = self._init_encoder_hidden.size(0)
        hidden_size = self._init_encoder_hidden.size(1)
        size = (state_layers, batch_size, hidden_size)
        init_encoder_hidden = self._init_encoder_hidden.unsqueeze(1).expand(*size)
        init_encoder_cell = self._init_encoder_cell.unsqueeze(1).expand(*size)
        init_encoder_hidden = init_encoder_hidden.contiguous()
        init_encoder_cell = init_encoder_cell.contiguous()
        init_encoder_states = (init_encoder_hidden, init_encoder_cell)
        return init_encoder_states

    def _pack_padded_sequence(self, src_embedding, src_lens):
        sort_index = sorted(range(len(src_lens)), key=lambda i: src_lens[i], reverse=True)
        src_lens = [src_lens[i] for i in sort_index]
        src_embedding = reorder_sequence(src_embedding, sort_index)
        packed_src = pack_padded_sequence(src_embedding, src_lens, batch_first=True)
        return packed_src, sort_index

    def _pad_packed_sequence(self, packed_output, final_encoder_states, sort_index):
        encoder_output, _ = pad_packed_sequence(packed_output, batch_first=True)
        back_map = {index: i for i, index in enumerate(sort_index)}
        reorder_index = [back_map[i] for i in range(len(sort_index))]
        encoder_output = reorder_sequence(encoder_output, reorder_index)
        final_encoder_states = reorder_lstm_states(final_encoder_states, reorder_index)
        return encoder_output, final_encoder_states