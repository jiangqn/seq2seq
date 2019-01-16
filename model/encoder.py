import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from model.utils import INIT

class Encoder(nn.Module):

    def __init__(self, rnn_type, embed_size, hidden_size, num_layers, bidirectional, dropout):
        super(Encoder, self).__init__()
        state_layers = num_layers * (2 if bidirectional else 1)
        self._rnn_type = rnn_type
        if rnn_type == 'LSTM':
            self._rnn = nn.LSTM(
                input_size=embed_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=dropout,
                batch_first=True
            )
            self._init_states = nn.ParameterList([
                nn.Parameter(
                    torch.Tensor(state_layers, hidden_size)
                ),
                nn.Parameter(
                    torch.Tensor(state_layers, hidden_size)
                )
            ])
            # init.uniform_(self._init_states[0], -INIT, INIT)
            # init.uniform_(self._init_states[1], -INIT, INIT)
            init.xavier_uniform_(self._init_states[0])
            init.xavier_uniform_(self._init_states[1])
        elif rnn_type == 'GRU':
            self._rnn = nn.GRU(
                input_size=embed_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=dropout,
                batch_first=True
            )
            self._init_states = nn.Parameter(
                torch.Tensor(state_layers, hidden_size)
            )
            # init.uniform_(self._init_states, -INIT, INIT)
            init.xavier_uniform_(self._init_states)
        else:
            raise ValueError('No Supporting.')

    def forward(self, src_embedding, src_lens):
        # src_embedding: Tensor (batch_size, time_step, embed_size)
        # src_lens: list (batch_size,)
        batch_size = src_embedding.size(0)
        assert batch_size == len(src_lens)
        init_states = self._get_init_states(batch_size)
        packed_src, sort_index = self._pack_padded_sequence(src_embedding, src_lens)
        packed_output, final_states = self._rnn(packed_src, init_states)
        output, final_states = self._pad_packed_sequence(packed_output, final_states, sort_index)
        return output, final_states

    def _get_init_states(self, batch_size):
        if self._rnn_type == 'LSTM':    # LSTM
            state_layers, hidden_size = self._init_states[0].size()
            size = (state_layers, batch_size, hidden_size)
            init_states = (
                self._init_states[0].unsqueeze(1).expand(*size).contiguous(),
                self._init_states[1].unsqueeze(1).expand(*size).contiguous()
            )
        else:   # GRU
            state_layers, hidden_size = self._init_states.size()
            size = (state_layers, batch_size, hidden_size)
            init_states = self._init_states.cuda().unsqueeze(1).expand(*size).contiguous()
        return init_states

    def _pack_padded_sequence(self, src_embedding, src_lens):
        sort_index = sorted(range(len(src_lens)), key=lambda i: src_lens[i], reverse=True)
        src_lens = [src_lens[i] for i in sort_index]
        src_embedding = self._reorder_sequence(src_embedding, sort_index)
        packed_src = pack_padded_sequence(src_embedding, src_lens, batch_first=True)
        return packed_src, sort_index

    def _pad_packed_sequence(self, packed_output, final_states, sort_index):
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        back_map = {index: i for i, index in enumerate(sort_index)}
        reorder_index = [back_map[i] for i in range(len(sort_index))]
        output = self._reorder_sequence(output, reorder_index)
        reorder_index = torch.LongTensor(reorder_index).cuda()
        if self._rnn_type == 'LSTM': # LSTM
            final_states = (
                final_states[0].index_select(index=reorder_index, dim=1),
                final_states[1].index_select(index=reorder_index, dim=1)
            )
        else:   # GRU
            final_states = final_states.index_select(index=reorder_index, dim=1)
        return output, final_states

    def _reorder_sequence(self, sequence, order):
        assert sequence.size(0) == len(order)
        order = torch.LongTensor(order).cuda()
        return sequence.index_select(index=order, dim=0)