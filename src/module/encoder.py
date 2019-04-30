import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from src.module.utils.sentence_clip import sentence_clip
from src.module.utils.constants import PAD_INDEX

class Encoder(nn.Module):

    def __init__(self, embedding, rnn_type, hidden_size, num_layers, bidirectional, dropout):
        super(Encoder, self).__init__()
        self.embedding = embedding
        self.embed_size = embedding.embedding_dim
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=self.embed_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=dropout,
                batch_first=True
            )
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=self.embed_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=dropout,
                batch_first=True
            )
        else:
            raise ValueError('No Supporting.')
        encoder_output_size = hidden_size * (2 if bidirectional else 1)
        self.output_projection = nn.Linear(encoder_output_size, hidden_size)
        if rnn_type == 'LSTM':
            self.hidden_projection = nn.Linear(encoder_output_size, hidden_size)
            self.cell_projection = nn.Linear(encoder_output_size, hidden_size)
        else:   # GRU
            self.hidden_projection = nn.Linear(encoder_output_size, hidden_size)

    def forward(self, src):
        """
        :param src: LongTensor (batch_size, time_step)
        :param src_lens: LongTensor (batch_size,)
        :return output: FloatTensor (batch_size, time_step, hidden_size)
        :return final_states: hidden or (hidden, cell)
            hidden: FloatTensor (num_layers, batch_size, hidden_size)
            cell: FloatTensor (num_layer, batch_size, hidden_size)
        """
        src = sentence_clip(src)
        src_mask = (src != PAD_INDEX)
        src_lens = src_mask.long().sum(dim=1, keepdim=False)
        src = self.embedding(src)
        src = F.dropout(src, p=self.dropout, training=self.training)
        src_lens, sort_index = src_lens.sort(descending=True)
        src = src.index_select(dim=0, index=sort_index)
        packed_src = pack_padded_sequence(src, src_lens, batch_first=True)
        packed_output, final_states = self.rnn(packed_src)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        reorder_index = sort_index.argsort(descending=False)
        output = output.index_select(dim=0, index=reorder_index)
        if self.rnn_type == 'LSTM':
            final_states = (
                final_states[0].index_select(dim=1, index=reorder_index),
                final_states[1].index_select(dim=1, index=reorder_index)
            )
        else:   # GRU
            final_states = final_states.index_select(dim=1, index=reorder_index)
        output = self.output_projection(output)
        # raise ValueError('debug')
        if self.rnn_type == 'LSTM':
            if self.bidirectional:
                final_states = (
                    torch.cat(final_states[0].chunk(chunks=2, dim=0), dim=2),
                    torch.cat(final_states[1].chunk(chunks=2, dim=0), dim=2)
                )
            final_states = (
                torch.stack([
                    self.hidden_projection(hidden) for hidden in final_states[0]
                ], dim=0),
                torch.stack([
                    self.cell_projection(cell) for cell in final_states[1]
                ], dim=0)
            )
        else: # GRU
            if self.bidirectional:
                final_states = torch.cat(final_states.chunk(chunks=2, dim=0), dim=2)
            final_states = torch.stack([
                self.hidden_projection(hidden) for hidden in final_states
            ])
        return output, src_mask, final_states