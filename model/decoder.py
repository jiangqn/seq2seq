import torch
import torch.nn as nn
import torch.nn.functional as F
from model.attention import Attention
from model.utils import sequence_mean

class Decoder(nn.Module):

    def __init__(self, embedding, lstm_cell, attention, hidden_size):
        super(Decoder, self).__init__()
        self._embedding = embedding
        self._lstm_cell = lstm_cell
        self._query_projection = nn.Linear(hidden_size, hidden_size)
        self._attn = attention
        self._output_projection = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, embedding.embedding_dim)
        )

    def forward(self, src_memory, src_mask, init_decoder_states, init_decoder_output, trg):
        max_len = trg.size(1)
        decoder_states = init_decoder_states
        decoder_output = init_decoder_output
        logits = []
        for i in range(max_len):
            token = trg[:, i: i + 1]    # token: Tensor (batch_size, 1)
            logit, decoder_states, decoder_output = self._step(src_memory, src_mask, token, decoder_states, decoder_output)
            logits.append(logit)
        logits = torch.stack(logits, dim=1)
        return logits

    def _step(self, src_memory, src_mask, token, prev_decoder_states, prev_decoder_output):
        token_embedding = self._embedding(token).squeeze(1)
        lstm_input = torch.cat([token_embedding, prev_decoder_output], dim=1)
        decoder_states = self._lstm_cell(lstm_input, prev_decoder_states)
        decoder_hidden, _ = decoder_states
        top_hidden = decoder_hidden[-1]
        query = self._query_projection(top_hidden)
        context = self._attn(query, src_memory, src_memory, src_mask)
        decoder_output = self._output_projection(torch.cat([top_hidden, context], dim=1))
        logit = torch.mm(decoder_output, self._embedding.weight.t())
        return logit, decoder_states, decoder_output

    def decode_step(self, src_memory, src_mask, token, prev_decoder_states, prev_decoder_output):
        logit, decoder_states, decoder_output = self._step(src_memory, src_mask, token, prev_decoder_states, prev_decoder_output)
        token_output = torch.max(logit, dim=1, keepdim=True)[1]
        return token_output, decoder_states, decoder_output

    def get_init_decoder_output(self, src_memory, src_lens, init_decoder_states):
        init_decoder_hidden, _ = init_decoder_states
        init_top_hidden = init_decoder_hidden[-1]
        src_mean = sequence_mean(src_memory, src_lens, dim=1)
        init_decoder_output = self._output_projection(torch.cat([init_top_hidden, src_mean], dim=1))
        return init_decoder_output

class MultiLayerLSTMCells(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout, bias=True):
        super(MultiLayerLSTMCells, self).__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._bias = bias
        self._dropout = dropout
        self._lstm_cells = nn.ModuleList([nn.LSTMCell(self._input_size, self._hidden_size, self._bias)])
        self._lstm_cells.extend(nn.ModuleList(
            nn.LSTMCell(self._hidden_size, self._hidden_size, self._bias)
            for _ in range(self._num_layers - 1)
        ))

    def forward(self, input, states):
        # input: Tensor (batch_size, input_size)
        # states: (hidden, cell)
        # hidden: Tensor (num_layers, batch_size, hidden_size)
        # cell: Tensor (num_layers, batch_size, hidden_size)
        hidden, cell = states
        output_hidden = []
        output_cell = []
        for i, lstm_cell in enumerate(self._lstm_cells):
            h, c = lstm_cell(input, (hidden[i], cell[i]))
            output_hidden.append(h)
            output_cell.append(c)
            input = F.dropout(h, p=self._dropout, training=self.training)
        output_hidden = torch.stack(output_hidden, dim=0)
        output_cell = torch.stack(output_cell, dim=0)
        return output_hidden, output_cell

    @property
    def input_size(self):
        return self._input_size

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def num_layers(self):
        return self._num_layers

    @property
    def bias(self):
        return self._bias