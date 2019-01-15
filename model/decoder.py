import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import sequence_mean, SOS_INDEX

class Decoder(nn.Module):

    def __init__(self, embedding, rnn_cell, attention, hidden_size):
        super(Decoder, self).__init__()
        self._embedding = embedding
        self._rnn_cell = rnn_cell
        self._query_projection = nn.Linear(hidden_size, hidden_size)
        self._attn = attention
        self._output_projection = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, embedding.embedding_dim)
        )

    def forward(self, src_memory, src_mask, init_states, init_output, trg, teacher_forcing_ratio=None):
        batch_size, max_len = trg.size()
        states = init_states
        output = init_output
        logits = []
        if teacher_forcing_ratio is not None:
            sample_probability = torch.FloatTensor([teacher_forcing_ratio] * batch_size).unsqueeze(-1).cuda()
            generated_token = torch.LongTensor([SOS_INDEX] * batch_size).unsqueeze(-1).cuda()
        for i in range(max_len):
            if teacher_forcing_ratio is not None:
                sample_distribution = torch.bernoulli(sample_probability).long()
                token = trg[:, i: i + 1] * sample_distribution + generated_token * (1 - sample_distribution)
            else:
                token = trg[:, i: i + 1]
            logit, states, output = self.step(src_memory, src_mask, token, states, output)
            logits.append(logit)
            if teacher_forcing_ratio is not None:
                generated_token = logit.max(dim=1, keepdim=True)[1]
        logits = torch.stack(logits, dim=1)
        return logits

    def step(self, src_memory, src_mask, token, prev_states, prev_output):
        # src_memory: Tensor (batch_size, time_step, hidden_size)
        # src_mask: Tensor (batch_size, time_step)
        # token: Tensor (batch_size, 1)
        # prev_states: tuple (hidden, cell)
        # hidden: (num_layers, batch_size, hidden_size)
        # cell: (num_layers, batch_size, hidden_size)
        # prev_output: (batch_size, embed_size)
        token_embedding = self._embedding(token).squeeze(1)
        rnn_input = torch.cat([token_embedding, prev_output], dim=1)
        states = self._rnn_cell(rnn_input, prev_states)
        if isinstance(states, tuple):   # LSTM
            top_hidden = states[0][-1]
        else:   # GRU
            top_hidden = states[-1]
        query = self._query_projection(top_hidden)
        context = self._attn(query, src_memory, src_memory, src_mask)
        output = self._output_projection(torch.cat([top_hidden, context], dim=1))
        logit = torch.mm(output, self._embedding.weight.t())
        return logit, states, output

    def get_init_output(self, src_memory, src_lens, init_states):
        if isinstance(init_states, tuple):  # LSTM
            init_top_hidden = init_states[0][-1]
        else:   # GRU
            init_top_hidden = init_states[-1]
        src_mean = sequence_mean(src_memory, src_lens, dim=1)
        init_output = self._output_projection(torch.cat([init_top_hidden, src_mean], dim=1))
        return init_output

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

class MultiLayerGRUCells(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, dropout, bias=True):
        super(MultiLayerGRUCells, self).__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._bias = bias
        self._dropout = dropout
        self._gru_cells = nn.ModuleList([nn.GRUCell(self._input_size, self._hidden_size, self._bias)])
        self._gru_cells.extend(nn.ModuleList(
            nn.GRUCell(self._hidden_size, self._hidden_size, self._bias)
            for _ in range(self._num_layers - 1)
        ))

    def forward(self, input, states):
        # input: Tensor (batch_size, input_size)
        # states: hidden
        # hidden: Tensor (num_layers, batch_size, hidden_size)
        hidden = states
        output_hidden = []
        for i, gru_cell in enumerate(self._gru_cells):
            h = gru_cell(input, hidden[i])
            output_hidden.append(h)
            input = F.dropout(h, p=self._dropout, training=self.training)
        output_hidden = torch.stack(output_hidden, dim=0)
        return output_hidden

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