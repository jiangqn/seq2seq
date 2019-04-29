import torch
from torch import nn
import torch.nn.functional as F
from src.module.rnn_cell.multi_layer_lstm_cell import MultiLayerLSTMCell
from src.module.rnn_cell.multi_layer_gru_cell import MultiLayerGRUCell
from src.module.attention.bilinear_attention import BilinearAttention

class Decoder(nn.Module):

    def __init__(self, embedding, rnn_type, hidden_size, num_layers=1, dropout=0, weight_tying=True):
        super(Decoder, self).__init__()
        self.embedding = embedding
        self.embed_size = embedding.embedding_dim
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.dropout = dropout
        if rnn_type == 'LSTM':
            self.rnn_cell = MultiLayerLSTMCell(
                input_size=self.embed_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout
            )
        elif rnn_type == 'GRU':
            self.rnn_cell = MultiLayerGRUCell(
                input_size=self.embed_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout
            )
        else:
            raise ValueError('No Supporting.')
        self.attention_query_projection = nn.Linear(hidden_size, hidden_size)
        self.attention = BilinearAttention(hidden_size, hidden_size)
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, self.embed_size)
        )
        vocab_size = embedding.num_embeddings
        self.generator = nn.Linear(self.embed_size, vocab_size)
        if weight_tying:
            self.generator.weight = embedding.weight


    def forward(self, src, src_mask, init_states, trg):
        """
        :param src: FloatTensor (batch_size, src_time_step, hidden_size)
        :param src_mask: ByteTensor (batch_size, src_time_step)
        :param init_states: hidden or (hidden, cell)
            hidden: FloatTensor (batch_size, hidden_size)
            cell: FloatTensor (batch_size, hidden_size)
        :param trg: LongTensor (batch_size, trg_time_step)
        :return:
        """
        src_lens = src_mask.long().sum(dim=1, keepdim=False)
        states = init_states
        output = self._get_init_output(src, src_lens, init_states)
        logit = []
        max_len = trg.size(1)
        for i in range(max_len):
            token = trg[:, i]
            step_logit, states, output = self.step(src, src_mask, token, states, output)
            logit.append(step_logit)
        logit = torch.stack(logit, dim=1)
        return logit

    def _get_init_output(self, src, src_lens, init_states):
        """
        :param src: FloatTensor (batch_size, time_step, hidden_size)
        :param src_lens: LongTensor (batch_size,)
        :param init_states: hidden or (hidden, cell)
            hidden: FloatTensor (batch_size, hidden_size)
            cell: FloatTensor (batch_size, hidden_size)
        :return init_output: FloatTensor (batch_size, embed_size)
        """
        if self.rnn_type == 'LSTM':
            init_top_hidden = init_states[0][-1]
        else:   # GRU
            init_top_hidden = init_states[-1]
        src_mean = src.sum(dim=1, keepdim=False) / src_lens.float().unsqueeze(-1)
        init_output = self.output_projection(torch.cat((init_top_hidden, src_mean), dim=1))
        return init_output


    def step(self, src, src_mask, token, prev_states, prev_output):
        """
        :param src: FloatTensor (batch_size, time_step, hidden_size)
        :param src_mask: ByteTensor (batch_size, time_step)
        :param token: LongTensor (batch_size,)
        :param prev_states: hidden or (hidden, cell)
            hidden: FloatTensor (batch_size, hidden_size)
            cell: FloatTensor (batch_size, hidden_size)
        :param prev_output: FloatTensor (batch_size, embed_size)
        :return:
        """
        token = self.embedding(token)
        rnn_input = torch.cat([token, prev_output], dim=1)
        states = self.rnn_cell(rnn_input, prev_states)
        if self.rnn_type == 'LSTM':
            top_hidden = states[0][-1]
        else:   # GRU
            top_hidden = states[-1]
        attention_query = self.attention_query_projection(top_hidden)
        context = self.attention(attention_query, src, src, src_mask)
        output = self.output_projection(torch.cat((top_hidden, context), dim=1))
        logit = self.generator(output)
        return logit, states, output

    def decode(self):
        pass

    def beam_decode(self):
        pass