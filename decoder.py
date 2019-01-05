import torch
import torch.nn as nn
from attention import Attention

class Decoder(nn.Module):

    def __init__(self, embedding, lstm_cell, hidden_size):
        super(Decoder, self).__init__()
        self._embedding = embedding
        self._lstm_cell = lstm_cell
        self._query_projection = nn.Linear(hidden_size, hidden_size)
        self._attn = Attention()
        self._output_projection = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, src_memory, src_mask, init_decoder_states, init_decoder_output, trg):
        max_len = trg.size(1)
        decoder_states = init_decoder_states
        decoder_output = init_decoder_output
        logits = []
        for i in range(max_len):
            token = trg[:, i: i + 1]
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

    def get_init_attn_out(self):
        pass