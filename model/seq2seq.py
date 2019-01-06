import torch
import torch.nn as nn
from utils import len_mask

class Seq2Seq(nn.Module):

    def __init__(self, embedding, encoder, bridge, decoder):
        super(Seq2Seq, self).__init__()
        self._embedding = embedding
        self._encoder = encoder
        self._bridge = bridge
        self._decoder = decoder

    def forward(self, src, src_lens, trg):
        # src: Tensor (batch_size, src_time_step)
        # src_lens: list (batch_size,)
        # trg: Tensor (batch_size, trg_time_step)
        src_embedding = self._embedding(src)
        src_memory, init_decoder_states = self._bridge(self._encoder(src_embedding, src_lens))
        src_mask = len_mask(src_lens, src.size(1))
        init_decoder_output = self._decoder.get_init_decoder_output(src_memory, src_lens, init_decoder_states)
        return self._decoder(src_memory, src_mask, init_decoder_states, init_decoder_output, trg)