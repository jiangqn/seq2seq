import torch
import torch.nn as nn
from model.utils import len_mask
from model.utils import SOS_INDEX

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
        encoder_output, final_encoder_states = self._encoder(src_embedding, src_lens)
        src_memory, init_decoder_states = self._bridge(encoder_output, final_encoder_states)
        src_mask = len_mask(src_lens, src_memory.size(1))
        init_decoder_output = self._decoder.get_init_decoder_output(src_memory, src_lens, init_decoder_states)
        return self._decoder(src_memory, src_mask, init_decoder_states, init_decoder_output, trg)

    def decode(self, src, src_lens, max_len):
        # src: Tensor (batch_size, time_step)
        # src_lens: list (batch_size,)
        src_embedding = self._embedding(src)
        encoder_output, final_encoder_states = self._encoder(src_embedding, src_lens)
        src_memory, init_decoder_states = self._bridge(encoder_output, final_encoder_states)
        src_mask = len_mask(src_lens, src_memory.size(1))
        init_decoder_output = self._decoder.get_init_decoder_output(src_memory, src_lens, init_decoder_states)
        batch_size = src_memory.size(0)
        token = torch.tensor([SOS_INDEX] * batch_size).unsqueeze(1).cuda()
        decoder_states = init_decoder_states
        decoder_output = init_decoder_output
        outputs = []
        for _ in range(max_len):
            token, decoder_states, decoder_output = self._decoder.decode_step(src_memory, src_mask, token, decoder_states, decoder_output)
            outputs.append(token[:, 0])
        outputs = torch.stack(outputs, dim=1)
        return outputs