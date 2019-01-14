import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import len_mask
from model.utils import SOS_INDEX
from model.beam_search import Beamer

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
        src_memory, src_mask, init_states = self._encode(src, src_lens)
        init_output = self._decoder.get_init_output(src_memory, src_lens, init_states)
        return self._decoder(src_memory, src_mask, init_states, init_output, trg)

    def _encode(self, src, src_lens):
        # src: Tensor (batch_size, time_step)
        # src_lens: list (batch_size,)
        src_embedding = self._embedding(src)
        encoder_output, final_encoder_states = self._encoder(src_embedding, src_lens)
        src_memory, init_states = self._bridge(encoder_output, final_encoder_states)
        src_mask = len_mask(src_lens, src_memory.size(1))
        return src_memory, src_mask, init_states

    def decode(self, src, src_lens, max_len):
        # src: Tensor (batch_size, time_step)
        # src_lens: list (batch_size,)
        src_memory, src_mask, init_states = self._encode(src, src_lens)
        init_output = self._decoder.get_init_output(src_memory, src_lens, init_states)
        batch_size = src_memory.size(0)
        token = torch.tensor([SOS_INDEX] * batch_size).unsqueeze(1).cuda()
        states = init_states
        output = init_output
        outputs = []
        for _ in range(max_len):
            logit, states, output = self._decoder.step(src_memory, src_mask, token, states, output)
            token = torch.max(logit, dim=1, keepdim=True)[1]
            outputs.append(token[:, 0])
        outputs = torch.stack(outputs, dim=1)
        return outputs

    def beam_decode(self, src, src_lens, max_len, beam_size):
        # src: Tensor (batch_size, time_step)
        # src_lens: list (batch_size,)
        # max_len: int
        # beam_size: int
        src_memory, src_mask, init_states = self._encode(src, src_lens)
        init_output = self._decoder.get_init_output(src_memory, src_lens, init_states)
        batch_size, time_step, hidden_size = src.size()
        src_memory = src_memory.repeat(beam_size, 1, 1).view(beam_size * batch_size, time_step, hidden_size).contiguous()
        src_mask = src_mask.repeat(beam_size, 1, 1).view(beam_size * batch_size, time_step, hidden_size).contiguous()
        beamer = Beamer(
            states=init_states,
            output=init_output,
            beam_size=beam_size
        )
        for _ in range(max_len):
            token, states, output = beamer.pack_batch()
            logit, states, output = self._decoder.step(
                src_memory, src_mask, token, states, output
            )
            log_prob = F.log_softmax(logit, dim=-1)
            log_prob, token = log_prob.topk(k=beam_size, dim=-1)
            beamer.next_beam(token, log_prob, states, output)
        outputs = beamer.get_best_sequences()
        return outputs