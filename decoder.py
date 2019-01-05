import torch
import torch.nn as nn

class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        pass

    def forward(self, src_memory, src_mask, init_decoder_states, trg):
        max_len = trg_embedding.size(1)
        decoder_hidden, decoder_cell = init_decoder_states
        logits = []
        for i in range(max_len):
            token = trg[:, i:i+1]
            pass

    def _step(self):
        pass

    def decode_step(self):
        pass
