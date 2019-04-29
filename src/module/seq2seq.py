import torch
from torch import nn

class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):
        """
        :param src: LongTensor (batch_size, src_time_step)
        :param trg: LongTensor (batch_size, trg_time_step)
        :return logit: FloatTensor (batch_size, trg_time_step, vocab_size)
        """
        src, src_mask, final_states = self.encoder(src)
        logit = self.decoder(src, src_mask, final_states, trg)
        return logit