import torch
from torch import nn
import torch.nn.functional as F
from src.module.encoder import Encoder
from src.module.decoder import Decoder

class Seq2Seq(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, rnn_type='LSTM',
                 num_layers=1, bidirectional=False, dropout=0, weight_tying=True):
        super(Seq2Seq, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.encoder = Encoder(
            embedding=self.embedding,
            rnn_type=rnn_type,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout
        )
        self.decoder = Decoder(
            embedding=self.embedding,
            rnn_type=rnn_type,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            weight_tying=weight_tying
        )

    def forward(self, src, trg):
        """
        :param src: LongTensor (batch_size, src_time_step)
        :param trg: LongTensor (batch_size, trg_time_step)
        :return logit: FloatTensor (batch_size, trg_time_step, vocab_size)
        """
        src, src_mask, final_states = self.encoder(src)
        logit = self.decoder(src, src_mask, final_states, trg)
        return logit