import torch
from torch import nn
from src.module.encoder import Encoder
from src.module.decoder import Decoder
from src.module.seq2seq import Seq2Seq

class NQG(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, rnn_type='LSTM',
                 num_layers=1, bidirectional=False, dropout=0, weight_tying=True):
        super(NQG, self).__init__()
        embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        encoder = Encoder(
            embedding=embedding,
            rnn_type=rnn_type,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout
        )
        decoder = Decoder(
            embedding=embedding,
            rnn_type=rnn_type,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            weight_tying=weight_tying
        )
        self.seq2seq = Seq2Seq(encoder, decoder)

    def forward(self, src, trg):
        """
        :param src: LongTensor (batch_size, src_time_step)
        :param trg: LongTenor (batch_size, trg_time_step)
        :return logit: FloatTensor (batch_size, trg_time_step)
        """
        return self.seq2seq(src, trg)

    def generate(self, src, max_len):
        self.seq2seq.decoder.decode()