import torch
from torch import nn
from src.module.encoder import Encoder
from src.module.decoder import Decoder
from src.module.seq2seq import Seq2Seq

class NMT(nn.Module):

    def __init__(self, src_vocab_size, trg_vocab_size, embed_size, hidden_size,
                 rnn_type='LSTM', num_layers=1, bidirectional=False, dropout=0, weight_tying=True):
        super(NMT, self).__init__()
        src_embedding = nn.Embedding(num_embeddings=src_vocab_size, embedding_dim=embed_size)
        trg_embedding = nn.Embedding(num_embeddings=trg_vocab_size, embedding_dim=embed_size)
        encoder = Encoder(
            embedding=src_embedding,
            rnn_type=rnn_type,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout
        )
        decoder = Decoder(
            embedding=trg_embedding,
            rnn_type=rnn_type,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            weight_tying=weight_tying
        )
        self.seq2seq = Seq2Seq(encoder, decoder)

    def load_encoder_embedding(self):
        pass

    def load_decoder_embedding(self):
        pass

    def forward(self, src, trg):
        """
        :param src: LongTensor (batch_size, src_time_step)
        :param trg: LongTensor (batch_size, trg_time_step)
        :return logit: FloatTensor (batch_size, trg_time_step, trg_vocab_size)
        """
        return self.seq2seq(src, trg)

    def translate(self, src, max_len):
        return self.seq2seq.decode(src, max_len)