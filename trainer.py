import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.encoder import Encoder
from model.bridge import Bridge
from model.decoder import Decoder, MultiLayerLSTMCells
from model.seq2seq import Seq2Seq
from dataset import Dataset

class Trainer(object):

    def __init__(self, config):
        self._config = config

    def _make_model(self):
        embedding = nn.Embedding(self._config.vocab_size, self._config.embed_size)
        encoder = Encoder(self._config.embed_size, self._config.hidden_size, self._config.num_layers,
                          self._config.bidirectional, self._config.dropout)
        bridge = Bridge(self._config.hidden_size, self._config.bidirectional)
        lstm_cell = MultiLayerLSTMCells(self._config.embed_size + self._config.hidden_size, self._config.hidden_size,
                                        self._config.num_layers, dropout=self._config.dropout)
        decoder = Decoder(embedding, lstm_cell, self._config.hidden_size)
        model = Seq2Seq(embedding, encoder, bridge, decoder)
        return model

    def _make_data(self):
        train_dataset = Dataset(self._config.train_path)
        dev_dataset = Dataset(self._config.dev_path)
        train_loader = DataLoader(train_dataset, self._config.batch_size, shuffle=True, num_workers=2)
        dev_loader = DataLoader(dev_dataset, self._config.batch_size, shuffle=True, num_workers=2)
        return train_loader, dev_loader

    def run(self):
        model = self._make_model()
        model = model.cuda()
        train_loader, dev_loader = self._make_data()
        for epoch in range(self._config.num_epoches):
            for data in train_loader:
                src, src_lens, trg, trg_lens = data
                src, src_lens, trg, trg_lens = src.cuda(), src_lens.cuda(), trg.cuda(), trg_lens.cuda()
                logits = model(src, src_lens, trg)

    def _loss(self, logits, trg, trg_lens):
        pass