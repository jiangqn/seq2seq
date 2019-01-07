import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.encoder import Encoder
from model.bridge import Bridge
from model.decoder import Decoder, MultiLayerLSTMCells
from model.seq2seq import Seq2Seq
from dataset import Seq2SeqDataset
from model.utils import len_mask

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
        train_dataset = Seq2SeqDataset(self._config.train_path)
        dev_dataset = Seq2SeqDataset(self._config.dev_path)
        train_loader = DataLoader(train_dataset, self._config.batch_size, shuffle=True, num_workers=2)
        dev_loader = DataLoader(dev_dataset, self._config.batch_size, shuffle=True, num_workers=2)
        return train_loader, dev_loader

    def run(self):
        model = self._make_model()
        model = model.cuda()
        print(model)
        criterion = nn.CrossEntropyLoss(reduction='none')
        optimizer = optim.Adam(model.parameters(), lr=self._config.learning_rate)
        train_loader, dev_loader = self._make_data()
        for epoch in range(self._config.num_epoches):
            sum_loss = 0
            sum_examples = 0
            for data in train_loader:
                src, src_lens, trg, trg_lens = data
                src, src_lens, trg, trg_lens = src.cuda(), src_lens.tolist(), trg.cuda(), trg_lens.tolist()
                optimizer.zero_grad()
                logits = model(src, src_lens, trg)
                loss = self._loss(logits, trg, trg_lens, criterion)
                sum_loss += loss * src.size(0)
                sum_examples += src.size(0)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self._config.clip)
                optimizer.step()
            avg_loss = sum_loss
            print('epoch %d: loss %.4f' % (epoch, avg_loss))

    def _loss(self, logits, trg, trg_lens, criterion):
        # logits: Tensor (batch_size, time_step, vocab_size)
        # trg: Tensor (batch_size, time_step)
        # trg_lens: list (batch_size,)
        mask = len_mask(trg_lens, trg.size(1))
        vocab_size = logits.size(2)
        logits = logits.view(-1, vocab_size)
        trg = logits.view(-1)
        mask = mask.view(-1)
        losses = criterion(logits, trg).masked_select(mask)
        loss = losses.mean()
        return loss