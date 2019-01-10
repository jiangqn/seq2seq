import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from model.encoder import Encoder
from model.bridge import Bridge
from model.attention import DotAttention, ScaledDotAttention, AdditiveAttention, MultiplicativeAttention, MultiLayerPerceptronAttention
from model.decoder import Decoder, MultiLayerLSTMCells
from model.seq2seq import Seq2Seq
from dataset import Seq2SeqDataset
from model.utils import len_mask, EOS, sentence_clip
import pickle

class Trainer(object):

    def __init__(self, config):
        self._config = config

    def _make_model(self):
        embedding = nn.Embedding(self._config.vocab_size, self._config.embed_size)
        embedding.weight.data.copy_(torch.from_numpy(np.load(self._config.embedding_file_name)))
        embedding.weight.requires_grad = False
        encoder = Encoder(self._config.embed_size, self._config.hidden_size, self._config.num_layers,
                          self._config.bidirectional, self._config.dropout)
        bridge = Bridge(self._config.hidden_size, self._config.bidirectional)
        lstm_cell = MultiLayerLSTMCells(2 * self._config.embed_size , self._config.hidden_size,
                                        self._config.num_layers, dropout=self._config.dropout)
        # attention = MultiplicativeAttention(self._config.hidden_size, self._config.hidden_size)
        attention = AdditiveAttention(self._config.hidden_size, self._config.hidden_size)
        decoder = Decoder(embedding, lstm_cell, attention, self._config.hidden_size)
        model = Seq2Seq(embedding, encoder, bridge, decoder)
        return model

    def _make_data(self):
        train_dataset = Seq2SeqDataset(self._config.train_path)
        dev_dataset = Seq2SeqDataset(self._config.dev_path)
        train_loader = DataLoader(train_dataset, self._config.batch_size, shuffle=True, num_workers=2)
        dev_loader = DataLoader(dev_dataset, self._config.batch_size, shuffle=False, num_workers=2)
        return train_loader, dev_loader

    def _make_vocab(self):
        with open(self._config.vocab_path, 'rb') as handle:
            self._index2word = pickle.load(handle)

    def run(self):
        self._make_vocab()
        model = self._make_model()
        model = model.cuda()
        print(model)
        criterion = nn.CrossEntropyLoss(reduction='none')
        optimizer = optim.Adam(model.parameters(), lr=self._config.learning_rate)
        train_loader, dev_loader = self._make_data()
        for epoch in range(1, self._config.num_epoches + 1):
            sum_loss = 0
            sum_examples = 0
            s_loss = 0
            for i, data in enumerate(train_loader):
                src, src_lens, trg, trg_lens = data
                src, src_lens, trg, trg_lens = src.cuda(), src_lens.tolist(), trg.cuda(), trg_lens.tolist()
                src = sentence_clip(src, src_lens)
                trg = sentence_clip(trg, trg_lens)
                optimizer.zero_grad()
                logits = model(src, src_lens, trg[:, 0:-1])
                loss = self._loss(logits, trg[:, 1:], trg_lens, criterion)
                sum_loss += loss.item() * src.size(0)
                sum_examples += src.size(0)
                s_loss += loss.item()
                if i > 0 and i % 100 == 0:
                    s_loss /= 100
                    print('[epoch %2d] [step %4d] [loss %.4f]' % (epoch, i, s_loss))
                    s_loss = 0
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self._config.clip)
                optimizer.step()
            avg_loss = sum_loss / sum_examples
            print('[epoch %2d] [loss %.4f]' % (epoch, avg_loss))
            self._eval(model, dev_loader, epoch)
            self._save_model(model, epoch)

    def _loss(self, logits, trg, trg_lens, criterion):
        # logits: Tensor (batch_size, time_step, vocab_size)
        # trg: Tensor (batch_size, time_step)
        # trg_lens: list (batch_size,)
        mask = len_mask(trg_lens, trg.size(1))
        vocab_size = logits.size(2)
        logits = logits.view(-1, vocab_size)
        trg = trg.contiguous().view(-1)
        mask = mask.view(-1)
        losses = criterion(logits, trg).masked_select(mask)
        loss = losses.mean()
        return loss

    def _tensor2texts(self, tensor):
        texts = []
        for vector in tensor:
            text = ''
            for index in vector:
                word = self._index2word[index.item()]
                if word == EOS:
                    break
                else:
                    text += word + ' '
            texts.append(text.strip())
        return texts

    def _eval(self, model, data_loader, epoch=None):
        pred = []
        for data in data_loader:
            src, src_lens, trg, trg_lens = data
            src, src_lens, trg_lens = src.cuda(), src_lens.tolist(), trg_lens.tolist()
            with torch.no_grad():
                output = model.decode(src, src_lens, max(trg_lens) + 1)
                texts = self._tensor2texts(output)
                print(texts[0])
                pred.extend(texts)
        path = './data/output/pred' + (('-epoch-' + str(epoch)) if epoch is not None else '') + '.txt'
        self._write_file(pred, path)

    def _write_file(self, texts, path):
        file = open(path, 'w', encoding=u'utf-8')
        for text in texts:
            file.write(text + '\n')

    def _save_model(self, model, epoch=None):
        path = './data/checkpoints/model' + (('-epoch-' + str(epoch)) if epoch is not None else '') + '.pkl'
        torch.save(model, path)