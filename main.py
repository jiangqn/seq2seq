import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from model.encoder import Encoder
from model.bridge import Bridge
from model.attention import DotAttention, ScaledDotAttention, AdditiveAttention, MultiplicativeAttention, MultiLayerPerceptronAttention
from model.decoder import Decoder, MultiLayerLSTMCells, MultiLayerGRUCells
from model.seq2seq import Seq2Seq
from dataset import Seq2SeqDataset
from model.utils import len_mask, EOS, PAD, sentence_clip
import pickle

class Trainer(object):

    def __init__(self, config):
        self._config = config

    def _make_model(self):
        # embedding
        embedding = nn.Embedding(
            num_embeddings=self._config.vocab_size,
            embedding_dim=self._config.embed_size
        )
        embedding.weight.data.copy_(torch.from_numpy(np.load(self._config.embedding_file_name)))
        embedding.weight.requires_grad = False
        # encoder
        encoder = Encoder(
            rnn_type=self._config.rnn_type,
            embed_size=self._config.embed_size,
            hidden_size=self._config.hidden_size,
            num_layers=self._config.num_layers,
            bidirectional=self._config.bidirectional,
            dropout=self._config.dropout
        )
        # birdge
        bridge = Bridge(
            rnn_type=self._config.rnn_type,
            hidden_size=self._config.hidden_size,
            bidirectional=self._config.bidirectional
        )
        # decoder rnn cell
        if self._config.rnn_type == 'LSTM':
            rnn_cell = MultiLayerLSTMCells(
                input_size=2 * self._config.embed_size,
                hidden_size=self._config.hidden_size,
                num_layers=self._config.num_layers,
                dropout=self._config.dropout
            )
        else:
            rnn_cell = MultiLayerGRUCells(
                input_size=2 * self._config.embed_size,
                hidden_size=self._config.hidden_size,
                num_layers=self._config.num_layers,
                dropout=self._config.dropout
            )
        # attention
        if self._config.attention_type == 'Dot':
            attention = DotAttention()
        elif self._config.attention_type == 'ScaledDot':
            attention = ScaledDotAttention()
        elif self._config.attention_type == 'Additive':
            attention = AdditiveAttention(
                query_size=self._config.hidden_size,
                key_size=self._config.hidden_size
            )
        elif self._config.attention_type == 'Multiplicative':
            attention = MultiplicativeAttention(
                query_size=self._config.hidden_size,
                key_size=self._config.hidden_size
            )
        elif self._config.attention_type == 'MLP':
            attention = MultiLayerPerceptronAttention(
                query_size=self._config.hidden_size,
                key_size=self._config.hidden_size,
                out_size=1
            )
        else:
            raise ValueError('No Supporting.')
        # decoder
        decoder = Decoder(embedding, rnn_cell, attention, self._config.hidden_size)
        # model
        model = Seq2Seq(embedding, encoder, bridge, decoder)
        return model

    def _make_data(self):
        train_dataset = Seq2SeqDataset(self._config.train_path)
        dev_dataset = Seq2SeqDataset(self._config.dev_path)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self._config.batch_size,
            shuffle=True,
            num_workers=2
        )
        dev_loader = DataLoader(
            dataset=dev_dataset,
            batch_size=self._config.batch_size,
            shuffle=False,
            num_workers=2
        )
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
                optimizer.zero_grad()
                logits = model(src, src_lens, sentence_clip(trg[:, 0:-1], trg_lens), teacher_forcing_ratio=0.8)
                loss = self._loss(logits, sentence_clip(trg[:, 1:], trg_lens), trg_lens, criterion)
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
                if word == EOS or word == PAD:
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
                output = model.beam_decode(src, src_lens, max(trg_lens) + 1, beam_size=3)
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

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

parser = argparse.ArgumentParser()
parser.add_argument('--rnn_type', type=str, default='LSTM')
parser.add_argument('--attention_type', type=str, default='Multiplicative', choices=['Dot', 'ScaledDot', 'Additive', 'Multiplicative', 'MLP'])
parser.add_argument('--embed_size', type=int, default=300)
parser.add_argument('--vocab_size', type=int, default=37411)
parser.add_argument('--hidden_size', type=int, default=600)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--bidirectional', type=bool, default=True)
parser.add_argument('--num_epoches', type=int, default=30)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--l2_reg', type=float, default=0)
parser.add_argument('--clip', type=float, default=5.0)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--embedding_file_name', type=str, default='data/vocab/glove.npy')
parser.add_argument('--vocab_path', type=str, default='./data/vocab/index2word.pickle')
parser.add_argument('--train_path', type=str, default='./data/processed/train.npz')
parser.add_argument('--dev_path', type=str, default='./data/processed/dev.npz')

config = parser.parse_args()

trainer = Trainer(config)
trainer.run()