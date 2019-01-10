import torch
import spacy
import re
import os
import numpy as np

INIT = 1e-2
INF = 1e18

PAD = '<PAD>'
SOS = '<SOS>'
EOS = '<EOS>'
UNK = '<UNK>'

PAD_INDEX = 0
SOS_INDEX = 1
EOS_INDEX = 2
UNK_INDEX = 3

url = re.compile('(<url>.*</url>)')
spacy_en = spacy.load('en')

def tokenize(text):
    return [tok.text for tok in spacy_en.tokenizer(url.sub('@URL@', text))]

def reorder_sequence(sequence_embedding, order):
    # sequence_embedding: Tensor (batch_size, time_step, embed_size)
    # order: list (batch_size,)
    assert sequence_embedding.size(0) == len(order)
    order = torch.LongTensor(order).cuda()
    return sequence_embedding.index_select(index=order, dim=0)

def reorder_lstm_states(states, order):
    # states: (hidden, cell)
    # hidden: Tensor (num_layers * num_directions, batch_size, hidden_size)
    # cell: Tensor (num_layers * num_directions, batch_size, hidden_size)
    assert isinstance(states, tuple)
    assert len(states) == 2
    assert states[0].size(1) == len(order)
    order = torch.LongTensor(order).cuda()
    return (
        states[0].index_select(index=order, dim=1),
        states[1].index_select(index=order, dim=1)
    )

def sequence_mean(sequence, seq_lens, dim):
    assert sequence.size(0) == len(seq_lens)
    seq_sum = torch.sum(sequence, dim=dim, keepdim=False)
    seq_mean = torch.stack([s / l for s, l in zip(seq_sum, seq_lens)], dim=0)
    return seq_mean

def len_mask(seq_lens, max_len):
    batch_size = len(seq_lens)
    mask = torch.ByteTensor(batch_size, max_len).cuda()
    mask.fill_(0)
    for i, l in enumerate(seq_lens):
        mask[i, :l].fill_(1)
    return mask

def load_word_embeddings(fname, vocab_size, embed_size, word2index):
    if not os.path.isfile(fname):
        raise IOError(ENOENT, 'Not a file', fname)

    word2vec = np.random.uniform(-0.01, 0.01, [vocab_size, embed_size])
    with open(fname, 'r', encoding='utf-8') as f:
        for line in f:
            content = line.split(' ')
            if content[0] in word2index:
                word2vec[word2index[content[0]]] = np.array(list(map(float, content[1:])))
    word2vec[word2index[PAD], :] = 0
    return word2vec

def sentence_clip(sentences, lens):
    max_len = max(lens)
    sentences = sentences[:, 0:max_len].contiguous()
    return sentences