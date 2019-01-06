import os
from utils import tokenize
from dataset import Vocab

base_path = './data'
train_src_path = os.path.join(base_path, 'raw/src-train.txt')
train_trg_path = os.path.join(base_path, 'raw/trg-train.txt')
dev_src_path = os.path.join(base_path, 'raw/src-dev.txt')
dev_trg_path = os.path.join(base_path, 'raw/trg-dev.txt')
test_src_path = os.path.join(base_path, 'raw/src-test.txt')
test_trg_path = os.path.join(base_path, 'raw/trg-test.txt')

data_log_path = os.path.join(base_path, 'log/data_log.txt')
log = open(data_log_path, 'w', encoding=u'utf-8')

def process(src_path, trg_path):
    src_max_len, trg_max_len = 0, 0
    src = open(src_path, 'r', encoding=u'utf-8').readlines()
    trg = open(trg_path, 'r', encoding=u'utf-8').readlines()
    assert len(src) == len(trg)
    num = len(src)
    for i in range(num):
        src[i] = tokenize(src[i])
        trg[i] = tokenize(trg[i])
        src_max_len = max(src_max_len, len(src[i]))
        trg_max_len = max(trg_max_len, len(trg[i]))
    return src, trg, num, src_max_len, trg_max_len

train_src, train_trg, train_num, train_src_max_len, train_trg_max_len = process(train_src_path, train_trg_path)
dev_src, dev_trg, dev_num, dev_src_max_len, dev_trg_max_len = process(dev_src_path, dev_trg_path)
test_src, test_trg, test_num, test_src_max_len, test_trg_max_len = process(test_src_path, test_trg_path)

vocab = Vocab()

for i in range(train_num):
    vocab.add_list(train_src[i])
    vocab.add_list(train_trg[i])

for i in range(dev_num):
    vocab.add_list(dev_src[i])
    vocab.add_list(dev_trg[i])

for i in range(test_num):
    vocab.add_list(test_src[i])
    vocab.add_list(test_trg[i])

word2index, index2word = vocab.get_vocab()
vocab_size = len(index2word)

print(train_num, dev_num, test_num)
print(vocab_size)