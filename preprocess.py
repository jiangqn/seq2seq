import os
import spacy

base_path = './data'
train_src_path = os.path.join(base_path, 'src-train.txt')
train_trg_path = os.path.join(base_path, 'trg-train.txt')
dev_src_path = os.path.join(base_path, 'src-dev.txt')
dev_trg_path = os.path.join(base_path, 'trg-dev.txt')
test_src_path = os.path.join(base_path, 'src-test.txt')
test_trg_path = os.path.join(base_path, 'trg-test.txt')

num_train = 0
num_dev = 0
num_test = 0

def process(src_path, trg_path):
    _src_max_len, _trg_max_len = 0, 0
    _src = open(src_path, 'r', encoding=u'utf-8').readlines()
    _trg = open(trg_path, 'r', encoding=u'utf-8').readlines()
    assert len(_src) == len(_trg)
    _num = len(_src)
    for i in range(_num):
        _src_piece, _trg_piece = _src[i], _trg[i]