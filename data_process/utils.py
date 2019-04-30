import numpy as np
from src.module.utils.constants import UNK, SOS, EOS

def text_file2word_lists(text_file, tokenizer):
    word_lists = []
    for text in text_file.readlines():
        word_lists.append([SOS] + tokenizer(text.strip()) + [EOS])
    return word_lists

def word_lists2numpy(word_lists, word2index):
    num = len(word_lists)
    max_len = 0
    for word_list in word_lists:
        max_len = max(max_len, len(word_list))
    for i in range(num):
        word_lists[i] = list(map(lambda x: word2index[x] if x in word2index else word2index[UNK], word_lists[i]))
        word_lists[i].extend([0] * (max_len - len(word_lists[i])))
    return np.array(word_lists)

def analyze(src_word_lists, trg_word_lists):
    assert len(src_word_lists) == len(trg_word_lists)
    num = len(src_word_lists)
    f = lambda x: len(x)
    src_lens = list(map(f, src_word_lists))
    trg_lens = list(map(f, trg_word_lists))
    return {
        'num': num,
        'src_max_len': max(src_lens),
        'src_avg_len': sum(src_lens) / num,
        'trg_max_len': max(trg_lens),
        'trg_avg_len': sum(trg_lens) / num
    }