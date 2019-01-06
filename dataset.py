import operator
from utils import PAD, SOS, EOS, UNK

class Vocab(object):

    def __init__(self):
        self._count_dict = dict()
        self._predefined_list = [PAD, SOS, EOS, UNK]

    def add(self, word):
        if word in self._count_dict:
            self._count_dict[word] += 1
        else:
            self._count_dict[word] = 1

    def get_vocab(self, max_num=None, min_freq=0):
        sorted_words = sorted(self._count_dict.items(), key=operator.itemgetter(1), reverse=True)
        word2index = {}
        for word in self._predefined_list:
            word2index[word] = len(word2index)
        for word, freq in sorted_words:
            if (max_num is not None and len(word2index) >= max_num) or freq < min_freq:
                word2index[word] = word2index[UNK]
            else:
                word2index[word] = len(word2index)
        index2word = {}
        index2word[word2index[UNK]] = UNK
        for word, index in word2index:
            if index == word2index[UNK]:
                continue
            else:
                index2word[index] = word
        return word2index, index2word