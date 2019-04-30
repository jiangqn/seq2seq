import os
import yaml
import numpy as np
import pickle
from data_process.vocab import Vocab
from data_process.tokenizer import fair_tokenizer, nltk_tokenizer, spacy_en_tokenizer, spacy_de_tokenizer
from data_process.utils import text_file2word_lists, word_lists2numpy, analyze

def data_process(config):
    task = config['task']
    config = config[task]

    src_tokenizer = fair_tokenizer
    trg_tokenizer = fair_tokenizer

    src_train_text = open(os.path.join(config['base_path'], 'raw/src_train.txt'), 'r', encoding='utf-8')
    trg_train_text = open(os.path.join(config['base_path'], 'raw/trg_train.txt'), 'r', encoding='utf-8')
    src_val_text = open(os.path.join(config['base_path'], 'raw/src_val.txt'), 'r', encoding='utf-8')
    trg_val_text = open(os.path.join(config['base_path'], 'raw/trg_val.txt'), 'r', encoding='utf-8')
    src_test_text = open(os.path.join(config['base_path'], 'raw/src_test.txt'), 'r', encoding='utf-8')
    trg_test_text = open(os.path.join(config['base_path'], 'raw/trg_test.txt'), 'r', encoding='utf-8')

    src_train_word_lists = text_file2word_lists(src_train_text, src_tokenizer)
    trg_train_word_lists = text_file2word_lists(trg_train_text, trg_tokenizer)
    src_val_word_lists = text_file2word_lists(src_val_text, src_tokenizer)
    trg_val_word_lists = text_file2word_lists(trg_val_text, trg_tokenizer)
    src_test_word_lists = text_file2word_lists(src_test_text, src_tokenizer)
    trg_test_word_lists = text_file2word_lists(trg_test_text, trg_tokenizer)

    if task == 'nmt':
        src_vocab = Vocab()
        trg_vocab = Vocab()
        for word_list in src_train_word_lists:
            src_vocab.add_list(word_list)
        for word_list in trg_train_word_lists:
            trg_vocab.add_list(word_list)

        src_word2index, src_index2word = src_vocab.get_vocab()
        trg_word2index, trg_index2word = trg_vocab.get_vocab()
    elif task == 'nqg':
        vocab = Vocab()
        for word_list in src_train_word_lists:
            vocab.add_list(word_list)
        for word_list in trg_train_word_lists:
            vocab.add_list(word_list)
        src_word2index, src_index2word = vocab.get_vocab()
        trg_word2index, trg_index2word = src_word2index, src_index2word
    else:
        raise ValueError('No Supporting.')

    src_train = word_lists2numpy(src_train_word_lists, src_word2index)
    trg_train = word_lists2numpy(trg_train_word_lists, trg_word2index)
    src_val = word_lists2numpy(src_val_word_lists, src_word2index)
    trg_val = word_lists2numpy(trg_val_word_lists, trg_word2index)
    src_test = word_lists2numpy(src_test_word_lists, src_word2index)
    trg_test = word_lists2numpy(trg_test_word_lists, trg_word2index)

    if not os.path.exists(os.path.join(config['base_path'], 'processed')):
        os.makedirs(os.path.join(config['base_path'], 'processed'))

    np.savez(os.path.join(config['base_path'], 'processed/train.npz'), src=src_train, trg=trg_train)
    np.savez(os.path.join(config['base_path'], 'processed/val.npz'), src=src_val, trg=trg_val)
    np.savez(os.path.join(config['base_path'], 'processed/test.npz'), src=src_test, trg=trg_test)

    if task == 'nmt':
        with open(os.path.join(config['base_path'], 'processed/src_word2index.pkl'), 'wb') as handle:
            pickle.dump(src_word2index, handle)
        with open(os.path.join(config['base_path'], 'processed/src_index2word.pkl'), 'wb') as handle:
            pickle.dump(src_index2word, handle)
        with open(os.path.join(config['base_path'], 'processed/trg_word2index.pkl'), 'wb') as handle:
            pickle.dump(trg_word2index, handle)
        with open(os.path.join(config['base_path'], 'processed/trg_index2word.pkl'), 'wb') as handle:
            pickle.dump(trg_index2word, handle)
        log = {
            'src_vocab_size': len(src_index2word),
            'src_oov_size': len(src_word2index) - len(src_index2word),
            'trg_vocab_size': len(trg_index2word),
            'trg_oov_size': len(trg_word2index) - len(trg_index2word),
            'train_data': analyze(src_train_word_lists, trg_train_word_lists),
            'val_data': analyze(src_val_word_lists, trg_val_word_lists),
            'test_data': analyze(src_test_word_lists, trg_test_word_lists)
        }
    else:
        with open(os.path.join(config['base_path'], 'processed/word2index.pkl'), 'wb') as handle:
            pickle.dump(src_word2index, handle)
        with open(os.path.join(config['base_path'], 'processed/index2word.pkl'), 'wb') as handle:
            pickle.dump(src_index2word, handle)
        log = {
            'vocab_size': len(src_index2word),
            'oov_size': len(src_word2index) - len(src_index2word),
            'train_data': analyze(src_train_word_lists, trg_train_word_lists),
            'val_data': analyze(src_val_word_lists, trg_val_word_lists),
            'test_data': analyze(src_test_word_lists, trg_test_word_lists)
        }

    if not os.path.exists(os.path.join(config['base_path'], 'log')):
        os.makedirs(os.path.join(config['base_path'], 'log'))
    with open(os.path.join(config['base_path'], 'log/log.yml'), 'w') as handle:
        yaml.safe_dump(log, handle, encoding='utf-8', allow_unicode=True, default_flow_style=False)