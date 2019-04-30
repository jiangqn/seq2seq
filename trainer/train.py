import torch.nn as nn
import torch.optim as optim
import os
import pickle
from trainer.make_model import make_model
from trainer.make_data import make_train_data
from trainer.masked_cross_entropy import masked_cross_entropy
from trainer.eval import nmt_eval

def train(config):
    model = make_model(config).cuda()
    print(model)
    task = config['task']
    config = config[task]
    vocab_path = os.path.join(config['base_path'], 'processed/' +
        'trg_index2word.pkl' if task == 'nmt' else 'index2word.pkl')
    with open(vocab_path, 'rb') as handle:
        index2word = pickle.load(handle)
    train_loader, val_loader = make_train_data(config)
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    for epoch in range(1, config['num_epoches'] + 1):
        total_samples = 0
        total_loss = 0
        model.train()
        for i, data in enumerate(train_loader):
            src, trg = data
            src, trg = src.cuda(), trg.cuda()
            optimizer.zero_grad()
            logit = model(src, trg[:, 0:-1])
            loss = masked_cross_entropy(logit, trg[:, 1:], criterion)
            total_loss += loss.item() * src.size(0)
            total_samples += src.size(0)
            if i > 0 and i % 10 == 0:
                avg_loss = total_loss / total_samples
                print('[epoch %2d] [step %4d] [train_loss %.4f]' % (epoch, i, avg_loss))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config['clip'])
            optimizer.step()
        val_loss = nmt_eval(model, val_loader, config['max_len'], criterion, index2word)
        print('[epoch %2d] [val_loss %.4f]' % (epoch, val_loss))