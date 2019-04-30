import os
import yaml
from src.model.nmt import NMT
from src.model.nqg import NQG

def make_model(config):
    task = config['task']
    config = config[task]
    log_path = os.path.join(config['base_path'], 'log/log.yml')
    log = yaml.load(open(log_path))
    if task == 'nmt':
        model = NMT(
            src_vocab_size=log['src_vocab_size'],
            trg_vocab_size=log['trg_vocab_size'],
            embed_size=config['embed_size'],
            hidden_size=config['hidden_size'],
            rnn_type=config['rnn_type'],
            num_layers=config['num_layers'],
            bidirectional=config['bidirectional'],
            dropout=config['dropout'],
            weight_tying=config['weight_tying']
        )
    else:
        pass
    return model