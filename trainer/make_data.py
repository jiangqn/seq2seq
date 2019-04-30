import os
from torch.utils.data import DataLoader
from data_process.dataset import Seq2SeqDataset

def make_train_data(config):
    train_dataset = Seq2SeqDataset(os.path.join(config['base_path'], 'processed/train.npz'))
    val_dataset = Seq2SeqDataset(os.path.join(config['base_path'], 'processed/val.npz'))
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2
    )
    return train_loader, val_loader

def make_test_data(config):
    test_dataset = Seq2SeqDataset(os.path.join(config['base_path'], 'processed/test.npz'))
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2
    )
    return test_loader