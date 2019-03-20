import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder, MultiLayerLSTMCells, MultiLayerGRUCells
from model.seq2seq import Seq2Seq
from model.attention import *
from model.bridge import Bridge

def make_model(config):
    embedding = nn.Embedding(
        num_embeddings=config['vocab_size'],
        embedding_dim=config['embed_size']
    )
    # encoder
    encoder = Encoder(
        rnn_type=config['rnn_type'],
        embed_size=config['embed_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        bidirectional=config['bidirectional'],
        dropout=config['dropout']
    )
    # birdge
    bridge = Bridge(
        rnn_type=config['rnn_type'],
        hidden_size=config['hidden_size'],
        bidirectional=config['bidirectional']
    )
    # decoder rnn cell
    if config['rnn_type'] == 'LSTM':
        rnn_cell = MultiLayerLSTMCells(
            input_size=2 * config['embed_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        )
    else:
        rnn_cell = MultiLayerGRUCells(
            input_size=2 * config['embed_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        )
    # attention
    if config['attention_type'] == 'Dot':
        attention = DotAttention()
    elif config['attention_type'] == 'ScaledDot':
        attention = ScaledDotAttention()
    elif config['attention_type'] == 'Additive':
        attention = AdditiveAttention(
            query_size=config['hidden_size'],
            key_size=config['hidden_size']
        )
    elif config['attention_type'] == 'Multiplicative':
        attention = MultiplicativeAttention(
            query_size=config['hidden_size'],
            key_size=config['hidden_size']
        )
    elif config['attention_type'] == 'MLP':
        attention = MultiLayerPerceptronAttention(
            query_size=config['hidden_size'],
            key_size=config['hidden_size'],
            out_size=1
        )
    else:
        raise ValueError('No Supporting.')
    # decoder
    decoder = Decoder(embedding, rnn_cell, attention, config['hidden_size'])
    # model
    model = Seq2Seq(embedding, encoder, bridge, decoder)
    return model