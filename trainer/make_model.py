import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder, MultiLayerLSTMCells, MultiLayerGRUCells
from model.seq2seq import Seq2Seq
from model.attention import *
from model.bridge import Bridge

def make_model(config):
    pass