import torch
from utils import SOS_INDEX

class BeamNode(object):

    def __init__(self, sequence, log_prob, states, output):
        '''
        sequence: list of int
        log_prob: scalar, float
        states: tuple (hidden, cell)
        output: Tensor
        '''
        self._sequence = sequence
        self._log_prob = log_prob
        self._states = states
        self._output = output

class BeamNodeGroup(object):

    def __init__(self):
        self._group = []

    @property
    def size(self):
        return len(self._group)

class Beamer(object):

    def __init__(self):
        pass

    def init_beam(self, states, output):
        pass