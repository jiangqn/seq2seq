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

    def __init__(self, states, output):
        self._nodes = []
        self._nodes.append(
            BeamNode(
                sequence=[SOS_INDEX],
                log_prob=0.0,
                states=states,
                output=output
            )
        )

    def pack_beam(self):
        token = torch.LongTensor([
            node.sequence[-1] for node in self._nodes
        ])
        states = (
            torch.stack([
                node.states[0] for node in self._nodes
            ], dim=1),
            torch.stack([
                node.states[1] for node in self._nodes
            ], dim=1)
        )
        output = (
            torch.stack([
                node.output for node in self._nodes
            ], dim=0)
        )
        return token, states, output

    def unpack_beam(self):
        pass

class Beamer(object):

    def __init__(self, states, output):
        '''
        states: tuple (hidden, cell)
        hidden: Tensor (num_layers, batch_size, hidden_size)
        cell: Tensor (num_layers, batch_size, hidden_size)
        output: Tensor (batch_size, output_size)
        '''
        self._groups = []
        hidden, cell = states
        batch_size = hidden.size(1)
        for i in range(batch_size):
            self._groups.append(
                BeamNodeGroup(
                    (hidden[:, i, :], cell[:, i, :]),
                    output[i]
                )
            )

    def pack_batch(self):
        token_list = []
        states_list = []
        output_list = []
        for group in self._groups:
            token, states, output = group.pack_beam()
            token_list.append(token)
            states_list.append(states)
            output_list.append(output)
        token = torch.stack(token_list, dim=1)
        states = (
            torch.stack([
                states[0] for states in states_list
            ], dim=2),
            torch.stack([
                states[1] for states in states_list
            ], dim=2)
        )
        output = torch.stack(output_list, dim=1)
        return token, states, output

    def unpack_batch(self):
        pass