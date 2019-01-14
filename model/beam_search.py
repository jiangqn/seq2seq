import torch
from collections import Counter
from utils import SOS_INDEX, EOS_INDEX, INF

class BeamNode(object):

    def __init__(self, sequence, log_prob, states, output):
        '''
        sequence: list of int
        log_prob: scalar, float
        states: tuple (hidden, cell)
        hidden: Tensor (num_layers, hidden_size)
        cell: Tensor (num_layers, hidden_size)
        output: Tensor (output_size,)
        '''
        self._sequence = sequence
        self._log_prob = log_prob
        self._states = states
        self._output = output

    def extend(self, token, log_prob, states, output):
        # token: Tensor (beam_size,)
        # log_prob: Tensor (beam_size,)
        # states: tuple (hidden, cell)
        # hidden: Tensor (num_layers, hidden_size)
        # cell: Tensor (num_layers, hidden_size)
        # output: Tensor (output_sizs,)
        beam_size = token.size(0)
        return [
            BeamNode(self._sequence + [token[i]], self._log_prob + log_prob[i], states, output)
            for i in range(beam_size)
        ]

    def __lt__(self, other):
        return (self._log_prob / len(self._sequence)) < (other._log_prob / len(other._sequence))

    def set_log_prob(self, log_prob):
        self._log_prob = log_prob

    def is_finished(self):
        return self._sequence[-1] == EOS_INDEX

    def has_repeat_triple_grams(self):
        triple_grams = [tuple(self._sequence[i: i + 3]) for i in range(len(self._sequence) - 2)]
        count = Counter(triple_grams)
        return not all((count[g] <= 1 for g in count))

    @property
    def hashcode(self):
        hash = 0
        for x in self._sequence:
            hash = hash * 10 + x
        return hash

    def get_sequence(self, max_len):
        sequence = self._sequence if self._sequence[-1] != EOS_INDEX else self._sequence[0: -1]
        sequence = sequence + [0] * (max_len - len(sequence))
        return torch.LongTensor(sequence).cuda()

class BeamNodeGroup(object):

    def __init__(self, states, output, beam_size, remove_repeat_triple_grams):
        '''
        states: tuple (hidden, cell)
        hidden: Tensor (num_layers, hidden_size)
        cell: Tensor (num_layers, hidden_size)
        output: Tensor (output_size,)
        '''
        self._beam_size = beam_size
        self._remove_repeat_triple_grams = remove_repeat_triple_grams
        self._nodes = [
            BeamNode(
                sequence=[SOS_INDEX],
                log_prob=0.0,
                states=states,
                output=output
            ) for _ in range(beam_size)
        ]
        self._finished_nodes = []

    def pack_beam(self):
        token = torch.LongTensor([
            node.sequence[-1] for node in self._nodes
        ]).cuda()
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

    def _unpack_beam(self, token, log_prob, states, output):
        # token: Tensor (beam_size, beam_size)
        # log_prob: Tensor (beam_size, beam_size)
        # states: tuple (hidden, cell)
        # hidden: Tensor (num_layers, beam_size, hidden_size)
        # cell: Tensor (num_layers, beam_size, hidden_size)
        # output: Tensor (beam_size, output_size)
        token_list = []
        log_prob_list = []
        states_list = []
        output_list = []
        for i in range(self._beam_size):
            token_list.append(token[i])
            log_prob_list.append(log_prob[i])
            states_list.append((
                states[0][:, i, :],
                states[1][:, i, :]
            ))
            output_list.append(output[i])
        return token_list, log_prob_list, states_list, output_list

    def get_best_sequence(self, max_len):
        self._nodes = sorted(self._nodes, reverse=True)
        self._finished_nodes = sorted(self._finished_nodes, reverse=True)
        best_node = self._nodes[0] if self._nodes[0] > self._finished_nodes[0] else self._finished_nodes[0]
        return best_node.get_sequence(max_len)

    def next_beam(self, token, log_prob, states, output):
        token_list, log_prob_list, states_list, output_list = self._unpack_beam(token, log_prob, states, output)
        extended_nodes = []
        for node, token, log_prob, states, output in zip(self._nodes, token_list, log_prob_list, states_list, output_list):
            extended_nodes.extend(node.extend(token, log_prob, states, output))
        new_nodes = []
        hashset = set()
        for node in extended_nodes:
            if node.hashcode in hashset:
                continue
            elif node.is_finished():
                self._finished_nodes.append(node)
            else:
                if node.has_repeat_triple_grams():
                    node.set_log_prob(-INF)
                new_nodes.append(node)
        new_nodes = sorted(new_nodes, reverse=True)
        self._nodes = []
        for node in new_nodes:
            self._nodes.append(node)
            if len(self._nodes) >= self._beam_size:
                break
        while len(self._nodes) < self._beam_size:
            self._nodes.append(self._nodes[0])


class Beamer(object):

    def __init__(self, states, output, beam_size, remove_repeat_triple_grams=True):
        '''
        states: tuple (hidden, cell)
        hidden: Tensor (num_layers, batch_size, hidden_size)
        cell: Tensor (num_layers, batch_size, hidden_size)
        output: Tensor (batch_size, output_size)
        '''
        self._groups = []
        self._beam_size = beam_size
        self._batch_size = output.size(0)
        for i in range(self._batch_size):
            self._groups.append(
                BeamNodeGroup(
                    states=(states[0][:, i, :], states[1][:, i, :]),
                    output=output[i],
                    beam_size=beam_size,
                    remove_repeat_triple_grams=remove_repeat_triple_grams
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
        beam_size, batch_size = token.size()
        assert beam_size == self._beam_size and batch_size == self._batch_size
        num_layers, _, _, hidden_size = states[0].size()
        token = token.view(beam_size * batch_size, 1).contiguous()
        states = (
            states[0].view(num_layers, beam_size * batch_size, hidden_size).contiguous(),
            states[1].view(num_layers, beam_size * batch_size, hidden_size).contiguous()
        )
        output = output.view(beam_size * batch_size, -1).contiguous()
        return token, states, output

    def _unpack_batch(self, token, log_prob, states, output):
        # token: Tensor (beam_size * batch_size, beam_size)
        # log_prob: Tensor (beam_size * batch_size, beam_size)
        # states: tuple (hidden, cell)
        # hidden: Tensor (num_layers, beam_size * batch_size, hidden_size)
        # cell: Tensor (num_layers, beam_size * batch_size, hidden_size)
        # output: Tensor (beam_size * batch_size, output_size)
        token = token.view(self._beam_size, self._batch_size, self._beam_size)
        log_prob = log_prob.view(self._beam_size, self._batch_size, self._beam_size)
        num_layers, _, hidden_size = states[0].size()
        states = (
            states[0].view(num_layers, self._beam_size, self._batch_size, hidden_size),
            states[1].view(num_layers, self._beam_size, self._batch_size, hidden_size)
        )
        output = output.view(self._beam_size, self._batch_size, -1)
        token_list = []
        log_prob_list = []
        states_list = []
        output_list = []
        batch_size = token.size(1)
        for i in range(batch_size):
            token_list.append(token[:, i, :])
            log_prob_list.append(log_prob[:, i, :])
            states_list.append((
                states[0][:, :, i, :],
                states[1][:, :, i, :]
            ))
            output_list.append(output[:, i, :])
        return token_list, log_prob_list, states_list, output_list

    def next_beam(self, token, log_prob, states, output):
        token_list, log_prob_list, states_list, output_list = self._unpack_batch(token, log_prob, states, output)
        for group, token, log_prob, states, output in zip(self._groups, token_list, log_prob_list, states_list, output_list):
            group.next_beam(token, log_prob, states, output)

    def get_best_sequences(self, max_len):
        best_sequences = torch.stack([
            group.get_best_sequence(max_len) for group in self._groups
        ], dim=0)
        return best_sequences