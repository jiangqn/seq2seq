import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import INF

class Attention(nn.Module):

    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, query, key, value, mask):
        # query: Tensor (batch_size, hidden_size)
        # key: Tensor (batch_size, time_step, hidden_size)
        # value: Tensor (batch_size, time_step, hidden_size)
        # mask: Tensor (batch_size, time_step)
        score = self._score(query, key)
        probability = self._probability_normalize(score, mask)
        output = self._attention_aggregate(probability, value)
        return output

    def _score(self, query, key):
        return query.unsqueeze(1).matmul(key.transpose(1, 2))

    def _probability_normalize(self, score, mask):
        score = score.masked_fill(mask==0, -INF)
        probability = F.softmax(score, dim=-1)
        return probability

    def _attention_aggregate(self, probability, value):
        return probability.matmul(value).squeeze(1)