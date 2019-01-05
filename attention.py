import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import INF

class Attention(nn.Module):

    def __init__(self):
        super(Attention, self).__init__()
        pass

    def forward(self, query, key, value, mask):
        score = self._score(query, key)
        probability = self._probability_normalize(score, mask)
        output = self._attention_aggregate(probability, value)
        return output

    def _score(self, query, key):
        pass

    def _probability_normalize(self, score, mask):
        score = score.masked_fill(mask==0, -INF)
        probability = F.softmax(score, dim=-1)
        return probability

    def _attention_aggregate(self, probability, value):
        return probability.matmul(value)