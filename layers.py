import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import math, copy, time
from typing import Optional, Any
from torch import Tensor

# @brief clone a module n times
# @credit https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
def clone(module, n):
    modules = nn.ModuleList([copy.deepcopy(module) for i in range(n)])

    # @TODO Implement recursive name definitions that bubble down base classes
    #       when modified
    for i, module in enumerate(modules):
        module.append_to_name(f"_{i}")

    return modules

class Layer(nn.Module):
    def __init__(self, name):
        super(Layer, self).__init__()
        self.name = name

# @brief Basic feedforward NN, capable of transforming n-length input sequences
#        into n-length output sequences, without knowledge of local context
class FeedForwardLayer(Layer):
    # @param d_embed Dimension of sequence elements, used as third tensor dim
    # @param d_hidden Hidden inner dims
    # @TODO Why isn't dropout helping?
    def __init__(self, d_embed: int, d_output: int = None, d_hidden: int = 50, dropout: int = 0.1, name="None"):
        super(FeedForwardLayer, self).__init__(name)

        if not d_output:
            d_output = d_embed

        # set params
        self.fc1 = nn.Linear(d_embed, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_output)
        self.dropout = nn.Dropout(p=dropout)

    # @brief Forward pass on input of dims (batch size, features, seq len)
    # @note Need to broadcast across the sequence dimension rather than feature
    #       dimension so the input and output are transposed
    def forward(self, x, logger_conf):
        x = torch.transpose(x, 1, 2)
        # single layer and relu
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # linear layer for full range of outputs
        x = self.fc2(x)

        return torch.transpose(x, 1, 2)

# @credit Adapted from https://towardsdatascience.com/
#                      how-to-code-the-transformer-in-pytorch-24db27c8f9ec
# @brief Scaled Dot-Product Multi-Headed Attention. Becomes self attention if
#        you pass the same tensor to each of x_k, x_q, x_v in forward pass.
# @brief The Transformer model from "Attention Is All You Need" uses queries
#        from previous decoder layers, and keys/values from the encoder output
# @TODO For faster self attention, allow None-valued x_q, x_v
class MultiHeadAttentionLayer(Layer):
    # @param n_heads Number of attention n_heads, should evenly divide the model
    #              embedding dimension
    # @param d_src Feature dimension of inputs used to compute values and keys.
    #              Is also used as the feature dimension (layer output dim) of
    #              computed values
    # @param d_tgt Feature dimension of inputs used to compute queries. Is also
    #              used as the feature dimension (layer output dim) of computed
    #              queries and keys. The final output has this same feature dimension
    # @param dropout Dropout probability, applied to final attention
    #                weights/scores
    def __init__(self, n_heads, d_src, d_tgt, dropout, name="None"):
        super(MultiHeadAttentionLayer, self).__init__(name)
        self.d_src = d_src
        self.d_tgt_= d_tgt
        self.d_k = d_tgt // n_heads
        self.d_v = d_src // n_heads
        self.h = n_heads
        self.dropout = dropout

        self.q_linear = nn.Linear(d_tgt, d_tgt)
        self.v_linear = nn.Linear(d_src, d_src)
        self.k_linear = nn.Linear(d_src, d_tgt)
        self.drop1 = nn.Dropout(self.dropout)
        self.output_layer = nn.Linear(d_src, d_src)

        # make sure heads evenly divides input's dimension so that we can
        # just divide the output of linear transformations across each head for
        # constant-ish compute
        assert self.d_k * n_heads == d_tgt
        assert self.d_v * n_heads == d_src

    # @brief Forward pass on input of dimension (B, S, E)
    #        - B: Batch size
    #        - S: Sequence length
    #        - E: Element embedding size
    # @param x_q Input used to compute query matrix
    # @param x_k Input used to compute key matrix
    # @param x_v Input used to compute value matrix
    # @return Self-attentive encoding with same dims (B, E, S)
    def forward(self, x_q, x_k, x_v, logger_conf=None, mask=None):
        bs = x_k.size(0)

        # extract keys, values, and queries with lin proj, by each attention head
        k = self.k_linear(x_k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(x_q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(x_v).view(bs, -1, self.h, self.d_v)

        # transpose to get dimensions (batch, head, seq, feats)
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        weighted_values = self.attention(q, k, v, self.d_k, logger_conf, mask, self.drop1)

        # concatenate n_heads and put through final linear layer
        concat = weighted_values.transpose(1,2).contiguous().view(bs, -1, self.d_src)
        output = torch.transpose(self.output_layer(concat), 1, 2)

        return output

    def attention(self, q, k, v, d_k, logger_conf=None, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            mask = mask.unsqueeze(1)
            # @TODO use user-defined device instead of cuda hard-coded
            thing = (mask == True).cuda()
            scores = scores.masked_fill(thing, -1e9)

        scores = F.softmax(scores, dim=-1)

        #if logger_conf['on']:
        #    utils.save_attn_heatmaps(scores, logger_conf['id'], self.name)

        if dropout is not None:
            scores = dropout(scores)

        return torch.matmul(scores, v)

class EncoderLayer(nn.Module):
    def __init__(self, d_embed, n_heads, d_ff_hidden, dropout, d_output = None):
        super(EncoderLayer, self).__init__()
        self.d_embed = d_embed
        self.n_heads = n_heads
        self.dropout = dropout
        self.d_ff_hidden = d_ff_hidden

        if not d_output:
            d_output = self.d_embed
        self.d_output = d_output

        self.bn1 = nn.BatchNorm1d(self.d_embed)

        # self attention layer
        self.self_attn1 = MultiHeadAttentionLayer(self.n_heads,
                self.d_embed, self.d_embed, self.d_embed, self.dropout['attn-self'], name="enc_self_attn1")
        self.bn2 = nn.BatchNorm1d(self.d_embed)

        # feed forward layer
        self.fc1 = FeedForwardLayer(self.d_embed, self.d_output, self.d_ff_hidden, self.dropout['ff'])
        self.bn3 = nn.BatchNorm1d(self.d_output)

    def forward(self, x, x_masks=None, logger_conf=None):
        x = self.bn2(x + self.self_attn1(x, x, x, logger_conf, mask=x_masks))
        x = self.bn3(x + self.fc1(x, logger_conf))

        return x

    def append_to_name(self, text):
        self.self_attn1.name += text
