import torch
import torch.nn as nn
import math
from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.nn import functional as F

init_emb_size = 200
gc1_emb_size = 200
emb_dim = 200



class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class IdeaModel(nn.Module):
    def __init__(self, num_entity, num_relation):
        super(IdeaModel, self).__init__()
        self.emb_e = nn.Embedding(num_entity, init_emb_size, padding_idx=0)
        self.emb_r = nn.Embedding(num_relation, init_emb_size, padding_idx=0)
        self.gc1 = GraphConvolution(init_emb_size, gc1_emb_size, num_relation)
        self.gc2 = GraphConvolution(gc1_emb_size, emb_dim, num_relation)
        self.drop = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(gc1_emb_size)
        self.bn2 = nn.BatchNorm1d(init_emb_size)

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_r.weight.data)
        xavier_normal_(self.gc1.weight.data)
        xavier_normal_(self.gc2.weight.data)

    def forward(self, e1, rel, adj):
        e = self.emb_e(e1)
        e = self.bn1(self.gc1(e, adj))
        e = F.relu(e)
        e = F.dropout(e, p=0.4, training=self.training)

        r = self.bn2(self.emb_r(rel))
        r = F.relu(r)

        x = torch.cat([e, r], dim=1)



