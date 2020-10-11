import torch
import torch.nn as nn
import math
from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.nn import functional as F
from torch.nn.parameter import Parameter

init_emb_size = 100
gc1_emb_size = 150
emb_dim = 200
conv1_size = 64
conv2_size = 128
conv3_size = 64
fc_size = 64 * 48


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
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
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
        self.gc_e1 = GraphConvolution(init_emb_size, gc1_emb_size)
        self.gc_e2 = GraphConvolution(gc1_emb_size, emb_dim)
        self.bn1 = nn.BatchNorm1d(gc1_emb_size)
        self.bn2 = nn.BatchNorm1d(emb_dim)

        self.emb_r = nn.Embedding(num_relation, emb_dim, padding_idx=0)
#         self.gc_r1 = GraphConvolution(init_emb_size, gc1_emb_size)
#         self.gc_r2 = GraphConvolution(gc1_emb_size, emb_dim)
        self.bn3 = nn.BatchNorm1d(gc1_emb_size)
        self.bn4 = nn.BatchNorm1d(emb_dim)

        self.conv1 = nn.Conv1d(in_channels=2, out_channels=conv1_size, kernel_size=1)
        self.bn5 = nn.BatchNorm1d(conv1_size)
        self.drop1 = nn.Dropout(p=0.5)
        self.conv2 = nn.Conv1d(in_channels=conv1_size, out_channels=conv2_size, kernel_size=3) # 128, 198
        self.bn6 = nn.BatchNorm1d(conv2_size)
        self.drop2 = nn.Dropout(p=0.5)
        self.pool1 = nn.MaxPool1d(kernel_size=2) # 128, 99
        self.conv3 = nn.Conv1d(in_channels=conv2_size, out_channels=conv3_size, kernel_size=3) # 64, 97
        self.bn7 = nn.BatchNorm1d(conv3_size)
        self.drop3 = nn.Dropout(p=0.5)
        self.pool2 = nn.MaxPool1d(kernel_size=2) # 64, 48
        self.fc1 = nn.Linear(in_features=fc_size, out_features=emb_dim)

        #         self.loss = torch.nn.BCELoss()

        self.loss = torch.nn.MSELoss()

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_r.weight.data)
        xavier_normal_(self.gc_e1.weight.data)
        xavier_normal_(self.gc_e2.weight.data)
#         xavier_normal_(self.gc_r1.weight.data)
#         xavier_normal_(self.gc_r2.weight.data)

    def forward(self, e1, rel, X_e, X_r, adj):
        e = self.emb_e(X_e)
        e = self.bn1(self.gc_e1(e, adj))
        #         e = F.relu(e)
        emb_all = self.bn2(self.gc_e2(e, adj))
        #         e = F.relu(e)
        e = emb_all[e1]
        e = e.unsqueeze(1)

        r = self.emb_r(X_r)
        #         r = self.bn3(self.gc_r1(r, rm))
        #         r = F.relu(r)
        #         r = self.bn4(self.gc_r2(r, rm))
        #         r = F.relu(r)
        r = r[rel]
        r = r.unsqueeze(1)

        # x = torch.cat([e, r], dim=1).unsqueeze(1)
        x = torch.cat([e, r], dim=1)
        x = self.bn5(self.conv1(x))
        x = F.relu(x)
        x = self.drop1(x)
        x = self.bn6(self.conv2(x))
        x = F.relu(x)
        x = self.drop2(self.pool1(x))
        x = self.bn7(self.conv3(x))
        x = F.relu(x)
        x = self.drop3(self.pool2(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, emb_all.transpose(1, 0))
        pred = F.sigmoid(x)
        return pred


# SACN
class SACN(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(SACN, self).__init__()

        self.emb_e = torch.nn.Embedding(num_entities, 100, padding_idx=0)
        self.gc1 = GraphConvolution(100, 150)
        self.gc2 = GraphConvolution(150, 200)
        self.emb_rel = torch.nn.Embedding(num_relations, 200, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(0.0)
        self.hidden_drop = torch.nn.Dropout(0.25)
        self.feature_map_drop = torch.nn.Dropout(0.25)
        self.loss = torch.nn.BCELoss()
        self.conv1 = torch.nn.Conv1d(2, 200, 5, stride=1, padding= int(math.floor(5/2)))
        self.bn0 = torch.nn.BatchNorm1d(2)
        self.bn1 = torch.nn.BatchNorm1d(200)
        self.bn2 = torch.nn.BatchNorm1d(200)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc1 = torch.nn.Linear(200 * 200, 200)
        self.bn3 = torch.nn.BatchNorm1d(150)
        self.bn4 = torch.nn.BatchNorm1d(200)
        self.bn_init = torch.nn.BatchNorm1d(100)

        print(num_entities, num_relations)

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)
        xavier_normal_(self.gc1.weight.data)
        xavier_normal_(self.gc2.weight.data)

    def forward(self, e1, rel, X, A):
        emb_initial = self.emb_e(X)
        x = self.gc1(emb_initial, A)
        x = self.bn3(x)
        x = torch.tanh(x)
        x = F.dropout(x, 0.25, training=self.training)
        x = self.bn4(self.gc2(x, A))
        e1_embedded_all = torch.tanh(x)
        e1_embedded_all = F.dropout(e1_embedded_all, 0.25, training=self.training)
        e1_embedded = e1_embedded_all[e1]
        e1_embedded = e1_embedded.unsqueeze(1)
        rel_embedded = self.emb_rel(rel)
        rel_embedded = rel_embedded.unsqueeze(1)
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, e1_embedded_all.transpose(1, 0))
        pred = F.sigmoid(x)
        return pred