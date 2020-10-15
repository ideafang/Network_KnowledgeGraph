import torch
import torch.nn as nn
import math
from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from dgl.nn import GraphConv

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
        #         self.bn3 = nn.BatchNorm1d(gc1_emb_size)
        #         self.bn4 = nn.BatchNorm1d(emb_dim)

        self.conv1 = nn.Conv1d(in_channels=2, out_channels=conv1_size, kernel_size=1)
        self.bn5 = nn.BatchNorm1d(conv1_size)
        self.drop1 = nn.Dropout(p=0.5)
        self.conv2 = nn.Conv1d(in_channels=conv1_size, out_channels=conv2_size, kernel_size=3)  # 128, 198
        self.bn6 = nn.BatchNorm1d(conv2_size)
        self.drop2 = nn.Dropout(p=0.5)
        self.pool1 = nn.MaxPool1d(kernel_size=2)  # 128, 99
        self.conv3 = nn.Conv1d(in_channels=conv2_size, out_channels=conv3_size, kernel_size=3)  # 64, 97
        self.bn7 = nn.BatchNorm1d(conv3_size)
        self.drop3 = nn.Dropout(p=0.5)
        self.pool2 = nn.MaxPool1d(kernel_size=2)  # 64, 48
        self.fc1 = nn.Linear(in_features=fc_size, out_features=emb_dim)
        self.bn8 = nn.BatchNorm1d(emb_dim)

        self.loss = torch.nn.BCELoss()
        #
        # self.loss = torch.nn.MSELoss()

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
        # e = F.relu(e)
        emb_all = self.bn2(self.gc_e2(e, adj))
        # e = F.relu(e)
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
        x = self.bn8(x)
        x = F.relu(x)
        x = torch.mm(x, emb_all.transpose(1, 0))
        pred = torch.sigmoid(x)
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
        self.conv1 = torch.nn.Conv1d(2, 200, 5, stride=1, padding=int(math.floor(5 / 2)))
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


class testModel(torch.nn.Module):
    def __init__(self, num_entity, num_relation):
        super(testModel, self).__init__()
        self.e_emb = nn.Embedding(num_entity, 200)
        self.e_gc1 = GraphConv(in_feats=200, out_feats=200)
        # self.e_gc2 = GraphConv(256, 256)
        self.e_bn1 = nn.BatchNorm1d(200)
        # self.e_bn2 = nn.BatchNorm1d(256)

        self.r_emb = nn.Embedding(num_relation, 200)

        self.x_bn0 = nn.BatchNorm2d(1)
        self.inp_drop = nn.Dropout(0.2)

        self.x1_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, 1))
        self.x2_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, 1))
        self.x2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1)
        self.x3_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, 1))
        self.x3_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), padding=2)
        self.x4_1 = nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=1)
        self.x4_2 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, 1))

        self.x_mp1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.x_dp1 = nn.Dropout(p=0.5)
        self.x_conv1 = nn.Conv1d(in_channels=256, out_channels=64, kernel_size=(3, 3), padding=1)
        self.x_bn = nn.BatchNorm2d(64)
        self.x_dp2 = nn.Dropout(p=0.5)

        self.fc = nn.Linear(in_features=64 * 100, out_features=200)
        self.hidden_drop = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(200)

        self.register_parameter('b', Parameter(torch.zeros(num_entity)))

        self.loss = nn.BCELoss()

    def init(self):
        xavier_normal_(self.e_emb.weight.data)
        xavier_normal_(self.r_emb.weight.data)
        xavier_normal_(self.e_gc1.weight.data)
        # xavier_normal_(self.e_gc2.weight.data)

    def forward(self, e1, r1, X_e, g):
        e_all = self.e_emb(X_e)
#         e_all = self.e_bn1(e_all)
        e_all = self.e_bn1(self.e_gc1(g, e_all))
        # e_all = self.e_bn2(self.e_gc2(g, e_all_1))
        # e_all.add_(e_all_1)
        # e_all = torch.relu(e_all)
        e = e_all[e1]
        e = e.view(e.size(0), 1, 10, 20) # batch, 1, 10, 20
        r = self.r_emb(r1)  # batch, 1, 256
        r = r.view(r.size(0), 1, 10, 20)
        x = torch.cat([e, r], dim=2)  # batch, 1, 20, 20
        x = self.x_bn0(x)
        x = self.inp_drop(x)
        x1 = torch.relu(self.x1_1(x))
        x2 = torch.relu(self.x2_2(torch.relu(self.x2_1(x))))
        x3 = torch.relu(self.x3_2(torch.relu(self.x3_1(x))))
        x4 = torch.relu(self.x4_2(self.x4_1(x)))
        x = torch.cat([x1, x2, x3, x4], dim=1)  # batch, 256, 20, 20
        x = self.x_dp1(self.x_mp1(x))  # batch, 256, 10, 10
        x = self.x_conv1(x)  # batch, 64, 10, 10
        x = self.x_dp2(x)
        x = torch.relu(self.x_bn(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = torch.mm(x, e_all.transpose(1, 0))
        x += self.b.expand_as(x)
        pred = torch.sigmoid(x)
        return pred


# ConvTransE
class ConvTransE(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(ConvTransE, self).__init__()

        self.emb_e = torch.nn.Embedding(num_entities, 100, padding_idx=0)
        self.gc_e1 = GraphConv(100, 100)
        self.bn0_e = nn.BatchNorm1d(100)
        self.gc_e2 = GraphConv(100, 100)
        self.bn1_e = nn.BatchNorm1d(100)
        self.gc_e3 = GraphConv(100, 100)
        self.bn2_e = nn.BatchNorm1d(100)
        self.lstm_e = nn.LSTM(100, 100, 1)  # 输入特征100， 输出100， 1层
        self.emb_rel = torch.nn.Embedding(num_relations, 100, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(0.0)
        self.hidden_drop = torch.nn.Dropout(0.4)
        self.feature_map_drop = torch.nn.Dropout(0.4)
        self.loss = torch.nn.BCELoss()

        self.conv1 =  nn.Conv1d(2, 200, 5, stride=1, padding= int(math.floor(5/2))) # kernel size is odd, then padding = math.floor(kernel_size/2)
        self.bn0 = torch.nn.BatchNorm1d(2)
        self.bn1 = torch.nn.BatchNorm1d(200)
        self.bn2 = torch.nn.BatchNorm1d(100)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(100*200, 100)
        #self.bn3 = torch.nn.BatchNorm1d(Config.gc1_emb_size)
        #self.bn4 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.bn_init = torch.nn.BatchNorm1d(100)

        print(num_entities, num_relations)

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel, X, g):

        emb_initial = self.emb_e(X)
        e1_embedded_all = self.bn_init(emb_initial)  # batch, 14541, 100
        e1_embedded_1 = self.bn0_e(self.gc_e1(g, e1_embedded_all))  # batch, 14541, 100
        e1_embedded_d = F.dropout(e1_embedded_1, 0.25, training=self.training)
        e1_embedded_2 = self.bn1_e(self.gc_e2(g, e1_embedded_d))  # batch, 14541, 100
        e1_embedded_d = F.dropout(e1_embedded_2, 0.25, training=self.training)
        e1_embedded_3 = self.bn2_e(self.gc_e3(g, e1_embedded_d))  # batch, 14541, 100
        e1_embedded_d = F.dropout(e1_embedded_3, 0.25, training=self.training)
        e1_embedded = torch.cat([e1_embedded_all[e1].unsqueeze(1), e1_embedded_1[e1].unsqueeze(1), e1_embedded_2[e1].unsqueeze(1), e1_embedded_3[e1].unsqueeze(1)], dim=1)  # batch, 4, 100
        e1_embedded = e1_embedded.transpose(1, 0)  # 4, batch, 100
        out, (h, c) = self.lstm_e(e1_embedded)  # 1, batch, 100
        e1_embedded = h.transpose(1, 0)  # batch, 1, 100
        rel_embedded = self.emb_rel(rel)  # batch, 100
        rel_embedded = rel_embedded.unsqueeze(1)  # batch, 1, 100
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)  # batch, 2, 100
        stacked_inputs = self.bn0(stacked_inputs)
        x= self.inp_drop(stacked_inputs)
        x= self.conv1(x)  # batch, 200, 200
        x= self.bn1(x)
        x= torch.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = torch.mm(x, e1_embedded_d.transpose(1, 0))
        pred = torch.sigmoid(x)

        return pred


class KerasModel(torch.nn.Module):
    def __init__(self, num_entity, num_rel):
        super(KerasModel, self).__init__()
        self.e_emb = nn.Embedding(num_entity, 200)
        self.e_conv = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)  # batch, 64, 200
        self.e_bn = nn.BatchNorm1d(200)
        self.e_lstm = nn.LSTM(input_size=200, hidden_size=128, batch_first=True)  # batch, 128, 64
        self.e_drop = nn.Dropout(p=0.5)

        self.r_emb = nn.Embedding(num_rel, 200)
        self.r_conv = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)  # batch, 64, 200
        self.r_bn = nn.BatchNorm1d(200)
        self.r_lstm = nn.LSTM(input_size=200, hidden_size=128, batch_first=True)  # batch, 128, 64
        self.r_drop = nn.Dropout(p=0.5)

        self.x_maxpool1 = nn.MaxPool1d(2)  # batch, 128, 64
        self.x_drop1 = nn.Dropout(p=0.5)
        self.x_conv1 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)  # batch, 128, 64
        self.x_bn = nn.BatchNorm1d(128)
        self.x_maxpool2 = nn.MaxPool1d(2)  # batch, 128, 50
        self.x_drop2 = nn.Dropout(p=0.5)
        #         self.x_lstm = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)  # batch, 64, 50
        self.x_conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1)
        self.fc = nn.Linear(in_features=64 * 32, out_features=num_entity)

        self.loss = nn.BCELoss()

    def forward(self, e1, r1, X_e, X_r):
        e_emb = self.e_emb(X_e)
        e = e_emb[e1]
        e = e.unsqueeze(1)
        e = self.e_conv(e)  # batch, 64, 200
        e = e.transpose(1, 2)  # batch, 200, 64
        e = self.e_bn(e)  # batch, 200, 64
        e = torch.relu(e)
        e = e.transpose(1, 2)  # batch, 64, 200
        e, (h, c) = self.e_lstm(e)  # batch, 64, 128
        e = self.e_drop(e)
        e = e.transpose(1, 2)  # batch, 128, 64

        r = self.r_emb(X_r)[r1]
        r = r.unsqueeze(1)
        r = self.r_conv(r)  # batch, 64, 200
        r = r.transpose(1, 2)  # batch, 200, 64
        r = self.r_bn(r)
        r = torch.relu(r)
        r = r.transpose(1, 2)  # batch, 64, 200
        r, (h, c) = self.r_lstm(r)  # batch, 64, 128
        r = self.r_drop(r)
        r = r.transpose(1, 2)  # batch, 128, 64

        x = torch.cat([e, r], dim=2)  # batch, 128, 128(stack)
        x = self.x_drop1(self.x_maxpool1(x))  # batch, 128, 64
        x = self.x_bn(self.x_conv1(x))  # batch, 128, 64
        x = torch.relu(x)
        x = self.x_drop2(self.x_maxpool2(x))  # barch, 128, 32
        #         x = x.transpose(1, 2)
        x = self.x_conv2(x)  # batch, 64, 32
        #         x = x.transpose(1, 2)
        x = x.reshape((x.size(0), -1))
        x = self.fc(x)
        #         x = torch.mm(x, e_emb.transpose(0, 1))
        pred = torch.sigmoid(x)

        return pred


class ConvE(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(ConvE, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, 200, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, 200, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(0.2)
        self.hidden_drop = torch.nn.Dropout(0.3)
        self.feature_map_drop = torch.nn.Dropout2d(0.2)
        self.loss = torch.nn.BCELoss()

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=True)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(200)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(10368, 200)
        print(num_entities, num_relations)

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        e1_embedded = self.emb_e(e1).view(-1, 1, 10, 20)
        rel_embedded = self.emb_rel(rel).view(-1, 1, 10, 20)

        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)

        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.emb_e.weight.transpose(1, 0))
        x += self.b.expand_as(x)
        pred = torch.sigmoid(x)

        return pred