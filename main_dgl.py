import torch
import torch.nn as nn
import numpy as np
import dgl
from dgl.data.knowledge_graph import load_data
from dgl.nn import GraphConv

DATASET = 'FB15k-237'
emb_size = 200
gc_size = 200


class testModel(torch.nn.Module):
    def __init__(self, num_nodes, num_rels):
        super(testModel, self).__init__()
        self.e_emb = nn.Embedding(num_nodes, emb_size)
        self.e_gc = GraphConv(emb_size, gc_size, norm='both', weight=True, bias=True)
        self.r_emb = nn.Embedding(num_rels, emb_size)


def build_graph(num_nodes, triples):
    h, r, t = triples.transpose()
    g = dgl.graph(([], []))
    g.add_nodes(num_nodes)
    h, t = np.concatenate((h, t)).transpose(), np.concatenate((t, h)).transpose()
    g.add_edges(h, t)
    print(f"# nodes: {num_nodes}, # edges: {len(h)}")
    return g


def triples_process(triples, num_nodes):
    print("triples processing...")
    dataset = {}
    node_list = []
    rel_list = []
    for edge in triples:
        if not edge[0] in dataset.keys():  # 头实体不在dataset里
            label = np.zeros(num_nodes, dtype=np.int)
            label[edge[2]] = 1
            dataset[edge[0]] = {edge[1]: label}
            node_list.append(edge[0])
            rel_list.append(edge[1])
        elif not edge[1] in dataset[edge[0]].keys():  # 头实体在而关系不在
            dataset[edge[0]][edge[1]] = [edge[2]]
            node_list.append(edge[0])
            rel_list.append(edge[1])
        else:  # 尾实体不在
            dataset[edge[0]][edge[1]].append(edge[2])
    if not len(node_list) == len(rel_list):
        print("triples_process error!")
        exit(0)
    else:
        print(f"# triples: {len(node_list)}")
    return dataset, node_list, rel_list

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, triples):
        self.triples = triples
        self.dataset, self.node, self.rel = triples_process(triples)

    def __len__(self):
        return len(self.node)

    def __getitem__(self, item):
        sample = [self.node[item], self.rel[item], self.dataset[self.node[item]][self.rel[item]]]



data = load_data(DATASET)
num_nodes = data.num_nodes
num_rels = data.num_rels
train = data.train
train_g = build_graph(num_nodes, train)
# train_graph, train_r, train_norm = build_graph(num_nodes, num_rels, train)
# train_deg = train_graph.in_degrees(range(train_graph.number_of_nodes())).float().view(-1, 1)
# train_node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
# train_r = torch.from_numpy(train_r)
# train_norm = node_norm_to_edge_norm(train_graph, torch.from_numpy(train_norm).view(-1, 1))
#
# adj_list, degree = get_adj_and_degrees(num_nodes, train)


