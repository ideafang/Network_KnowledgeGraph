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


def comp_deg_norm(g):
    g = g.local_var()
    in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    norm = 1.0 / in_deg
    norm[np.isinf(norm)] = 0
    return norm


def build_graph(num_nodes, num_rels, triples):
    h, r, t = triples.transpose()
    rel = np.concatenate((r, r + num_rels))
    g = dgl.graph(([], []))
    g.add_nodes(num_nodes)
    h, t = np.concatenate((h, t)).transpose(), np.concatenate((t, h)).transpose()
    g.add_edges(h, t)
    norm = comp_deg_norm(g)
    print(f"# nodes: {num_nodes}, # edges: {len(h)}")
    return g, rel.astype('int64'), norm.astype('int64')


def node_norm_to_edge_norm(g, node_norm):
    g = g.local_var()
    g.ndata['norm'] = node_norm
    g.apply_edges(lambda edges: {'norm': edges.dst['norm']})
    return g.edata['norm']


def get_adf_and_degrees(num_nodes, triples):
    adj_list = [[] for _ in range(num_nodes)]
    for i, triplet in enumerate(triples):
        adj_list[triplet[0]].append([i, triplet[2]])
        adj_list[triplet[2]].append([i, triplet[0]])

    degrees = np.array([len(a) for a in adj_list])
    adj_list = [np.array(a) for a in adj_list]
    return adj_list, degrees


def negative_sampling(pos_samples, num_entity, negative_rate):
    batch_size = len(pos_samples)
    num_to_generate = batch_size * negative_rate
    neg_samples = np.tile(pos_samples, (negative_rate, 1))
    labels = np.zeros(batch_size * (negative_rate + 1), dtype=np.float32)
    labels[: batch_size] = 1
    values = np.random.randint(num_entity, size=num_to_generate)
    choices = np.random.uniform(size=num_to_generate)
    subj = choices > 0.5
    obj = choices <= 0.5
    neg_samples[subj, 0] = values[subj]
    neg_samples[obj, 2] = values[obj]
    return np.concatenate((pos_samples, neg_samples)), labels

def generate_sampled_graph_and_labels(triplets, sample_size, split_size, num_rels, adj_list, degrees, negative_rate):
    # perform edge neighbor sampling
    all_edges = np.arange(len(triplets))
    edges = np.random.choice(all_edges, sample_size, replace=False)
    # relabel nodes to have consecutive node ids
    edges = triplets[edges]
    h, r, t = edges.transpose()
    uniq_v, edges = np.unique((h, t), return_inverse=True)
    h, t = np.reshape(edges, (2, -1))
    relabeled_edges = np.stack((h, r, t)).transpose()

    # negative sampling
    samples, labels = negative_rate(relabeled_edges, len(uniq_v), negative_rate)

    split_size = int(sample_size * split_size)
    graph_split_ids = np.random.choice(np.arange(sample_size), size=split_size, replace=False)

    h, r, t = h[graph_split_ids], r[graph_split_ids], t[graph_split_ids]

    # build DGL Graph
    g, rel, norm = build_graph(len(uniq_v), num_rels, (h, r, t))


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


data = load_data(DATASET)
num_nodes = data.num_nodes
num_rels = data.num_rels
train = data.train
valid = data.valid
test = data.test
torch.cuda.set_device(0)
valid = torch.LongTensor(valid)
test = torch.LongTensor(test)
train_g, train_r, train_norm = build_graph(num_nodes, num_rels, train)
train_deg = train_g.in_degrees(range(train_g.number_of_nodes())).float().view(-1, 1)
train_node = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
train_r = torch.from_numpy(train_r)
train_norm = node_norm_to_edge_norm(train_g, torch.from_numpy(train_norm).view(-1, 1))
# build adj list and calculate degrees for sampling
adj_list, degrees = get_adf_and_degrees(num_nodes, train)

