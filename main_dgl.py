import torch
from dgl.data.knowledge_graph import load_data
from dgl_utils import build_graph, node_norm_to_edge_norm

DATASET = 'FB15k-237'

data = load_data(DATASET)
num_nodes = data.num_nodes
num_rels = data.num_rels
train = data.train
train_graph, train_r, train_norm = build_graph(num_nodes, num_rels, train)
train_deg = train_graph.in_degrees(range(train_graph.number_of_nodes())).float().view(-1, 1)
train_node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
train_r = torch.from_numpy(train_r)
train_norm = node_norm_to_edge_norm(train_graph, torch.from_numpy(train_norm).view(-1, 1))
