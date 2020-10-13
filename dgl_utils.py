import dgl
import numpy as np


def comp_deg_norm(g):
    g = g.local_var()
    in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    norm = 1.0 / in_deg
    norm[np.isinf(norm)] = 0
    return norm


def build_graph(num_nodes, num_rels, triples):
    h, r, t = triples.transpose()
    g = dgl.graph(([], []))
    g.add_nodes(num_nodes)
    h, t = np.concatenate((h, t)), np.concatenate((t, h))
    r = np.concatenate((r, r + num_rels))
    triples = sorted(zip(t, h, r))
    t, h, r = np.array(triples).transpose()
    g.add_edges(h, t)
    norm = comp_deg_norm(g)
    print(f"# nodes: {num_nodes}, # edges: {len(h)}")
    return g, r.astype('int64'), norm.astype('int64')


def node_norm_to_edge_norm(g, node_norm):
    g = g.local_var()
    g.ndata['norm'] = node_norm
    g.apply_edges(lambda edges: {'norm': edges.dst['node_norm']})
    return g.edata['norm']