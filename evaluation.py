import torch
from torch.utils.data import DataLoader

batch_size = 128

def evalutaion(model, dataset, g, num_nodes, filter_node):
    '''
    model - cuda()
    dataset - cpu() # MyDataset
    g - cuda() # dgl graph
    num_nodes = num_entity
    filter_node # a flush node dict
    '''
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    label = dataset.get_label()
    X = torch.LongTensor([i for i in range(num_nodes)]).cuda()
    hits1, hits3, hits10, total = 0, 0, 0, 0
    with torch.no_grad():
        for batch_data in dataloader:
            e = batch_data['entity'].cuda()
            r = batch_data['relation'].cuda()
            batch_pred = model.forward(e, r, X, g).cpu()
            e, r = e.cpu(), r.cpu()
            for i, pred in enumerate(batch_pred):
                # flush all known cases
                e1, rel, pred = e[i].item(), r[i].item(), pred.numpy()
                if e1 in filter_node.keys():
                    if rel in filter_node[e1].keys():
                        pred[filter_node[e1][rel]] = 0.0
                # print(f"e: {e1}, r: {rel}")
                # create pred dict
                pred_dict = dict([(idx, pred_num) for idx, pred_num in enumerate(pred)])
                pred_dict = sorted(pred_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
                idx_list_3 = [idx for (idx, _) in pred_dict[:3]]
                idx_list_10 = [idx for (idx, _) in pred_dict[:10]]
                for goal in label[e1][rel]:
                    if goal == pred_dict[0][0]:
                        hits1 += 1
                        hits3 += 1
                        hits10 += 1
                    elif goal in idx_list_3:
                        hits3 += 1
                        hits10 += 1
                    elif goal in idx_list_10:
                        hits10 += 1
                    total += 1

    print(f"hits1: {float(hits1)/float(total)}, hits3: {float(hits3)/float(total)}, hit10: {float(hits10)/float(total)}")
    exit(0)




