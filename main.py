from dataset.dataloader import MyDataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

DATASET = 'FB15k-237'

dataset = MyDataset(DATASET, label='train', load_from_disk=True)
train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
adj_matricx = dataset.data.get_adj_matricx()
num_entity, num_relation = dataset.data.num_entity, dataset.data.num_relation

for i, sample in enumerate(train_loader):
    e = F.one_hot(sample['entity'])
    r = F.one_hot(sample['relation'])
    print(e)



