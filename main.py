from dataset.dataloader import MyDataset
from model import IdeaModel, SACN, KerasModel, testModel
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


# 判断label中的n个entity是否位于pred中值最大的n位
def num_true(batch_pred, label_dict, e, r):
    num = 0
    for i, pred in enumerate(batch_pred):
        e1 = e[i].item()
        rel = r[i].item()
        label = label_dict[e1][rel]
        acc_flag = True
        for _ in range(len(label)):
            pred_idx = pred.argmax().item()
            if pred_idx in label:
                pred[pred_idx] = 0
            else:
                acc_flag = False
        # print(i, acc_flag)
        if acc_flag:
            num += 1
    return num


DATASET = 'FB15k-237'
epochs = 20
lr = 0.001

dataset = MyDataset(DATASET, type='train', load_from_disk=True)
# num_true方法需要使用label_dict
label_dict = dataset.get_label()
train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
print('getting adj_matricx...')
adj_matricx = dataset.data.get_adj_matricx().float().cuda()
# print('getting rel_matricx...')
# rel_matricx = dataset.data.get_rel_matricx().float().cuda()
num_entity, num_relation = dataset.data.num_entity, dataset.data.num_relation

X_e = torch.LongTensor([i for i in range(num_entity)]).cuda()
X_r = torch.LongTensor([i for i in range(num_relation)]).cuda()

model = testModel(num_entity, num_relation).cuda()

# model.init()
total_param_size = []
params = [value.numel() for value in model.parameters()]
print(params)
print(np.sum(params))

opt = torch.optim.Adam(model.parameters(), lr=lr)
for epoch in range(epochs):
    model.train()
    total_num, true_num, epoch_loss = 0, 0, 0.0
    for i, sample in enumerate(train_loader):
        e = sample['entity'].cuda()
        r = sample['relation'].cuda()
        label = sample['label'].float().cuda()
#         pred = model.forward(e, r, X_e, adj_matricx)
        pred = model.forward(e, r, X_e, X_r, adj_matricx)
        loss = model.loss(pred, label)
        loss.backward()
        opt.step()
        true_num += num_true(pred.cpu(), label_dict, e.cpu(), r.cpu())
        total_num += label.shape[0]
        epoch_loss += loss
    train_acc = float(true_num) / float(total_num)
    print(f"epoch: {epoch+1}, train_loss: {epoch_loss}, train_acc: {train_acc}")

