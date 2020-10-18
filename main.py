from dataset.dataloader import MyDataset
from model import IdeaModel, SACN, KerasModel, ConvE, testModel, ConvTransE
import numpy as np
import torch
from torch.utils.data import DataLoader
from evaluation import evalutaion, evaluation_gpu


# 判断label中的n个entity是否位于pred中值最大的n位
def num_true(batch_pred, label_list):
    num = 0
    for i, pred in enumerate(batch_pred):
        length = label_list[i][0]
        label = label_list[i][1:length+1]
        acc_flag = True
        for _ in range(length):
            pred_idx = pred.argmax().item()
            if pred_idx in label:
                pred[pred_idx] = 0
            else:
                acc_flag = False
        # print(i, acc_flag)
        if acc_flag:
            num += 1
    return num


def num_true1(batch_pred, label):
    num = 0
    for i, pred in enumerate(batch_pred):
        pred_idx = pred.argmax().item()
        if label[i][pred_idx] == 1.0:
            num += 1
    return num


DATASET = 'FB15k-237'
epochs = 100
lr = 0.003


dataset = MyDataset(DATASET, type='train', load_from_disk=True)
valid = MyDataset(DATASET, type='valid', load_from_disk=True)
test = MyDataset(DATASET, type='test', load_from_disk=True)
train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
print('getting dgl graph...')
g = dataset.data.get_dgl_graph().to(0)
# print('getting adj_matricx...')
# adj_matricx = dataset.data.get_adj_matricx().float().cuda()
# print('getting rel_matricx...')
# rel_matricx = dataset.data.get_rel_matricx().float().cuda()
num_entity, num_relation = dataset.data.num_entity, dataset.data.num_relation

X_e = torch.LongTensor([i for i in range(num_entity)]).cuda()

model = ConvTransE(num_entity, num_relation).cuda()

model.init()
total_param_size = []
params = [value.numel() for value in model.parameters()]
print(params)
print(np.sum(params))

opt = torch.optim.Adam(model.parameters(), lr=lr)
for epoch in range(epochs):
    model.train()
    result = open('result.txt', 'a+')
    print(f"# Epoch: {epoch+1}")
    result.write(f"# Epoch: {epoch+1}\ntrain: ")
    total_num, true_num, epoch_loss = 0, 0, 0.0
    for i, sample in enumerate(train_loader):
        opt.zero_grad()
        e = sample['entity'].cuda()
        r = sample['relation'].cuda()
        label = sample['label'].float().cuda()
        filter_node = sample['filter'].cuda()
        # label smoothing
        # label = (0.9 * label) + (1.0 / label.size(1))
        pred = model.forward(e, r, X_e, g)
        loss = model.loss(pred, label)
        loss.backward()
        opt.step()
        true_num += num_true(pred, filter_node)
        total_num += label.shape[0]
        epoch_loss += loss
    train_acc = float(true_num) / float(total_num)
    print(f"train_loss: {round(float(epoch_loss), 3)}, train_acc: {round(train_acc, 4)}, num_true: {true_num}")
    result.write(f"train_loss: {round(float(epoch_loss), 3)}, train_acc: {round(train_acc, 4)}, num_true: {true_num}\n")
    result.close()

    model.eval()
    with open('result.txt', 'a+') as f:
        f.write('valid: ')
    evaluation_gpu(model, valid, g, num_entity)
    if (epoch+1) % 5 == 0:
        with open('result.txt', 'a+') as f:
            f.write('test: ')
        evaluation_gpu(model, test, g, num_entity)
    # evalutaion(model, valid, g, num_entity, filter_node)
    # with torch.no_grad():
    #     total_num, true_num = 0, 0
    #     for sample in valid_loader:
    #         e = sample['entity'].cuda()
    #         r = sample['relation'].cuda()
    #         label = sample['label'].float().cuda()
    #         label_list = sample['label_list'].cuda()
    #         pred = model.forward(e, r, X_e, g)
    #         #             pred = model.forward(e, r, X_e, adj_matricx)
    #         true_num += num_true1(pred, label)
    #         total_num += label.shape[0]
    #     valid_acc = float(true_num) / float(total_num)
    #     print(f"epoch: {epoch + 1}, valid_acc: {valid_acc}, true_num: {true_num}")
