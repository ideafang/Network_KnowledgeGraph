import torch
from model import KerasModel
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset.dataloader import IdeaDataset, MyDataset, OriginDataset

# x = torch.LongTensor([i for i in range(10)])
# print(x)


# 判断label中的n个entity是否位于pred中值最大的n位
def accuarcy(batch_pred, label_list):
    total_num = len(label_list)
    true_num = 0
    for i, pred in enumerate(batch_pred):
        label = label_list[i]
        acc_flag = True
        for _ in range(len(label)):
            pred_idx = pred.argmax().item()
            if pred_idx in label:
                pred[pred_idx] = 0
            else:
                acc_flag = False
        # print(i, acc_flag)
        if acc_flag:
            true_num += 1
    acc = float(true_num) / float(total_num)
    return acc

# 测试accuracy方法
def check_accuracy():
    batch_pred = [[0, 1, 1, 0],
                  [1, 0, 0, 0],
                  [0, 0, 0, 1]]
    label_list = [[1, 2], [0], [0]]
    acc = accuarcy(torch.tensor(batch_pred), label_list)
    print(acc)

if __name__ == "__main__":
    # test_label = MyDataset('FB15k-237', type='train', load_from_disk=True)
    # label_dict = test_label.get_label()
    # print(label_dict)

    # check_accuracy()

    # test = OriginDataset('FB15k-237', load_from_disk=True)
    # g = test.get_dgl_graph()

    d = {'1': 0.1, '2': 0.2, '3': 0.3}
    l = sorted(d.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    print(type(l[0][0]))