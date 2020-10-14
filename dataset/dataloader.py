import numpy as np
import pickle
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


# 构建一个torch.utils.data.Dataset类型的数据集， data_name为数据集名称， label可选train, valid, test
# e1和rel均为int，label为长度为num_entity的list，目标e2索引的值为1，其余为0（类似one-hot）
class MyDataset(Dataset):
    def __init__(self, data_name, delimiter='\t', type='train', load_from_disk=False):
        if not os.path.exists(f"./dataset/{data_name}"):
            print("Error: data_name not exist!")
            exit(-1)
        self.type = type
        self.data = IdeaDataset(data_name, delimiter=delimiter, load_from_disk=load_from_disk)
        self.e1, self.rel, self.label = self.data.get_dataset(type=type)
        # self.adj_matricx = self.data.get_adj_matricx()
        # self.num_entity = self.data.num_entity
        # self.num_relation = self.data.num_relation

    def __len__(self):
        return len(self.e1)

    def __getitem__(self, item):
        sample = {'entity': self.e1[item], 'relation': self.rel[item], 'label': self.label[item]}
        return sample

    def get_label(self):
        return self.data.filter_node[self.type]


class KGDataset():
    def __init__(self, dataset, delimiter='\t', load_from_disk=False):
        if not os.path.exists(f"./dataset/{dataset}"):
            print('Error: Can not find Dataset directory!')
            exit(1)
        self.delimiter = delimiter
        self.dataset = dataset
        self.num_entity = 0
        self.num_relation = 0
        self.entity2idx = {}
        self.rel2idx = {}
        self.idx2entity = {}
        self.idx2rel = {}
        if load_from_disk:
            if os.path.exists(f"./dataset/{self.dataset}/dict_data"):
                print('loading dict data...')
                self.load_from_disk()
            else:
                print('Error: Can not find dict_data file!')
        # else:
        #     self.data = self.read_data()
        #     print('Processing data...')
        #     self.entity2idx, self.idx2entity = self.token_dict(label='entity')
        #     self.rel2idx, self.idx2rel = self.token_dict(label='relation')
        #     print('saving data to disk...')
        #     self.save_to_disk()

    # 从原始数据集中加载数据
    def read_data(self):
        files = f"./dataset/{self.dataset}/triples.txt"
        data = np.loadtxt(files, dtype=np.str, delimiter=self.delimiter)
        print(f"Total samples number: {data.shape[0]}")
        return data

    # 构建实体和联系的 str <-> int 转换字典
    def list2dict(self, token_list):
        token2idx = dict([(token, num) for num, token in enumerate(token_list)])
        # token2idx['OOV'] = int(0)
        # token2idx[''] = int(1)
        idx2token = dict([(num, token) for num, token in enumerate(token_list)])
        # idx2token[int(0)] = 'OOV'
        # idx2token[int(1)] = ''
        return token2idx, idx2token

    # label可取entity、relation
    def token_dict(self, label):
        e1, rel, e2 = np.hsplit(self.data, 3)
        if label == 'entity':
            entity = np.vstack((e1, e2)).reshape((-1))
            entity = list(set(entity))
            self.num_entity = len(entity)
            token2idx, idx2token = self.list2dict(entity)
        elif label == 'relation':
            rel = rel.reshape((-1))
            rel = list(set(rel))
            self.num_relation = len(rel)
            token2idx, idx2token = self.list2dict(rel)
        else:
            print(f"dataset/dataloader.py function 'token_dict' got a wrong label: {label}")
            print('Expect label is entity or relation')
            exit(1)
        return token2idx, idx2token

    # 根据idx得到原始token， label可选entity或relation
    def get_token(self, idx, label):
        if label == 'relation':
            if idx in self.idx2rel.keys():
                return self.idx2rel[idx]
            else:
                return 'OOV'
        if label == 'entity':
            if idx in self.idx2entity.keys():
                return self.idx2entity[idx]
            else:
                return 'OOV'

    # 根据token得到idx， label可选entity或relation
    def get_idx(self, token, label):
        if label == 'relation':
            if token in self.rel2idx.keys():
                return self.rel2idx[token]
            else:
                return -1
        if label == 'entity':
            if token in self.entity2idx.keys():
                return self.entity2idx[token]
            else:
                return -1

    # 从本地存储中载入字典信息, 同时需要计算entity和relation的数量, 返回bool
    def load_from_disk(self, path='!'):
        if path == '!':
            path = f"./dataset/{self.dataset}/dict_data"
        if not os.path.exists(path):
            print("Error: Can not find dict_data file!")
            return False
        self.entity2idx, self.rel2idx, self.idx2entity, self.idx2rel = pickle.load(open(path, 'rb'))
        self.num_entity = len(self.entity2idx)
        self.num_relation = len(self.rel2idx)
        return True

    # 将字典信息储存到本地存储
    def save_to_disk(self, path='!'):
        if path == '!':
            path = f"./dataset/{self.dataset}/dict_data"
        else:
            path = os.path.join(path, 'dict_data')
        print('saving dict data...')
        pickle.dump([self.entity2idx, self.rel2idx, self.idx2entity, self.idx2rel], open(path, 'wb'))


class OriginDataset(KGDataset):
    def __init__(self, dataset, delimiter='\t', load_from_disk=False):
        super().__init__(dataset, delimiter, load_from_disk=load_from_disk)
        self.triple = {}
        self.filter_node = {}
        if load_from_disk:
            if os.path.exists(f"./dataset/{self.dataset}/filter_node"):
                print('loading filter node...')
                self.filter_node = pickle.load(open(f"./dataset/{self.dataset}/filter_node", 'rb'))
            else:
                print('Error: Can not find filter_node file!')
                exit(1)
            if os.path.exists(f"./dataset/{self.dataset}/triple_data"):
                print('loading triple data...')
                self.triple = pickle.load(open(f"./dataset/{self.dataset}/triple_data", 'rb'))
            else:
                print('Error: Can not find triple_data file!')
                exit(1)
        else:
            self.data = self.read_data()
            print('processing data...')
            print('generating token dict map...')
            self.entity2idx, self.idx2entity = self.token_dict(label='entity')
            self.rel2idx, self.idx2rel = self.token_dict(label='relation')
            print('generating filter node...')
            file = ['train', 'valid', 'test']
            for p in file:
                self.generate_filter_node(label=p)
            self.save_to_disk()

    # 由于数据集具有train, valid, test三个数据文件，因此重载read_data方法
    def read_data(self):
        data = np.zeros(shape=(1, 3))
        files = ['train', 'valid', 'test']
        for p in files:
            tmp = np.loadtxt(f"./dataset/{self.dataset}/{p}.txt", dtype=np.str, delimiter=self.delimiter)
            self.triple[p] = tmp
            data = np.vstack((data, tmp))
        data = data[1:]
        print(f"Total samples number: {data.shape[0]}")
        return data

    # 由于存在filter_node文件，因此需要重载save_to_disk方法
    def save_to_disk(self, path='!'):
        if path == '!':
            path = f"./dataset/{self.dataset}/dict_data"
        else:
            path = os.path.join(path, 'dict_data')
        print('saving dict data...')
        pickle.dump([self.entity2idx, self.rel2idx, self.idx2entity, self.idx2rel], open(path, 'wb'))
        if self.filter_node:
            print("saving filter_node dict...")
            pickle.dump(self.filter_node, open(f"./dataset/{self.dataset}/filter_node", 'wb'))
        if self.triple:
            print("saving triple data...")
            pickle.dump(self.triple, open(f"./dataset/{self.dataset}/triple_data", 'wb'))

    # 将三元组（h, r, t）信息添加到self.filter_node中
    def add_filter_node(self, e1, rel, e2, label):
        if e1 not in self.filter_node[label].keys():
            triple_dict = {rel: []}
            triple_dict[rel].append(e2)
            self.filter_node[label][e1] = triple_dict
        else:
            if rel not in self.filter_node[label][e1].keys():
                self.filter_node[label][e1][rel] = []
                self.filter_node[label][e1][rel].append(e2)
            else:
                if e2 not in self.filter_node[label][e1][rel]:
                    self.filter_node[label][e1][rel].append(e2)

    # 生成训练集中已知节点的链接关系，label可取train, valid, test
    def generate_filter_node(self, label='train'):
        self.filter_node[label] = {}
        for triple in self.triple[label]:
            e1, rel, e2 = self.get_idx(triple[0], label='entity'), self.get_idx(triple[1], label='relation'), \
                          self.get_idx(triple[2], label='entity')
            # 因为知识图谱为无向图，所以三元组的两种链接方向都需要添加
            self.add_filter_node(e1, rel, e2, label)
            self.add_filter_node(e2, rel, e1, label)

    # 通过e1和rel，获取已知链接的节点列表
    def get_filter_node(self, e1, rel, label='train'):
        if e1 not in self.filter_node[label].keys():
            return 'Wrong entity1!'
        elif rel not in self.filter_node[label][e1].keys():
            return 'Wrong relation!'
        else:
            return self.filter_node[label][e1][rel]

    # 生成one-hot类型的标签
    def label_matrix(self, label):
        tensor = np.zeros((len(label), self.num_entity), dtype=np.int)
        for i in range(len(label)):
            idx = label[i]
            tensor[i][idx] = 1
        return tensor

    # 返回转换为idx的数据集, 标签返回为one-hot编码的矩阵, type可选'trian', 'valid', 'test'
    def get_dataset(self, type):
        entity_ = []
        rel_ = []
        for e1 in self.filter_node[type].keys():
            for rel in self.filter_node[type][e1].keys():
                entity_.append(e1)
                rel_.append(rel)
        # entity_ = np.array(entity_, dtype=np.int)
        # rel_ = np.array(rel_, dtype=np.int)
        num_data = len(entity_)
        label_ = np.zeros(shape=(num_data, self.num_entity), dtype=np.int)
        for i in range(num_data):
            e1, rel = entity_[i], rel_[i]
            label_[i][self.filter_node[type][e1][rel]] = self.filter_node[type][e1][rel]
        entity_ = torch.tensor(entity_)
        rel_ = torch.tensor(rel_)
        label_ = torch.tensor(label_)
        return entity_, rel_, label_
        # data_tmp = self.triple[type]
        # data_ = np.array([[self.get_idx(triple[0], label='entity'), self.get_idx(triple[1], label='relation')]
        #                   for triple in data_tmp])
        # label_ = np.array([self.get_idx(triple[2], label='entity') for triple in data_tmp])
        # label_ = self.label_matrix(label_)
        # return data_, label_

    # 获取训练图的邻接矩阵
    def get_adj_matricx(self):
        matricx = np.zeros(shape=(self.num_entity, self.num_entity), dtype=np.int)
        for triple in self.triple['train']:
            e1, rel, e2 = self.get_idx(triple[0], label='entity'), self.get_idx(triple[1], label='relation'), \
                          self.get_idx(triple[2], label='entity')
            if matricx[e1][e2] == 0:
                matricx[e1][e2] = 1
            if matricx[e2][e1] == 0:
                matricx[e2][e1] = 1
        return torch.tensor(matricx)



class IdeaDataset(OriginDataset):
    def __init__(self, dataset, delimiter='\t', load_from_disk=False):
        super().__init__(dataset=dataset, delimiter=delimiter, load_from_disk=load_from_disk)
        if load_from_disk:
            if os.path.exists(f"./dataset/{self.dataset}/feature"):
                print('loading feature data...')
                self.entity_feature, self.rel_feature = pickle.load(open(f"./dataset/{self.dataset}/feature", 'rb'))
            else:
                print('Error: Can not find feature file!')
                exit(1)
        else:
            print('generating feature data...')
            self.rel_dict = self.generate_rel_dict()
            self.entity_feature = self.filter_node['train']
            self.rel_feature = self.generate_relation_feature()
            self.save_feature()

    # 生成entity关联的relation字典
    def generate_rel_dict(self):
        rel_dict = {}
        for triple in self.triple['train']:
            e1, rel, e2 = self.get_idx(triple[0], label='entity'), self.get_idx(triple[1], label='relation'), \
                          self.get_idx(triple[2], label='entity')
            if e1 not in rel_dict.keys():
                rel_dict[e1] = []
            rel_dict[e1].append(rel)
            if e2 not in rel_dict.keys():
                rel_dict[e2] = []
            rel_dict[e2].append(rel)
        return rel_dict

    # 生成entity特征矩阵entity_feature
    def generate_entity_feature(self):
        pass
        # entity_feature = {}
        # for e1 in self.filter_node.keys():
        #     feature = np.zeros(shape=(self.num_relation+2, self.num_entity+2), dtype=np.int)
        #     for rel in self.filter_node[e1].keys():
        #         feature[rel][self.filter_node[e1][rel]] = 1
        #     entity_feature[e1] = feature
        # return entity_feature

    # 生成relation特征矩阵relation_feature
    def generate_relation_feature(self):
        relation_feature = {}
        for triple in self.triple['train']:
            e1, rel, e2 = self.get_idx(triple[0], label='entity'), self.get_idx(triple[1], label='relation'), \
                          self.get_idx(triple[2], label='entity')
            if rel not in relation_feature.keys():
                relation_feature[rel] = {}
                relation_feature[rel][e1] = self.rel_dict[e1]
                relation_feature[rel][e2] = self.rel_dict[e2]
            else:
                if e1 not in relation_feature[rel].keys():
                    relation_feature[rel][e1] = self.rel_dict[e1]
                if e2 not in relation_feature[rel].keys():
                    relation_feature[rel][e2] = self.rel_dict[e2]
        return relation_feature

    # 获取relation的邻接矩阵
    def get_rel_matricx(self):
        rm = np.zeros(shape=(self.num_relation, self.num_relation), dtype=np.int)
        for r1 in self.rel_feature.keys():
            for e in self.rel_feature[r1].keys():
                for r2 in self.entity_feature[e].keys():
                    if rm[r1][r2] == 0:
                        rm[r1][r2] = 1
                    if rm[r2][r1] == 0:
                        rm[r2][r1] = 1
        return torch.tensor(rm)

    # 保存feature data
    def save_feature(self, path='!'):
        if path == '!':
            path = f"./dataset/{self.dataset}/feature"
        else:
            path = os.path.join(path, 'feature')
        print('saving feature data...')
        pickle.dump([self.entity_feature, self.rel_feature], open(path, 'wb'))

    # 按照batch_size生成数据集，label可选train, valid, test
    # 需要大量存储空间，方案舍弃
    def streamBatch(self, label, batch_size=128):
        total_data = []
        for e1 in self.filter_node[label].keys():
            for rel in self.filter_node[label][e1].keys():
                total_data.append([e1, rel])
        batch_number = int(len(total_data) / batch_size) + 1
        print(f"batch number: {batch_number}")
        print(f"sample number: {len(total_data)}")
        index = np.arange(batch_number)
        np.random.shuffle(index)
        for i in range(batch_number):
            start = index[i] * batch_size
            end = min((index[i] + 1) * batch_size - 1, len(total_data))
            batch_length = end - start + 1
            entity_tensor = np.zeros(shape=(batch_length, self.num_relation + 2, self.num_entity + 2), dtype=np.int)
            relation_tensor = np.zeros(shape=(batch_length, self.num_entity + 2, self.num_relation + 2), dtype=np.int)
            label_tensor = np.zeros(shape=(batch_length, self.num_entity + 2), dtype=np.int)
            for j in range(start, end + 1):
                e1, rel = total_data[j]
                for rel_e1 in self.filter_node[label][e1].keys():
                    entity_tensor[j - start][rel_e1][self.filter_node[label][e1][rel_e1]] = 1
                for e1_rel in self.rel_feature[rel].keys():
                    relation_tensor[j - start][e1_rel][self.rel_feature[rel][e1_rel]] = 1
                label_tensor[j - start][self.entity_feature[e1][rel]] = 1
            # 按batch_size保存特征数据
            if not os.path.exists(f"./dataset/{self.dataset}/process"):
                os.mkdir(f"./dataset/{self.dataset}/process")
            print(f"saving batch {i} data...")
            path = f"./dataset/{self.dataset}/process/batch{i}"
            pickle.dump([entity_tensor, relation_tensor, label_tensor], open(path, 'wb'))
