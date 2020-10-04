# Network_KnowledgeGraph
an network area knowledge graph loader

## 数据集载入: dataloader

### 目标

- 汇总原始数据集中的实体和关系数量。
- 生成（实体、关系）和（数字）之间的互相映射。
- 生成图节点与边的邻接矩阵
- 构建训练，测试数据集
- 提供batch_size迭代数据集功能
- 构建模型训练和验证（ToDo）过程

### 功能

- 构建KGDataset基类，集成加载数据`read_data`，生成映射关系`token_dict`，保存映射关系`save_to_disk`，从映射关系处加载`load_from_disk`，映射查询`get_token, get_idx`等方法。
- 构建Origin_Dataset类，继承KGDataset类，重写加载数据`read_data`方法，继承KGDataset的所有功能。
- 在Origin_Dataset类中，构建`generate_filter_node`方法，生成已知节点和链接对应的节点列表
- 构建`get_filter_node`方法，提供通过e1和rel查询已知节点列表功能
- 重载pytorch的Dataset方法，将数据集转为pytorch.dataset类型，可以在后续调用pytorch的Dataloader方法，实现batch小样本训练
- 在model.py中搭建module
- 在main.py构建accuracy评价方法

### ToDo

- [x] 构建可以指定batch_size的迭代器
- [x] 构建accuracy评价方法
- [ ] 构建验证方法
- [ ] 构建测试方法

