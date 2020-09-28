from dataset.dataloader import OriginDataset
# from model import ConvE
# from torch.utils.data import TensorDataset, DataLoader
# import torch
import numpy as np
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPool1D, Dense, LSTM, Flatten, Dropout, Input, BatchNormalization, Activation


DATASET = 'FB15k-237'

dataset = OriginDataset(DATASET, load_from_disk=True)
# print(dataset.num_relation)
# print(dataset.num_entity)
# token = dataset.get_token(200, label='entity')
# idx = dataset.get_idx(token, label='entity')
# print(token)
# print(idx)
# print(type(idx))
# print('generating filter node...')
# dataset.generate_filter_node()
# dataset.save_to_disk()
# e1 = dataset.get_idx('/m/027rn', label='entity')
# rel = dataset.get_idx('/location/country/form_of_government', label='relation')
# print(dataset.get_filter_node(e1, rel))


# test_data, test_label = dataset.get_dataset(type='test')
# test_data = torch.from_numpy(test_data).float()
# test_label = torch.from_numpy(test_label).float()
# test_dataset = TensorDataset(test_data, test_label)
# test_loader = DataLoader(dataset=test_dataset, batch_size=10, shuffle=True, num_workers=0)
#
# X = torch.LongTensor([i for i in range(dataset.num_entity)])
#
# model = ConvE(dataset.num_entity, dataset.num_relation)
#
# model.cuda()
# X = X.cuda()
#
# model.init()
#
# total_param_size = []
# params = [value.numel() for value in model.parameters()]
# print(params)
# print(np.sum(params))
# opt = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=0.00)
#
# for epoch in range(5):
#     model.train()
#     for step, (batch_x, batch_y) in enumerate(test_loader):
#         opt.zero_grad()
#         e1 = batch_x[0].cuda()
#         rel = batch_x[1].cuda()
#         e2 = batch_y.float().cuda()

def my_model(num_entitys):
    input_tensor = Input(shape=(num_entitys, 1))
    x = Conv1D(filters=64, kernel_size=5, padding='valid')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x= MaxPool1D(pool_size=3, padding='valid')(x)
    x = Dropout(rate=0.2)(x)
    x = Conv1D(filters=128, kernel_size=3, padding='valid')(x)
    x = Conv1D(filters=128, kernel_size=3, padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool1D(pool_size=2, padding='valid')(x)
    x = Dropout(rate=0.2)(x)
    x = LSTM(units=128, return_sequences=True)(x)
    x = Dropout(rate=0.3)(x)
    x = Flatten()(x)
    x = Dense(num_entitys, activation='softmax')(x)
    model = Model(input_tensor, x)
    return model


train_x, train_y = dataset.get_dataset(type='test')
valid_x, valid_y = dataset.get_dataset(type='valid')

model = my_model(dataset.num_entity)
model.summary()
model.get_config()

model.compile(loss='categorical_crossentropy',
              optimizer='nadam',
              metrics=['accuracy'])
hist = model.fit(train_x, train_y, batch_size=64, epochs=20, validation_data=(valid_x, valid_y))