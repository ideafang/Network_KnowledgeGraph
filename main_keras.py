from dataset.dataloader import OriginDataset
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Conv1D, MaxPool1D, Dense, LSTM, Flatten, Dropout, Input, BatchNormalization, Activation, concatenate


DATASET = 'FB15k-237'

dataset = OriginDataset(DATASET, load_from_disk=False)
train_e, train_r, train_y = dataset.get_dataset('train')
valid_e, valid_r, valid_y = dataset.get_dataset('valid')
test_e, test_r, test_y = dataset.get_dataset('test')
print(train_e.shape)
print(train_y.shape)

def MyModel(num_entity, num_rel):
    entity = Input(1)
    x1 = Embedding(input_dim=num_entity, output_dim=200, input_length=num_entity)(entity)  # 1，200
    x1 = Conv1D(filters=64, kernel_size=3, padding='valid', data_format='channels_first')(x1)  # 64, 200
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = LSTM(units=128, return_sequences=True)(x1)  # 64, 128
    x1 = Dropout(rate=0.5)(x1)
    relation = Input(1)
    x2 = Embedding(input_dim=num_rel, output_dim=200, input_length=num_rel)(relation)  # 1, 200
    # x2 = tf.reshape(x2, shape=(x2.shape[0], 200, 1))
    x2 = Conv1D(filters=64, kernel_size=3, padding='valid', data_format='channels_first')(x2)  # 64, 200
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = LSTM(units=128, return_sequences=True)(x2)  # 64, 128
    x2 = Dropout(rate=0.5)(x2)
    x = concatenate([x1, x2], axis=1)  # 128, 128
    x = MaxPool1D(pool_size=2, padding='valid')(x)  # 64, 128
    x = Dropout(rate=0.5)(x)
    x = Conv1D(filters=128, kernel_size=3, padding='valid')(x)  # 64, 128
    x = Conv1D(filters=128, kernel_size=3, padding='valid')(x)  # 64, 128
    x = BatchNormalization()(x)
    x = MaxPool1D(pool_size=2, padding='valid')(x)  # 32, 128
    x = Dropout(rate=0.5)(x)
    x = LSTM(units=64, return_sequences=True)(x)  # 32, 64
    x = Flatten()(x)
    x = Dense(num_entity, activation='sigmoid')(x)
    model = Model([entity, relation], x)
    return model

num_entity = dataset.num_entity+2
num_rel = dataset.num_relation+2
model = MyModel(num_entity, num_rel)
model.summary()
model.get_config()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
hist = model.fit([train_e, train_r], train_y, batch_size=128, epochs=5, validation_data=([valid_e, valid_r], valid_y))

scores = model.evaluate([test_e, test_r], test_y, batch_size=128)
print(f"loss: {scores[0]}\naccuracy: {scores[1]}")

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

# def my_model(num_entitys):
#     input_tensor = Input(shape=(num_entitys, 1))
#     x = Conv1D(filters=64, kernel_size=5, padding='valid')(input_tensor)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x= MaxPool1D(pool_size=3, padding='valid')(x)
#     x = Dropout(rate=0.2)(x)
#     x = Conv1D(filters=128, kernel_size=3, padding='valid')(x)
#     x = Conv1D(filters=128, kernel_size=3, padding='valid')(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = MaxPool1D(pool_size=2, padding='valid')(x)
#     x = Dropout(rate=0.2)(x)
#     x = Conv1D(filters=128, kernel_size=3, padding='valid')(x)
#     x = Conv1D(filters=128, kernel_size=3, padding='valid')(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = MaxPool1D(pool_size=2, padding='valid')(x)
#     x = Dropout(rate=0.2)(x)
#     x = Conv1D(filters=128, kernel_size=3, padding='valid')(x)
#     x = Conv1D(filters=128, kernel_size=3, padding='valid')(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = MaxPool1D(pool_size=2, padding='valid')(x)
#     x = Dropout(rate=0.2)(x)
#     x = Flatten()(x)
#     x = Dense(64)(x)
#     x = Dense(num_entitys, activation='softmax')(x)
#     model = Model(input_tensor, x)
#     return model


# train_x, train_y = dataset.get_dataset(type='train')
# print(train_x.shape)
# valid_x, valid_y = dataset.get_dataset(type='valid')
#
# model = my_model(dataset.num_entity)
# model.summary()
# model.get_config()
#
# model.compile(loss='categorical_crossentropy',
#               optimizer='nadam',
#               metrics=['accuracy'])
# hist = model.fit(train_x, train_y, batch_size=64, epochs=20, validation_data=(valid_x, valid_y))