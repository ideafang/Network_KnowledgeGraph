from dataset.dataloader import KGDataset, OriginDataset

DATASET = 'FB15k-237'

# dataset = KGDataset(dataset=DATASET, load_from_txt=False)
dataset = OriginDataset('FB15k-237', load_from_disk=True)
# print(dataset.num_relation)
# print(dataset.num_entity)
e1 = dataset.get_idx('/m/027rn', label='entity')
rel = dataset.get_idx('/location/country/form_of_government', label='relation')
# token = dataset.get_token(200, label='entity')
# print(token)
# print(dataset.get_idx(token, label='entity'))
# print('generating filter node...')
# dataset.generate_filter_node()
# dataset.save_to_disk()
print(dataset.get_filter_node(e1, rel))