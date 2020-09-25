import pickle

path = 'filter_node'
filter_node = pickle.load(open(path, 'rb'))
print(filter_node)