import os
import pickle
from converter import Converter
from functools import reduce
from client_data_params import DistributeData
from select_params import ToClientData
import tensorflow_federated as tff
import collections
import tensorflow as tf

# data_type = 'text'  # or image
# input_path = './data/csv/test.csv'  # accepts either folder or csv file

data_type = 'image'  # or image
input_path = './data/image'  # accepts either folder or csv file

output_path = './output/'
if not os.path.exists(output_path):
    os.makedirs(output_path)

obj = Converter(data_type, input_path)

data, labels = obj.read_data()

dataset = obj.convert_to_client_data(data, labels, number_of_clients=10)

NUM_CLIENTS = 10
NUM_EPOCHS = 5
BATCH_SIZE = 20
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10

def preprocess(dataset):
    def batch_format_fn(element):
        """Flatten a batch `pixels` and return the features as an `OrderedDict`."""
        return collections.OrderedDict(
            x=tf.reshape(data, [-1, reduce(lambda x, y: x*y, data.shape)]),
            y=tf.reshape(labels, [-1, 1]))

    return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER).batch(
        BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)


def make_federated_data(client_data, client_ids):
    return [
        preprocess(client_data.create_tf_dataset_for_client(x))
        for x in client_ids
    ]


sample_clients = dataset.client_ids[0:NUM_CLIENTS]

federated_train_data = make_federated_data(dataset, sample_clients)

print('Number of client datasets: {l}'.format(l=len(federated_train_data)))
print('First dataset: {d}'.format(d=federated_train_data[1]))

#
# # how to reach the client id data and labels
# print(federated_train_data.client_ids[0])
# example_dataset = federated_train_data.create_tf_dataset_for_client(
#     federated_train_data.client_ids[0])
#
# example_element = next(iter(example_dataset))
# print(example_element['label'].numpy())
# print(example_element['data'].numpy())




# client_obj = ToClientData(data, labels)
#
# distributed_data, distributed_label = client_obj.distribute_data(data_type=data_type,
#                                                                  type='niid',
#                                                                  selected_feature=3,
#                                                                  min_seen_labels=2,
#                                                                  max_seen_labels=4,
#                                                                  number_of_clients = 10)
#
# x_train, x_test, y_train, y_test = client_obj.split_data(distributed_data, distributed_label, type='sample',
#                                                          train_data_fraction=0.9)
#
# print(len(x_train))
# print(len(x_test))
# print(len(y_train))
# print(len(y_test))

# train_dataset = tff.simulation.datasets.TestClientData(x_train) # for newer versions

# client_train_dataset = collections.OrderedDict()
# for i in range(1,10):
#     client_name = "client_" + str(i)
#     data = collections.OrderedDict((('label', y_train ), ('pixels', x_train)))
#     client_train_dataset[client_name] = data
#
# train_dataset = tff.simulation.datasets.TestClientData(client_train_dataset) # for newer versions

# with open(os.path.join(output_path, 'x_train.txt'), "w") as file1:
#     file1.write("\n".join(str(item) for item in x_train))
#
# with open(os.path.join(output_path, 'x_test.txt'), "w") as file1:
#     file1.write("\n".join(str(item) for item in x_test))
#
# with open(os.path.join(output_path, 'y_train.txt'), "w") as file1:
#     file1.write("\n".join(str(item) for item in y_train))
#
# with open(os.path.join(output_path, 'y_test.txt'), "w") as file1:
#     file1.write("\n".join(str(item) for item in y_test))
