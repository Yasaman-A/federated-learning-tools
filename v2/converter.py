import tensorflow_federated as tff
import collections
from distributor import Distribute
import tensorflow as tf


def convert_to_client_data(data, labels, data_type, **kwargs):

    distributor_obj = Distribute(data, labels)

    distributed_data, distributed_label = distributor_obj.distribute_data(data_type=data_type, **kwargs)

    client_train_dataset = collections.OrderedDict()

    for i in range(len(distributed_data)):
        client_name = "client_" + str(i)
        data = collections.OrderedDict((('label', distributed_label[i]), ('data', distributed_data[i])))
        client_train_dataset[client_name] = data

    print(f'Converting data to {len(distributed_data)} client data...')

    train_dataset = tff.simulation.datasets.TestClientData(client_train_dataset)

    return train_dataset
