import os
from converter import convert_to_client_data
from reader import Reader

# data_type = 'csv'
# input_path = './data/csv/fake_job_postings.csv'  # accepts either folder or csv file

data_type = 'text'
input_path = './data/text/topics_sample'  # accepts either folder or csv file

# data_type = 'image'
# input_path = './data/image/fish_sample'  # accepts either folder or csv file

# Read Data
obj = Reader(data_type, input_path)
data, labels = obj.read_data()

# Distribute Data and return clients
dataset = convert_to_client_data(data, labels,  data_type,
                                                type = 'niid',
                                                number_of_clients = 20,
                                                selected_feature=3,
                                                min_seen_labels=2,
                                                max_seen_labels=4)


print('Finished loading client data!')

print(dataset.client_ids[-1])

# example_dataset = dataset.create_tf_dataset_for_client(dataset.client_ids[-1])
# example_element = next(iter(example_dataset))
# #
# #
# print(example_element['label'].numpy())
# print(example_element['data'].numpy())
#
# print(len(example_element['data'].numpy()))