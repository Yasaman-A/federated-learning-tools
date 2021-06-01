import os
import pickle
from converter import Converter
from select_params import ToClientData

data_type = 'text' # or image
input_path = './data/csv/test.csv' # accepts either folder or csv file

output_path = './output/'
if not os.path.exists(output_path):
    os.makedirs(output_path)


obj = Converter(data_type, input_path)

data, labels = obj.read_data()

number_of_clients = 10

client_obj = ToClientData(data, labels, number_of_clients)

distributed_data, distributed_label = client_obj.distribute_data(type = 'niid',
                                                                 selected_feature = 3,
                                                                 min_seen_labels= 10,
                                                                 max_seen_labels= 20)

x_train , x_test, y_train, y_test = client_obj.split_data(distributed_data, distributed_label, type = 'user')


with open(os.path.join(output_path, 'x_train.txt'), "w") as file1:
    file1.write("\n".join(str(item) for item in x_train))

with open(os.path.join(output_path, 'x_test.txt'), "w") as file1:
    file1.write("\n".join(str(item) for item in x_test))

with open(os.path.join(output_path, 'y_train.txt'), "w") as file1:
    file1.write("\n".join(str(item) for item in y_train))

with open(os.path.join(output_path, 'y_test.txt'), "w") as file1:
    file1.write("\n".join(str(item) for item in y_test))
