import os
import pickle
from converter import Converter
from select_params import ToClientData

# data_type = 'text'  # or image
# input_path = './data/csv/test.csv'  # accepts either folder or csv file

data_type = 'image' # or image
input_path = './data/image' # accepts either folder or csv file

output_path = './output/'
if not os.path.exists(output_path):
    os.makedirs(output_path)

obj = Converter(data_type, input_path)

data, labels = obj.read_data()

client_obj = ToClientData(data, labels)

distributed_data, distributed_label = client_obj.distribute_data(data_type=data_type,
                                                                 type='niid',
                                                                 train_data_fraction=0.9,
                                                                 selected_feature=3,
                                                                 min_seen_labels=2,
                                                                 max_seen_labels=4,)

x_train, x_test, y_train, y_test = client_obj.split_data(distributed_data, distributed_label, type='sample')

with open(os.path.join(output_path, 'x_train.txt'), "w") as file1:
    file1.write("\n".join(str(item) for item in x_train))

with open(os.path.join(output_path, 'x_test.txt'), "w") as file1:
    file1.write("\n".join(str(item) for item in x_test))

with open(os.path.join(output_path, 'y_train.txt'), "w") as file1:
    file1.write("\n".join(str(item) for item in y_train))

with open(os.path.join(output_path, 'y_test.txt'), "w") as file1:
    file1.write("\n".join(str(item) for item in y_test))
