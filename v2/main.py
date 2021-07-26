import os
from converter import convert_to_client_data
from reader import Reader

from datetime import datetime
start_time = datetime.now()

# data_type = 'csv'
# input_path = '/Users/Vesal/Desktop/federated-learning-tools/v2/data/csv/fake_job_postings_newLabel.csv'  # accepts only a csv file

# data_type = 'text'
# input_path = '/Users/Vesal/Desktop/federated-learning-tools/v2/data/text/topics_sample'  # accepts either folder or csv file

data_type = 'image'
input_path = '/Users/Vesal/Desktop/federated-learning-tools/v2/data/image/fish_sample'  # accepts folder only

# Read Data
obj = Reader(data_type, input_path)
data, labels = obj.read_data()

# Distribute Data and return clients
NUM_CLIENTS = 10
SELECTED_FEATURE = 17
dataset = convert_to_client_data(data, labels,  data_type,
                                                dist_type = 'niid',
                                                number_of_clients = NUM_CLIENTS,
                                                selected_feature=SELECTED_FEATURE,
                                                min_seen_labels=2,
                                                max_seen_labels=4
                                                )

end_time = datetime.now()

print("Finished loading client data! Total run time:", (end_time - start_time))