import random
from random import randrange
import time
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
from collections import Counter

def partition_list (list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]

class Distribute:
    def __init__(self, data, label):

        self.data = data
        self.label = label
        self.selected_feature = label
        self.type = 'iid'
        self.client_no = 10
        self.data_sample_fraction = 0.1
        self.min_user_number = 10
        self.max_user_number = 20
        self.train_data_fraction = 0.9
        self.random_sampling_seed = 4
        self.random_split_seed = 1
        self.max_seen_labels = 10
        self.min_seen_labels = 1
        self.split_type = 'sample'

    def __shuffle(self, data, label):
        random.Random(self.random_sampling_seed).shuffle(data)
        random.Random(self.random_sampling_seed).shuffle(label)

    def _iid_no_clint(self):
        size = random.randrange(2, len(self.data))
        self.__shuffle(self.data, self.label)

        glist = []
        glabel = []
        group_size = int(len(self.data) / size)
        for i in range(size):
            glist.append(self.data[group_size * i: group_size * (i + 1)])
            glabel.append(self.label[group_size * i: group_size * (i + 1)])

        return glist, glabel

    def _iid_clint(self, number_of_clients):

        self.__shuffle(self.data, self.label)

        glist = []
        glabel = []
        group_size = int(len(self.data) / number_of_clients)

        for i in range(number_of_clients):
            glist.append(self.data[group_size * i: group_size * (i + 1)])
            glabel.append(self.label[group_size * i: group_size * (i + 1)])

        return glist, glabel

    def _iid(self, **kwargs):
        number_of_clients = kwargs.get('number_of_clients')
        if number_of_clients:
            return self._iid_clint(number_of_clients)
        else:
            return self._iid_no_clint()

    def _niid(self, **kwargs):

        selected_feature = kwargs.get('selected_feature', self.selected_feature)
        min_seen_labels = kwargs.get('min_seen_labels', self.min_seen_labels)
        max_seen_labels = kwargs.get('max_seen_labels', self.max_seen_labels)
        min_user_number = kwargs.get('min_user_number', self.min_user_number)
        max_user_number = kwargs.get('max_user_number', self.max_user_number)
        number_of_clients = kwargs.get('number_of_clients')

        data_type = kwargs.get('data_type')

        if data_type == 'image':
            if number_of_clients:
                if number_of_clients > len(self.data):
                    raise ValueError('Total number of data:', len(self.data),
                                     'is less than total number of clients specified:', number_of_clients)
                else:
                    data, label = self.__select_feature_image_client(min_user_number, max_user_number,
                                                                     number_of_clients,
                                                                     min_seen_labels, max_seen_labels)
            else:
                data, label = self.__select_feature_image_no_client(min_user_number, max_user_number,
                                                                    min_seen_labels, max_seen_labels)
        elif data_type == 'text':
            if number_of_clients:
                if number_of_clients > len(self.data):
                    raise ValueError('Total number of data:', len(self.data),
                                     'is less than total number of clients specified:', number_of_clients)
                else:
                    data, label = self.__select_feature_text_client(min_user_number, max_user_number,
                                                                     number_of_clients,
                                                                     min_seen_labels, max_seen_labels)
            else:
                data, label = self.__select_feature_text_no_client(min_user_number, max_user_number,
                                                                    min_seen_labels, max_seen_labels)

        elif data_type == 'csv':
            if number_of_clients:
                if number_of_clients > len(self.data):
                    raise ValueError('Total number of data:', len(self.data),
                                     'is less than total number of clients specified:', number_of_clients)
                else:
                    data, label = self.__select_feature_csv_client(selected_feature,
                                                                   min_seen_labels, max_seen_labels,
                                                                   min_user_number, max_user_number,
                                                                   number_of_clients)
            else:
                data, label = self.__select_feature_csv_no_client(selected_feature,
                                                                  min_seen_labels, max_seen_labels,
                                                                  min_user_number, max_user_number)
        else:
            raise ValueError(
                f'Given data type: "{data_type}" is not correct, choose between options "text" or "image".')

        return data, label

    def distribute_data(self, **kwargs):
        if kwargs.get('dist_type', self.type) == 'iid':
            return self._iid(**kwargs)
        else:
            return self._niid(**kwargs)

    def __select_feature_image_no_client(self, min_user_number, max_user_number, min_seen_labels, max_seen_labels):

        remained_data = self.data
        remained_label = self.label

        unique_features = list(set(self.label))
        max_feature_len = len(unique_features)

        if max_seen_labels < min_seen_labels:
            raise ValueError(
                f'Max seen labels value: ({max_seen_labels}) cannot be less'
                f' than min seen labels: ({min_seen_labels})')

        if min_seen_labels > min(max_seen_labels, max_feature_len):
            raise ValueError(
                f'Total number of unique features: ({max_feature_len}) is more'
                f' than what is set for min seen labels: ({min_seen_labels})')
        else:
            unique_feature_size = random.randint(min_seen_labels, min(max_seen_labels, max_feature_len))

        grouped_data = []
        grouped_data_label = []

        while remained_data:
            random_selected_features = random.sample(unique_features, k=unique_feature_size)

            data_index = [i for i, e in enumerate(remained_label) if e in random_selected_features]

            selected_data = [remained_data[i] for i in data_index]

            if selected_data:

                rng = randrange(min_user_number, max_user_number)
                user_size = len(selected_data) if rng > len(selected_data) else rng

                random_selected_index = random.sample(data_index, k=user_size)

                random_selected_data = [remained_data[ind] for ind in random_selected_index]
                random_selected_labels = [remained_label[ind] for ind in random_selected_index]

                grouped_data.append(random_selected_data)
                grouped_data_label.append(random_selected_labels)

                remained_data = [x for i, x in enumerate(remained_data) if i not in random_selected_index]
                remained_label = [x for i, x in enumerate(remained_label) if i not in random_selected_index]

        return grouped_data, grouped_data_label

    def __select_feature_image_client(self, min_user_number, max_user_number,
                                      number_of_clients, min_seen_labels, max_seen_labels):
        remained_data = self.data
        remained_label = self.label

        unique_features = list(set(self.label))
        max_feature_len = len(unique_features)

        if max_seen_labels < min_seen_labels:
            raise ValueError(
                f'Max seen labels value: ({max_seen_labels}) cannot be less'
                f' than min seen labels: ({min_seen_labels})')

        if min_seen_labels > min(max_seen_labels, max_feature_len):
            raise ValueError(
                f'Given total number of unique features: ({max_feature_len}) is less'
                f' than what is set for min seen labels: ({min_seen_labels})')
        else:
            unique_feature_size = random.randint(min_seen_labels, min(max_seen_labels, max_feature_len))

        selected_data = []

        grouped_data = []
        grouped_data_label = []

        distributed_labels = []

        for i in range(number_of_clients):
          distributed_labels.append(random.sample(unique_features, k=unique_feature_size))

        flatted_distributed_labels = [item for sublist in distributed_labels for item in sublist]

        data_dict={}

        for label, frequency in Counter(flatted_distributed_labels).items():
          datapoints = [i for i, e in enumerate(self.label) if e == label]
          data_dict[label] = partition_list(datapoints, frequency)


        for client in distributed_labels:
          selected_index = []
          selected_data = []
          selected_label = []
          for client_label in client:
            index = random.sample(data_dict[client_label], k=1)

            data_dict[client_label].remove(index[0])
            selected_index.append(index[0])

            for i in range(len(index[0])):
              selected_label.append(client_label)


          flatted_selected_index = [item for sublist in selected_index for item in sublist]
          selected_data = [self.data[i] for i in flatted_selected_index]

          grouped_data_label.append(selected_label)
          grouped_data.append(selected_data)

        return grouped_data, grouped_data_label

    def __select_feature_text_no_client(self, min_user_number, max_user_number, min_seen_labels, max_seen_labels):

        remained_data = self.data
        remained_label = self.label

        unique_features = list(set(self.label))
        max_feature_len = len(unique_features)

        if max_seen_labels < min_seen_labels:
            raise ValueError(
                f'Max seen labels value: ({max_seen_labels}) cannot be less'
                f' than min seen labels: ({min_seen_labels})')

        if min_seen_labels > min(max_seen_labels, max_feature_len):
            raise ValueError(
                f'Total number of unique features: ({max_feature_len}) is more'
                f' than what is set for min seen labels: ({min_seen_labels})')
        else:
            unique_feature_size = random.randint(min_seen_labels, min(max_seen_labels, max_feature_len))

        grouped_data = []
        grouped_data_label = []

        selected_data = []

        while remained_data:

            random_selected_features = random.sample(unique_features, k=unique_feature_size)

            data_index = [i for i, e in enumerate(remained_label) if e in random_selected_features]

            selected_data = [remained_data[i] for i in data_index]

            if selected_data:

              rng = randrange(min_user_number, max_user_number)
              user_size = len(selected_data) if rng > len(selected_data) else rng

              random_selected_index = random.sample(data_index, k=user_size)

              random_selected_data = [remained_data[ind] for ind in random_selected_index]
              random_selected_labels = [remained_label[ind] for ind in random_selected_index]

              grouped_data.append(random_selected_data)
              grouped_data_label.append(random_selected_labels)

              remained_data = [x for i, x in enumerate(remained_data) if i not in random_selected_index]
              remained_label = [x for i, x in enumerate(remained_label) if i not in random_selected_index]

        return grouped_data, grouped_data_label

    def __select_feature_text_client(self, min_user_number, max_user_number,
                                     number_of_clients,
                                     min_seen_labels, max_seen_labels):

        unique_features = list(set(self.label))
        max_feature_len = len(unique_features)

        if max_seen_labels < min_seen_labels:
            raise ValueError(
                f'Max seen labels value: ({max_seen_labels}) cannot be less'
                f' than min seen labels: ({min_seen_labels})')

        if min_seen_labels > min(max_seen_labels, max_feature_len):
            raise ValueError(
                f'Given total number of unique features: ({max_feature_len}) is less'
                f' than what is set for min seen labels: ({min_seen_labels})')
        else:
            unique_feature_size = random.randint(min_seen_labels, min(max_seen_labels, max_feature_len))

        selected_data = []

        grouped_data = []
        grouped_data_label = []

        distributed_labels = []

        for i in range(number_of_clients):
          distributed_labels.append(random.sample(unique_features, k=unique_feature_size))

        flatted_distributed_labels = [item for sublist in distributed_labels for item in sublist]

        data_dict={}

        for label, frequency in Counter(flatted_distributed_labels).items():
          datapoints = [i for i, e in enumerate(self.label) if e == label]
          data_dict[label] = partition_list(datapoints, frequency)


        for client in distributed_labels:
          selected_index = []
          selected_data = []
          selected_label = []
          for client_label in client:
            index = random.sample(data_dict[client_label], k=1)

            data_dict[client_label].remove(index[0])
            selected_index.append(index[0])

            for i in range(len(index[0])):
              selected_label.append(client_label)

          flatted_selected_index = [item for sublist in selected_index for item in sublist]
          selected_data = [self.data[i] for i in flatted_selected_index]

          grouped_data_label.append(selected_label)
          grouped_data.append(selected_data)

        return grouped_data, grouped_data_label

    def __select_feature_csv_no_client(self, feature_column,
                                       min_seen_labels, max_seen_labels,
                                       min_user_number, max_user_number):
        if '' in [item[feature_column] for item in self.data]:
          raise ValueError('Selected feature should not have any null values. Select another feature column')

        unique_features = list(set([item[feature_column] for item in self.data]))
        max_feature_len = len(unique_features)

        if max_seen_labels < min_seen_labels:
            raise ValueError(
                f'Max seen labels value: ({max_seen_labels}) cannot be less'
                f' than min seen labels: ({min_seen_labels})')

        if min_seen_labels > min(max_seen_labels, max_feature_len):
            raise ValueError(
                f'Total number of unique features: ({max_feature_len}) for column ({feature_column}) is more'
                f' than what is set for min seen labels: ({min_seen_labels})')
        else:
            unique_feature_size = random.randint(min_seen_labels, min(max_seen_labels, max_feature_len))

        grouped_data = []
        grouped_data_label = []
        remained_data = self.data
        remained_label = self.label

        while remained_data:
            random_selected_features = random.sample(unique_features, k=unique_feature_size)

            data_index = [i for i, x in enumerate(remained_label) if x[feature_column] in random_selected_features]
            selected_data = [remained_data[i] for i in data_index]

            if selected_data:
                rng = randrange(min_user_number, max_user_number)
                user_size = len(selected_data) if rng > len(selected_data) else rng
                random_selected_index = random.sample(data_index, k=user_size)

                random_selected_data = [remained_data[ind] for ind in random_selected_index]
                random_selected_labels = [remained_label[ind] for ind in random_selected_index]

                grouped_data.append(random_selected_data)
                grouped_data_label.append(random_selected_labels)

                remained_data = [x for i, x in enumerate(remained_data) if i not in random_selected_index]
                remained_label = [x for i, x in enumerate(remained_label) if i not in random_selected_index]

        return grouped_data, grouped_data_label

    def __select_feature_csv_client(self, feature_column,
                                    min_seen_labels, max_seen_labels,
                                    min_user_number, max_user_number,
                                    number_of_clients):

        print(set([item[feature_column] for item in self.data]))

        if '' in [item[feature_column] for item in self.data]:
          raise ValueError('Selected feature should not have any null values. Select another feature column')

        unique_features = list(set([item[feature_column] for item in self.data]))
        max_feature_len = len(unique_features)

        if max_seen_labels < min_seen_labels:
            raise ValueError(
                f'Max seen labels value: ({max_seen_labels}) cannot be less'
                f' than min seen labels: ({min_seen_labels})')

        if min_seen_labels > min(max_seen_labels, max_feature_len):
            raise ValueError(
                f'Total number of unique features: ({max_feature_len}) for column ({feature_column}) is more'
                f' than what is set for min seen labels: ({min_seen_labels})')
        else:
            unique_feature_size = random.randint(min_seen_labels, min(max_seen_labels, max_feature_len))

        selected_data = []

        grouped_data = []
        grouped_data_label = []

        distributed_labels = []

        for i in range(number_of_clients):
          distributed_labels.append(random.sample(unique_features, k=unique_feature_size))

        flatted_distributed_labels = [item for sublist in distributed_labels for item in sublist]

        data_dict={}

        for label, frequency in Counter(flatted_distributed_labels).items():
          datapoints = [i for i, e in enumerate(self.label) if e == label]
          data_dict[label] = partition_list(datapoints, frequency)


        for client in distributed_labels:
          selected_index = []
          selected_data = []
          selected_label = []
          for client_label in client:
            index = random.sample(data_dict[client_label], k=1)

            data_dict[client_label].remove(index[0])
            selected_index.append(index[0])

            for i in range(len(index[0])):
              selected_label.append(client_label)

          flatted_selected_index = [item for sublist in selected_index for item in sublist]
          selected_data = [self.data[i] for i in flatted_selected_index]

          grouped_data_label.append(selected_label)
          grouped_data.append(selected_data)

        return grouped_data, grouped_data_label

        # grouped_data = []
        # grouped_data_label = []

        # random_selected_data = []
        # random_selected_labels = []

        # remained_data = self.data
        # remained_label = self.label

        # g_size = number_of_clients

        # while remained_data and g_size > 0:

        #     random_selected_features = random.sample(unique_features, k=unique_feature_size)

        #     data_index = [i for i, x in enumerate(remained_data) if x[feature_column] in random_selected_features]
        #     selected_data = [remained_data[i] for i in data_index]

        #     if selected_data:

        #         rng = randrange(min_user_number, max_user_number)
        #         user_size = len(selected_data) if rng > len(selected_data) else rng
        #         random_selected_index = random.sample(data_index, k=user_size)

        #         if g_size != 1:

        #             random_selected_data = [remained_data[ind] for ind in random_selected_index]
        #             random_selected_labels = [remained_label[ind] for ind in random_selected_index]

        #         elif g_size == 1:
        #             random_selected_data = remained_data
        #             random_selected_labels = remained_label

        #         grouped_data.append(random_selected_data)
        #         grouped_data_label.append(random_selected_labels)

        #         remained_data = [x for i, x in enumerate(remained_data) if i not in random_selected_index]
        #         remained_label = [x for i, x in enumerate(remained_label) if i not in random_selected_index]

        #         g_size -= 1

        # if g_size != 0:
        #   warnings.warn(f"Total number of clients: {number_of_clients} cannot be reached using random choices made by program. Total number of clients are: {number_of_clients - g_size}")

        # return grouped_data, grouped_data_label

    def split_data(self, x, y, **kwargs):
        train_data_fraction = kwargs.get('train_data_fraction', self.train_data_fraction)
        if kwargs.get('type', self.type) == 'sample':
            return self._sample_split(x, y, train_data_fraction)
        else:
            return self._user_split(train_data_fraction)

    def _user_split(self, train_data_fraction):
        rng_seed = (self.random_split_seed if (self.random_split_seed is not None and self.random_split_seed >= 0)
                    else int(time.time()))
        rng = random.Random(rng_seed)
        # randomly sample from user_files to pick training set users
        num_users = self.client_no
        num_train_users = int(train_data_fraction * num_users)
        indices = [i for i in range(num_users)]
        train_indices = rng.sample(indices, num_train_users)
        train_blist = [False for i in range(num_users)]
        for i in train_indices:
            train_blist[i] = True
        train_user_files = []
        test_user_files = []
        train_labels = []
        test_labels = []

        for i in range(num_users):
            if train_blist[i]:
                train_user_files.append(self.data[i])
                train_labels.append(self.label[i])
            else:
                test_user_files.append(self.data[i])
                test_labels.append(self.label[i])

        return train_user_files, test_user_files, train_labels, test_labels

    def _sample_split(self, x, y, train_data_fraction):
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_data_fraction)
        return x_train, x_test, y_train, y_test
