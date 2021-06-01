import random
from random import randrange
import time
from sklearn.model_selection import train_test_split


class ToClientData:
    def __init__(self, data, label,
                 number_of_clients=10,
                 is_iid=False,
                 data_sample_fraction=0.1,
                 min_user_number=1,
                 max_user_number=10,
                 train_data_fraction=0.9,
                 random_sampling_seed=4,
                 random_split_seed=1,
                 max_seen_labels=10,
                 min_seen_labels=0,
                 split_type='user'):

        self.data = data
        self.label = label
        self.selected_feature = label
        self.client_no = number_of_clients
        self.type = is_iid
        self.data_sample_fraction = data_sample_fraction
        self.min_user_number = min_user_number
        self.max_user_number = max_user_number
        self.train_data_fraction = train_data_fraction
        self.random_sampling_seed = random_sampling_seed
        self.random_split_seed = random_split_seed
        self.max_seen_labels = max_seen_labels
        self.min_seen_labels = min_seen_labels
        self.split_type = split_type

    def __shuffle(self):
        random.Random(self.random_sampling_seed).shuffle(self.data)
        random.Random(self.random_sampling_seed).shuffle(self.label)

    def _iid(self):
        self.__shuffle()
        '''
        divide list l among g groups
        each group has either int(len(l)/g) or int(len(l)/g)+1 elements
        returns a list of groups
        '''
        num_elems = len(self.data)
        group_size = int(len(self.data) / self.client_no)
        num_big_groups = num_elems - self.client_no * group_size
        num_small_groups = self.client_no - num_big_groups
        glist = []
        glabel = []
        for i in range(num_small_groups):
            glist.append(self.data[group_size * i: group_size * (i + 1)])
            glabel.append(self.label[group_size * i: group_size * (i + 1)])

        bi = group_size * num_small_groups
        group_size += 1
        for i in range(num_big_groups):
            glist.append(self.data[bi + group_size * i:bi + group_size * (i + 1)])
            glabel.append(self.label[bi + group_size * i:bi + group_size * (i + 1)])

        return glist, glabel

    def _niid(self, **kwargs):

        selected_feature = kwargs.get('selected_feature', self.selected_feature)
        min_seen_labels = kwargs.get('min_seen_labels', self.min_seen_labels)
        max_seen_labels = kwargs.get('max_seen_labels', self.max_seen_labels)
        min_user_number = kwargs.get('min_user_number', self.min_user_number)
        max_user_number = kwargs.get('max_user_number', self.max_user_number)

        data, label = self.__select_feature(selected_feature,
                                   min_seen_labels, max_seen_labels,
                                   min_user_number, max_user_number)

        return data, label

    def distribute_data(self, **kwargs):

        if kwargs.get('type', self.type) == 'iid':
            return self._iid()
        else:
            return self._niid(**kwargs)

    def __select_feature(self, feature_column, min_seen_labels, max_seen_labels, min_user_number, max_user_number):

        unique_features = list(set([item[feature_column] for item in self.data]))
        max_feature_len = len(unique_features)

        if min_seen_labels > min(max_seen_labels, max_feature_len):
            raise ValueError(
                f'Total number of unique features: ({max_feature_len}) for column ({feature_column}) is more'
                f' than what is set for min seen labels: ({min_seen_labels})')
        else:
            unique_feature_size = random.randint(min_seen_labels, min(max_seen_labels, max_feature_len))

        remained_data = self.data

        grouped_data = []
        grouped_data_label = []

        while remained_data:
            random_selected_features = random.choices(unique_features, k=unique_feature_size)

            selected_data = [x for x in self.data if x[feature_column] in random_selected_features]

            rng = randrange(min_user_number, max_user_number)
            user_size = len(selected_data) if rng > len(selected_data) else rng

            random_selected_data = random.choices(selected_data, k=user_size)
            random_selected_index = [self.data.index(x) for x in random_selected_data]
            selected_label = [self.label[x] for x in random_selected_index]

            grouped_data.append(random_selected_data)
            grouped_data_label.append(selected_label)

            remained_data = [x for x in remained_data if x not in random_selected_data]

        return grouped_data, grouped_data_label

    def split_data(self,  x, y, **kwargs):

        if kwargs.get('type', self.type) == 'sample':
            return self._sample_split(x,y)
        else:
            return self._user_split()

    def _user_split(self):
        rng_seed = (self.random_split_seed if (self.random_split_seed is not None and self.random_split_seed >= 0)
                    else int(time.time()))
        rng = random.Random(rng_seed)
        # randomly sample from user_files to pick training set users
        num_users = self.client_no
        num_train_users = int(self.train_data_fraction * num_users)
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

    def _sample_split(self, x, y):
        x_train, x_test , y_train, y_test = train_test_split(x,y, test_size=self.train_data_fraction)
        return x_train, x_test , y_train, y_test
