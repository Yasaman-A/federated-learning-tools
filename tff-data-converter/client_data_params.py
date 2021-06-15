import random
from random import randrange
import time
from sklearn.model_selection import train_test_split
import numpy as np


class DistributeData:
    def __init__(self, dataset):
        pass

    def __shuffle(self, data, label):
        pass

    def _iid_no_clint(self):
        pass

    def _iid_clint(self, number_of_clients):
        pass

    def _iid(self, **kwargs):
        pass

    def _niid(self, **kwargs):
        pass

    def distribute_data(self, **kwargs):
        pass

    def __select_feature_image_no_client(self, min_user_number, max_user_number, min_seen_labels, max_seen_labels):
        pass

    def __select_feature_image_client(self, min_user_number, max_user_number,
                                      number_of_clients, min_seen_labels, max_seen_labels):
        pass
    def __select_feature_csv_no_client(self, feature_column,
                                       min_seen_labels, max_seen_labels,
                                       min_user_number, max_user_number):
        pass

    def __select_feature_csv_client(self, feature_column,
                                    min_seen_labels, max_seen_labels,
                                    min_user_number, max_user_number,
                                    number_of_clients):
        pass
    def split_data(self, x, y, **kwargs):
        pass

    def _user_split(self, train_data_fraction):
        pass

    def _sample_split(self, x, y, train_data_fraction):
        pass
