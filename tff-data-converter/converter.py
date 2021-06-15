import tensorflow as tf
import os
import glob
import itertools
import csv
import tensorflow_federated as tff
import collections

class Converter:
    def __init__(self, data_type, path):
        self.data_type = data_type.lower()
        self.path = path

    def _path_checker_(self):
        return True if os.path.splitext(self.path)[-1][-3:] == 'csv' or os.path.isdir(self.path) else False

    def _data_type_checker_(self):
        return True if self.data_type in ['image', 'text'] else False

    def _read_image_(self) -> object:
        img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, rotation_range=20)
        images, labels = next(img_gen.flow_from_directory(self.path))

        return images, labels

    def _read_text_folder_(self):
        sub_folders = [f.path for f in os.scandir(self.path) if f.is_dir()]

        if sub_folders:
            labels = [label.split('/')[-1] for label in sub_folders]
            all_files = [f for s in sub_folders for f in glob.glob(s + "/*.txt")]
        else:
            labels = self.path.split('/')[-1]
            all_files = [f for f in glob.glob(self.path + "/*.txt")]

        dataset = list(tf.data.TextLineDataset(all_files).as_numpy_iterator())

        return dataset, labels

    def _read_text_csv_(self, columns):

        dataset = tf.data.experimental.CsvDataset(
            self.path,
            [tf.string for i in range(0, columns)],
            # exclude_cols = [columns-1]
            select_cols=[i for i in range(0, columns)]
        )

        dataset = list(dataset.as_numpy_iterator())

        return [item[:-1] for item in dataset], [item[-1] for item in dataset]

    def _read_csv_(self):
        if self._data_type_checker_:
            if self.data_type == 'text':
                f = open(self.path, 'r')
                reader1, reader2 = itertools.tee(csv.reader(f, delimiter=','))
                columns = len(next(reader1))
                print(f"last csv column is considered as the label. Total number of columns:{columns}")
                return self._read_text_csv_(columns)
            else:
                raise ValueError(
                    f'csv file cannot contain an image, check the inputs again!')
        else:
            raise ValueError(f'Data type: "{self.data_type}" is incorrect. It should be "text" or "image" instead')

    def _read_folder_(self):
        if self._data_type_checker_:
            return self._read_text_folder_() if self.data_type == 'text' else self._read_image_()
        else:
            raise ValueError(f'Data type: "{self.data_type}" is incorrect. It should be "text" or "image" instead')

    def read_data(self):
        if self._path_checker_:
            return self._read_folder_() if os.path.isdir(self.path) else self._read_csv_()
        else:
            raise ValueError(f'Given path: {self.path} is not a csv file or a folder, check the path and try again.')

    def convert_to_client_data(self, data, labels, number_of_clients):
        client_train_dataset = collections.OrderedDict()
        for i in range(1, number_of_clients):
            client_name = "client_" + str(i)
            data = collections.OrderedDict((('label', labels), ('data', data)))
            client_train_dataset[client_name] = data

        train_dataset = tff.simulation.datasets.TestClientData(client_train_dataset)  # for newer versions

        # # how to reach the client id data and labels
        # print(train_dataset.client_ids[0])
        # example_dataset = train_dataset.create_tf_dataset_for_client(
        #     train_dataset.client_ids[0])
        #
        # example_element = next(iter(example_dataset))
        # print(example_element['label'].numpy())
        # print(example_element['data'].numpy())

        return train_dataset