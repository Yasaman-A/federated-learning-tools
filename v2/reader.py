import tensorflow as tf
import os
import glob
import itertools
import csv
import numpy as np
import pandas as pd
from PIL import Image
import os.path


class Reader:
    def __init__(self, data_type, path):
        self.data_type = data_type.lower()
        self.path = path

    def _path_checker_(self):
        return True if os.path.splitext(self.path)[-1][-3:] == 'csv' or os.path.isdir(self.path) else False

    def _data_type_checker_(self):
        return True if self.data_type in ['image', 'text', 'csv'] else False

    def _read_image_(self) -> object:
        sub_folders = [f.path for f in os.scandir(self.path) if f.is_dir()]

        all_files = []
        if sub_folders:
            all_files = [f for s in sub_folders for f in glob.glob(s + "/*.png")]

        print(f'Total number of image files: {len(sub_folders)} folders and {len(all_files)} image files')

        images = [np.array(Image.open(filename)) for filename in all_files]
        labels = [label.split('/')[-2] for label in all_files]

        return images, labels

    def _read_text_folder_(self):
        sub_folders = [f.path for f in os.scandir(self.path) if f.is_dir()]

        all_files = []
        if sub_folders:
            all_files = [f for s in sub_folders for f in glob.glob(s + "/*.txt")]

        print(f'Total number of text files: {len(sub_folders)} folders and {len(all_files)} text files')

        texts = ["".join(open(filename, "r").readlines()) for filename in all_files]
        labels = [label.split('/')[-2] for label in all_files]

        return texts, labels

    def _read_text_csv_(self, columns):

        dataset = pd.read_csv(self.path, dtype=str)
        dataset.fillna('', inplace=True)

        return dataset[list(dataset.iloc[:,:columns-1])].values.tolist(), list(dataset.iloc[:, columns-1])

    def _read_csv_(self):
        if self._data_type_checker_:
            if self.data_type == 'csv':
                f = open(self.path, 'r')
                reader1, reader2 = itertools.tee(csv.reader(f, delimiter=','))
                columns = len(next(reader1))
                print(f"last csv column is considered as the label. Total number of columns:{columns}")
                return self._read_text_csv_(columns)

            elif self.data_type in ['image', 'text']:
                raise ValueError(
                    f'CSV file cannot contain an image or text file, check the input type and try again.')
            else:
                raise ValueError(
                    f'Given path: {self.path} is not pointing to text file, check the path again.')

        else:
            raise ValueError(f'Data type: "{self.data_type}" is incorrect. It should be "text", "image" or "csv" instead')

    def _read_folder_(self):
        if self._data_type_checker_:
            return self._read_text_folder_() if self.data_type == 'text' else self._read_image_()
        else:
            raise ValueError(f'Data type: "{self.data_type}" is incorrect. It should be "text","image", or "csv" instead')

    def read_data(self):
        print('Reading data...')
        if self._path_checker_:
            return self._read_folder_() if os.path.isdir(self.path) else self._read_csv_()
        else:
            raise ValueError(f'Given path: {self.path} is not a csv file or a folder, check the path and try again.')
