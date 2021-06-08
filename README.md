# Converting data to tensorflow federated data

## Tutorial

In `tff-data-converter` folder, you can see `main.py` file, as an example of how to run and use the library to creae a distributed data. In the following, you can find the details for each part in the `main.py` file.

### 1. Input Type

The input type can be either `image` or `text`. The input path can be pointed to either a `cav` file or a `folder` which contains different folders or files including text or image.
As an example, one of the following can be specified.

```
data_type = 'text'
input_path = './data/csv/test.csv'

```
or 

```
data_type = 'image'
input_path = './data/image'
```

### 2. Loading Data

After specifying the `data type` and `input path`, you need to create a converter object, and load the data.

```
obj = Converter(data_type, input_path)
data, labels = obj.read_data()

```

### 3. Distribute Data

After reading the data, you first need to create a `ToClientData` object, and then distribute the data with arbitrary features:

```
client_obj = ToClientData(data, labels)

distributed_data, distributed_label = client_obj.distribute_data(data_type=data_type,
                                                                 type='niid',
                                                                 train_data_fraction=0.9,
                                                                 selected_feature=3,
                                                                 min_seen_labels=2,
                                                                 max_seen_labels=4)

```

#### Different options to select:

- **type**: `iid` to sample in an i.i.d. manner, or `niid` to sample in a non-i.i.d. manner <sup id="a1">[1](#f1)</sup>, default is `niid`
- **data_type**: type of data, default is `csv`
- **selected_feature**: can be any column in `csv` file, or any label in `folder` format, default is `label` column (last column, last feature)
- **number_of_clients**: number of users; expressed as the total number of users <sup id="a1">[2](#f2)</sup>
- **min_user_number**:  minimum number of samples per user, default value is `1` 
- **max_user_number**:  maximum number of samples per user, default value is `10`
- **min_seen_labels**:  minimum number of labels seen per user, default value is `1` 
- **max_seen_labels**:  maximum number of labels seen per user, default value is `10`
- **train_data_fraction**: fraction of data in training set, written as a decimal; default is `0.9`
- **random_sampling_seed**: seed to be used before random sampling of data, default is `4`
- **random_split_seed**: seed to be used before random split of data, default is `1`
- **split_type**: split type that can be `sample` or `user`, default is `sample`

### 4. Split Data to train/test samples

Split the data can be done by two different types:
      
   1.   Split using `user` to partition users into train-test groups, 
   2.   Split using `sample` to partition each user's samples into train-test groups
      
This can be done as follow:

```
x_train, x_test, y_train, y_test = client_obj.split_data(distributed_data, 
                                                         distributed_label, 
                                                         type='sample')

```

### 4. Final Step

After having the data splitted and ready, it can be used to train any model, or be saved in a file:

```
with open(os.path.join(output_path, 'x_train.txt'), "w") as file1:
    file1.write("\n".join(str(item) for item in x_train))

with open(os.path.join(output_path, 'x_test.txt'), "w") as file1:
    file1.write("\n".join(str(item) for item in x_test))

with open(os.path.join(output_path, 'y_train.txt'), "w") as file1:
    file1.write("\n".join(str(item) for item in y_train))

with open(os.path.join(output_path, 'y_test.txt'), "w") as file1:
    file1.write("\n".join(str(item) for item in y_test))

```

# Footnotes
---

- <b id="f1">1</b>
  * In the i.i.d. sampling scenario, each datapoint is equally likely to be sampled. Thus, all users have the same underlying distribution of data.
  * In the non-i.i.d. sampling scenario, the underlying distribution of data for each user is consistent with the raw data. Since we assume that data distributions vary between user in the raw data, we refer to this sampling process as non-i.i.d.

- <b id="f2">2</b>
  * Both `iid` and `niid` can be given number of clients. However, for `iid`, the default is a random number between `2` and `input data size`.



