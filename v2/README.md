# Documentation

## 1. Reader

`reader.py` file defines Reader class, which requires `data_type` and `path` as the inputs for its object.
It first checks if type of `data_type` and `path` are correct. If yes, based on the `data_type`, it reads the data and returns the data and labels to be distributed.

## 2. Converter

`converter.py` distributes the data using `distributor.py` file, described in tutorial section here, given its parameters. It returns ClientData data type.

https://github.com/Yasaman-A/federated-learning-tools/tree/library


## 3. Run

In the `main.py`, you can see a sample of running the model with example data in the `data` folder, explained fully in tutorial:

https://github.com/Yasaman-A/federated-learning-tools/tree/library
