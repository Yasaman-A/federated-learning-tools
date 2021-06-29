# Documentation

## Installation

To install the required packages, do the following steps:

  1. Go to the directory where requirements.txt is located.
  2. Activate your virtualenv.
  3. Run following command in your shell: 
    ```
    pip install -r requirements.txt
    ```

## 1. Converter

In the `converter.py` file, we defined a class of converter, which requires `data_type` and `path`.
It reads the data with `read_data` function. First check if type of `data_type` and `path` are correct. If yes, based on the `data_type`, it reads the data and returns the data and labels in tensorflow format

## 2. Parameters

In the `select_params.py` file, we defined all the different functions to distribute the data, described in tutorial section here:

https://github.com/Yasaman-A/federated-learning-tools/tree/library

As well as splitting them after checking the required types and inputs.

## 3. Run

In the `main.py`, you can see a sample of running the model with example data in the `data` folder, explained fully in tutorial:

https://github.com/Yasaman-A/federated-learning-tools/tree/library
