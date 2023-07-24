import os
import pickle
import pandas as pd


class my_io:
    def __init__(self, dataset_file_path: str = None):
        if dataset_file_path is not None:
            self.dataset_file_path = dataset_file_path
        self.data = []

    def read_dataset(self, dataset_file_path: str = None):
        if dataset_file_path is not None:
            self.dataset_file_path = dataset_file_path
        else:
            if self.dataset_file_path is None:
                print("Error: dataset path is not defined!")
                return

        if not os.path.exists(self.dataset_file_path):
            print("Error: dataset file does not exist!")
            return

        # Read the dataset file with pandas and convert it to numpy array
        data = pd.read_csv(
            self.dataset_file_path, sep="\t", header=None
        ).values

        # Remove the first column (id) and the first row (header)
        return data[1:, 1:]

    def save_data(self, data, filename):
        with open(filename, "wb") as f:
            pickle.dump(data, f)

    def load_data(self, filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
        return data
