import pandas as pd
import numpy as np
from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate, train_test_split


def read_unregular_csv(csv_path):
    data = []
    with open(csv_path, encoding="utf-8"):
        return data


if __name__ == "__main__":
    data_list = read_unregular_csv("data/try.txt")
    df = pd.read_csv("data/try.txt", header=None)
    print(type(df))
    print(df)
    print(df.shape)
