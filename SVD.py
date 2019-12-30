import pandas as pd
import numpy as np
from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate, train_test_split


if __name__ == "__main__":
    data_path = "data/train.txt"
    data_type = {'user_id': np.int32, 'item_id': np.int32, 'rating': np.float32}
    names = ['user_id', 'item_id', 'rating']
    ratings = pd.read_csv(data_path, dtype=data_type, usecols=range(3), names=names)
    print(type(ratings))
    print(ratings.shape)
