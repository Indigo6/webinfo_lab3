import pandas as pd
import numpy as np
from surprise import SVDpp, SVD, accuracy
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate, train_test_split
import pickle


if __name__ == "__main__":
    data_path = "data/train.txt"
    # reader = Reader(line_format='user item rating timestamp', sep=',')
    # # 加载数据
    # data = Dataset.load_from_file(data_path, reader=reader)
    data_type = {'user_id': np.int32, 'item_id': np.int32, 'rating': np.float32}
    names = ['user_id', 'item_id', 'rating']
    ratings = pd.read_csv(data_path, dtype=data_type, usecols=range(3), names=names)
    print(type(ratings))
    print(ratings.shape)

    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(ratings, reader)
    trainset, testset = train_test_split(data, test_size=.25)
    # model = SVDpp(n_epochs=1, verbose=True)
    model = SVD(n_epochs=1, verbose=True)
    model.fit(trainset)
    predictions = model.test(testset)
    # 然后计算RMSE
    accuracy.rmse(predictions)
    # trainset = data.build_full_trainset()
    # model.train(trainset)

    # Save to file in the current working directory
    pkl_filename = "pickle_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)
