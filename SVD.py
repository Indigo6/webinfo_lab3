import pandas as pd
import numpy as np
import time
import pickle
import argparse

from surprise import SVDpp, SVD, accuracy
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate, train_test_split
from surprise.model_selection import GridSearchCV


def fmt_time(dtime):
    if dtime <= 0:
        return '0:00.000'
    elif dtime < 60:
        return '0:%02d.%03d' % (int(dtime), int(dtime * 1000) % 1000)
    elif dtime < 3600:
        return '%d:%02d.%03d' % (int(dtime / 60), int(dtime) % 60, int(dtime * 1000) % 1000)
    else:
        return '%d:%02d:%02d.%03d' % (int(dtime / 3600), int((dtime % 3600) / 60), int(dtime) % 60,
                                      int(dtime * 1000) % 1000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Surpurise based SVD')
    parser.add_argument('-m', help='Mode: train or predict', required=True, dest='mode', type=str)

    args = parser.parse_args()
    if(args.mode == 'train'):
        if_train = True
    else:
        if_train = False
    if_search = False
    train_data_path = "data/train.txt"
    test_data_path = "data/test.txt"
    pkl_filename = "model.pkl"
    if if_train:
        data_type = {'user_id': np.int32, 'item_id': np.int32, 'rating': np.float32}
        names = ['user_id', 'item_id', 'rating']
        ratings = pd.read_csv(train_data_path, dtype=data_type, usecols=range(3), names=names)
        print(type(ratings))
        print(ratings.shape)

        reader = Reader(rating_scale=(0, 5))
        train_data = Dataset.load_from_df(ratings, reader)
        trainset, validateset = train_test_split(train_data, test_size=.25)
        if if_search:
            # 跑不动，太慢了
            # 指定参数选择范围
            param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005],
                          'reg_all': [0.4, 0.6]}

            gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=5, n_jobs=-1, joblib_verbose=2)

            time_start = time.time()
            gs.fit(train_data)
            elapsed = time.time() - time_start
            print('Elapsed(search best): %s' % (fmt_time(elapsed)))

            results_df = pd.DataFrame.from_dict(gs.cv_results)
            results_df.to_csv("grid_result.csv")
            # 打印最好的均方根误差RMSE
            print(gs.best_score['rmse'])

            # 打印取得最好RMSE的参数集合
            print(gs.best_params['rmse'])

            # 现在可以使用产生最佳RMSE的算法
            model = gs.best_estimator['rmse']
            time_start = time.time()
            model.fit(train_data.build_full_trainset())
            elapsed = time.time() - time_start
            print('Elapsed(best fit): %s' % (fmt_time(elapsed)))
            with open(pkl_filename, 'wb') as file:
                pickle.dump(model, file)

        else:
            epoch = 20
            total_epoch = 210
            done_epoch = 0
            time_start = time.time()
            for i in range(1, epoch+1):
                model = SVD(n_epochs=i, verbose=False)
                model.fit(trainset)
                # predictions_train = model.test(trainset)
                predictions_test = model.test(validateset)
                # train_nmse = accuracy.rmse(predictions_train)
                test_nmse = accuracy.rmse(predictions_test)
                with open("model_e"+str(i)+".pkl", 'wb') as file:
                    pickle.dump(model, file)
                elapsed = time.time() - time_start
                done_epoch += i
                eta = (total_epoch - done_epoch) * elapsed / done_epoch if done_epoch > 0 else 0
                # print('[%d/%d] Elapsed: %s, ETA: %s >> train_nmse:%s, test_nmse:%s' %
                #       (i+1, epoch, fmt_time(elapsed), fmt_time(eta), train_nmse, test_nmse))
                print('[%d/%d] Elapsed: %s, ETA: %s >> test_nmse:%s' %
                      (i, epoch, fmt_time(elapsed), fmt_time(eta), test_nmse))
                # print('Elapsed(train): %s' % (fmt_time(elapsed)))

            # Save to file in the current working directory
            time_start = time.time()
            predictions = model.test(validateset)
            elapsed = time.time() - time_start
            print('Elapsed(predict): %s' % (fmt_time(elapsed)))
            # 然后计算RMSE
            accuracy.rmse(predictions)
            with open(pkl_filename, 'wb') as file:
                pickle.dump(model, file)
    else:
        data_type = {'user_id': np.int32, 'item_id': np.int32}
        names = ['user_id', 'item_id']
       # test_data = pd.read_csv('data/test/test.txt', dtype=data_type, usecols=range(2), names=names)
     #   print(test_data.shape)

        model = pickle.load(open("model_e8.pkl", 'rb'))
        time_start = time.time()
      #  length = test_data.shape[0]
        with open('submit2.txt', mode='w') as f_out:
            with open('data/test/test.txt', mode='r', encoding='utf8') as test_file:
                for line in test_file:
                # print(row['user_id'], row['item_id'])
                    line = line.split(',')
                    predictions = model.predict(np.int32(line[0]),np.int32(line[1]))
                # print(predictions)
                    f_out.write(str(int(round(predictions.est))) + '\n')
                 # elapsed = time.time() - time_start
                # eta = (length - index - 1) * elapsed / (index + 1) if (index + 1) > 0 else 0

     #   print('[%d/%d] Elapsed: %s, ETA: %s ' % (index+1, length, fmt_time(elapsed), fmt_time(eta)))
        # 然后计算RMSE
