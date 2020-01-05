# 基于 surprise 开源包的电影推荐系统

## 依赖安装

+ pandas

+ surprise

  + 推荐：在 conda 环境中使用 `conda install -c conda-forge scikit-surprise` 命令安装

  + 不推荐：直接使用pip 安装

    ```python
    pip install numpy
    pip install scikit-surprise
    # 需要 Visual C++ 环境现编译 whl
    ```

## 数据

+ 新建 data 目录，把数据放到 data/ 路径下即可

## 使用

+ 使用 `python SVD.py -m train` 训练
+ 使用 `python SVD.PY -m predict` 预测，可以下载训练好的模型	

## 代码解释

### 1. 载入并划分数据

```python
data_type = {'user_id': np.int32, 'item_id': np.int32, 'rating': np.float32}
names = ['user_id', 'item_id', 'rating']
ratings = pd.read_csv(train_data_path, dtype=data_type, usecols=range(3), names=names)
print(type(ratings))
print(ratings.shape)

reader = Reader(rating_scale=(0, 5))
train_data = Dataset.load_from_df(ratings, reader)
trainset, validateset = train_test_split(train_data, test_size=.25)
```

### 2. 建立并训练模型

```python
model = SVD(n_epochs=i, verbose=False)
model.fit(trainset)
```

### 3. 进行预测

```python
model = pickle.load(open("model_e8.pkl", 'rb'))
time_start = time.time()
length = test_data.shape[0]
for index, row in test_data.iterrows():
	print(row['user_id'], row['item_id'])
	predictions = model.predict(row['user_id'], row['item_id'])
```

