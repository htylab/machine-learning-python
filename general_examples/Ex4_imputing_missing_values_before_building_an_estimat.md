# 通用範例/範例四: Imputing missing values before building an estimator

http://scikit-learn.org/stable/auto_examples/missing_values.html#example-missing-values-py

在這範例說明輸入missing values可以比對刪掉missing values得到更好的結果。但這並不適用於所有的情況，仍然需要進行交叉驗證。<br />
missing values可以用均值、中位值，或者頻繁出現的值代替。中位值對大數量級的數據來說是比較穩定的估計值。

## (一)引入函式庫及內建測試資料庫

引入之函式庫如下

1. `sklearn.ensemble.RandomForestRegressor`: 隨機森林回歸
2. `sklearn.pipeline.Pipeline`: 串聯估計器
3. `sklearn.preprocessing.Imputer`: Imputation transformer for completing missing values
4. `sklearn.cross_validation import cross_val_score`:交叉验证函数

## (二)引入內建測試資料庫(boston房產資料)
使用 `datasets.load_boston()` 將資料存入， `boston` 為一個dict型別資料，我們看一下資料的內容。<br />
n_samples 為樣本數<br />
n_features 為特徵數

```python
dataset = load_boston()
X_full, y_full = dataset.data, dataset.target
n_samples = X_full.shape[0]
n_features = X_full.shape[1]
```

| 顯示 | 說明 |
| -- | -- |
| ('data', (506, 13))| 機器學習數據 |
| ('feature_names', (13,)) |  |
| ('target', (506,)) | 回歸目標 |
| DESCR | 資料之描述 |


## (三)估計整個數據集的得分
全部的資料使用隨機森林回歸函數進行交叉驗證，得到一個分數。<br />

Score with the entire dataset = 0.56
```python
estimator = RandomForestRegressor(random_state=0, n_estimators=100)
score = cross_val_score(estimator, X_full, y_full).mean()
print("Score with the entire dataset = %.2f" % score)
```

## (四)設定損失比例並估計得分
損失比例75%，損失樣本數為379筆，剩餘樣本為127筆。<br />
將127筆資料進行隨機森林回歸函數進行交叉驗證，並得到一個分數。<br />

Score without the samples containing missing values = 0.49
```python
missing_rate = 0.75
n_missing_samples = np.floor(n_samples * missing_rate)
missing_samples = np.hstack((np.zeros(n_samples - n_missing_samples,
                                      dtype=np.bool),
                             np.ones(n_missing_samples,
                                     dtype=np.bool)))
rng.shuffle(missing_samples)
missing_features = rng.randint(0, n_features, n_missing_samples)

# Estimate the score without the lines containing missing values
X_filtered = X_full[~missing_samples, :]
y_filtered = y_full[~missing_samples]
estimator = RandomForestRegressor(random_state=0, n_estimators=100)
score = cross_val_score(estimator, X_filtered, y_filtered).mean()
print("Score without the samples containing missing values = %.2f" % score)
```