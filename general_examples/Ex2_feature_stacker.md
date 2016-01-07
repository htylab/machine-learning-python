
# 通用範例/範例二: Concatenating multiple feature extraction methods

http://scikit-learn.org/stable/auto_examples/feature_stacker.html

在許多真的案例中，會有很多方法可以從一個數據集中提取特徵。也常常會組合多個方法來獲得良好的特徵。這個例子說明如何使用` FeatureUnion` 來結合由` PCA` 和` univariate selection` 時的特徵。雖然在這個例子中使用此方法並沒有特殊幫助，只是用來說明如何使用` FeatureUnion` 。

這個範例的主要目的：
* 使用iris 鳶尾花資料集
* 使用`FeatureUnion`


# (一)資料匯入及描述

* 首先先匯入iris 鳶尾花資料集，使用from sklearn.datasets import load_iris將資料存入
* 準備X (特徵資料) 以及 y (目標資料)


```python
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

iris = load_iris()

X, y = iris.data, iris.target
```

測試資料：<br />
`iris`為一個dict型別資料。

| 顯示 | 說明 |
| -- | -- |
| ('target_names', (3L,))| 共有三種鳶尾花 setosa, versicolor, virginica |
| ('data', (150L, 4L)) | 有150筆資料，共四種特徵 |
| ('target', (150L,))| 這150筆資料各是那一種鳶尾花|
| DESCR | 資料之描述 |
| feature_names| 四個特徵代表的意義 |

# (二)PCA與SelectKBest
* `PCA(n_components = 主要成份數量)`:Principal Component Analysis(PCA)主成份分析，是一個常用的將資料維度減少的方法。它的原理是找出一個新的座標軸，將資料投影到該軸時，數據的變異量會最大。利用這個方式減少資料維度，又希望能保留住原數據點的特性。

* `SelectKBest(score_func , k )`: `score_func`是選擇特徵值所依據的函式，而`K`值則是設定要選出多少特徵。


```python
# This dataset is way to high-dimensional. Better do PCA:
pca = PCA(n_components=2)

# Maybe some original features where good, too?
selection = SelectKBest(k=1)
```

# (三)FeatureUnionc
使用sklearn.pipeline.FeatureUnion合併主成分分析(PCA)和綜合篩選(SelectKBest)。<br />
最後得到選出的特徵。



```python
# Build estimator from PCA and Univariate selection:

combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])

# Use combined features to transform dataset:
X_features = combined_features.fit(X, y).transform(X)
```

# (四)找到最佳的結果
* Scikit-lenarn的支持向量機分類涵式庫提供使用簡單易懂的指令，可以用 SVC() 建立運算物件，之後並可以用運算物件內的方法 .fit() 與 .predict() 來做訓練與預測。

* 使用`GridSearchCV`交叉驗證，得到由參數網格計算出的分數網格，並找到分數網格中最優的點。

    最後印出這個點所代表的參數





```python
svm = SVC(kernel="linear")

# Do grid search over k, n_components and C:

pipeline = Pipeline([("features", combined_features), ("svm", svm)])

param_grid = dict(features__pca__n_components=[1, 2, 3],
                  features__univ_select__k=[1, 2],
                  svm__C=[0.1, 1, 10])

grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=10)
grid_search.fit(X, y)
print(grid_search.best_estimator_)
```
結果顯示
``` Fitting 3 folds for each of 18 candidates, totalling 54 fits
    [CV] features__univ_select__k=1, features__pca__n_components=1, svm__C=0.1
    [CV]  features__univ_select__k=1, features__pca__n_components=1, svm__C=0.1, score=0.960784 -   0.0s
```
