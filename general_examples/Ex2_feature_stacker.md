
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

測試資料：
* `iris`為一個dict型別資料。

| 顯示 | 說明 |
| -- | -- |
| ('target_names', (3L,))| 共有三種鳶尾花 setosa, versicolor, virginica |
| ('data', (150L, 4L)) | 有150筆資料，共四種特徵 |
| ('target', (150L,))| 這150筆資料各是那一種鳶尾花|
| DESCR | 資料之描述 |
| feature_names| 四個特徵代表的意義 |

# (二)PCA與SelectKBest
* Principal Component Analysis(PCA)主成份分析，是最常用的線性降維方法，它的目標是通過某種線性投影，將高維的數據映射到低維的空間中表示，並期望在所投影的維度上數據的方差最大，以此使用較少的數據維度，同時保留住較多的原數據點的特性。<br />
 class sklearn.decomposition.PCA(n_components=None, copy=True, whiten=False)
 
 

* 使用 `SelectKBest` 設定要用哪種目標函式，以挑出可提供信息的特徵。<br />
 class sklearn.feature_selection.SelectKBest(score_func=<function f_classif at 0x7f49246ca048>, k=10)



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

<<<<<<< HEAD
    Fitting 3 folds for each of 18 candidates, totalling 54 fits
    [CV] features__univ_select__k=1, features__pca__n_components=1, svm__C=0.1 
    [CV]  features__univ_select__k=1, features__pca__n_components=1, svm__C=0.1, score=0.960784 -   0.0s
    [CV] features__univ_select__k=1, features__pca__n_components=1, svm__C=0.1 
    [CV]  features__univ_select__k=1, features__pca__n_components=1, svm__C=0.1, score=0.901961 -   0.0s
    [CV] features__univ_select__k=1, features__pca__n_components=1, svm__C=0.1 
    [CV]  features__univ_select__k=1, features__pca__n_components=1, svm__C=0.1, score=0.979167 -   0.0s
    [CV] features__univ_select__k=1, features__pca__n_components=1, svm__C=1 
    [CV]  features__univ_select__k=1, features__pca__n_components=1, svm__C=1, score=0.941176 -   0.0s
    [CV] features__univ_select__k=1, features__pca__n_components=1, svm__C=1 
    [CV]  features__univ_select__k=1, features__pca__n_components=1, svm__C=1, score=0.921569 -   0.0s
    [CV] features__univ_select__k=1, features__pca__n_components=1, svm__C=1 
    [CV]  features__univ_select__k=1, features__pca__n_components=1, svm__C=1, score=0.979167 -   0.0s
    [CV] features__univ_select__k=1, features__pca__n_components=1, svm__C=10 
    [CV]  features__univ_select__k=1, features__pca__n_components=1, svm__C=10, score=0.960784 -   0.0s
    [CV] features__univ_select__k=1, features__pca__n_components=1, svm__C=10 
    [CV]  features__univ_select__k=1, features__pca__n_components=1, svm__C=10, score=0.921569 -   0.0s
    [CV] features__univ_select__k=1, features__pca__n_components=1, svm__C=10 
    [CV]  features__univ_select__k=1, features__pca__n_components=1, svm__C=10, score=0.979167 -   0.0s
    [CV] features__univ_select__k=2, features__pca__n_components=1, svm__C=0.1 
    [CV]  features__univ_select__k=2, features__pca__n_components=1, svm__C=0.1, score=0.960784 -   0.0s
    [CV] features__univ_select__k=2, features__pca__n_components=1, svm__C=0.1 
    [CV]  features__univ_select__k=2, features__pca__n_components=1, svm__C=0.1, score=0.921569 -   0.0s
    [CV] features__univ_select__k=2, features__pca__n_components=1, svm__C=0.1 
    [CV]  features__univ_select__k=2, features__pca__n_components=1, svm__C=0.1, score=0.979167 -   0.0s
    [CV] features__univ_select__k=2, features__pca__n_components=1, svm__C=1 
    [CV]  features__univ_select__k=2, features__pca__n_components=1, svm__C=1, score=0.960784 -   0.0s
    [CV] features__univ_select__k=2, features__pca__n_components=1, svm__C=1 
    [CV]  features__univ_select__k=2, features__pca__n_components=1, svm__C=1, score=0.921569 -   0.0s
    [CV] features__univ_select__k=2, features__pca__n_components=1, svm__C=1 
    [CV]  features__univ_select__k=2, features__pca__n_components=1, svm__C=1, score=1.000000 -   0.0s
    [CV] features__univ_select__k=2, features__pca__n_components=1, svm__C=10 

    [Parallel(n_jobs=1)]: Done   1 jobs       | elapsed:    0.0s
    [Parallel(n_jobs=1)]: Done   2 jobs       | elapsed:    0.0s
    [Parallel(n_jobs=1)]: Done   5 jobs       | elapsed:    0.0s
    [Parallel(n_jobs=1)]: Done   8 jobs       | elapsed:    0.0s
    [Parallel(n_jobs=1)]: Done  13 jobs       | elapsed:    0.0s
    [Parallel(n_jobs=1)]: Done  18 jobs       | elapsed:    0.0s
    

    
    [CV]  features__univ_select__k=2, features__pca__n_components=1, svm__C=10, score=0.980392 -   0.0s
    [CV] features__univ_select__k=2, features__pca__n_components=1, svm__C=10 
    [CV]  features__univ_select__k=2, features__pca__n_components=1, svm__C=10, score=0.901961 -   0.0s
    [CV] features__univ_select__k=2, features__pca__n_components=1, svm__C=10 
    [CV]  features__univ_select__k=2, features__pca__n_components=1, svm__C=10, score=1.000000 -   0.0s
    [CV] features__univ_select__k=1, features__pca__n_components=2, svm__C=0.1 
    [CV]  features__univ_select__k=1, features__pca__n_components=2, svm__C=0.1, score=0.960784 -   0.0s
    [CV] features__univ_select__k=1, features__pca__n_components=2, svm__C=0.1 
    [CV]  features__univ_select__k=1, features__pca__n_components=2, svm__C=0.1, score=0.901961 -   0.0s
    [CV] features__univ_select__k=1, features__pca__n_components=2, svm__C=0.1 
    [CV]  features__univ_select__k=1, features__pca__n_components=2, svm__C=0.1, score=0.979167 -   0.0s
    [CV] features__univ_select__k=1, features__pca__n_components=2, svm__C=1 
    [CV]  features__univ_select__k=1, features__pca__n_components=2, svm__C=1, score=0.980392 -   0.0s
    [CV] features__univ_select__k=1, features__pca__n_components=2, svm__C=1 
    [CV]  features__univ_select__k=1, features__pca__n_components=2, svm__C=1, score=0.941176 -   0.0s
    [CV] features__univ_select__k=1, features__pca__n_components=2, svm__C=1 
    [CV]  features__univ_select__k=1, features__pca__n_components=2, svm__C=1, score=0.979167 -   0.0s
    [CV] features__univ_select__k=1, features__pca__n_components=2, svm__C=10 
    [CV]  features__univ_select__k=1, features__pca__n_components=2, svm__C=10, score=0.980392 -   0.0s
    [CV] features__univ_select__k=1, features__pca__n_components=2, svm__C=10 
    [CV]  features__univ_select__k=1, features__pca__n_components=2, svm__C=10, score=0.941176 -   0.0s
    [CV] features__univ_select__k=1, features__pca__n_components=2, svm__C=10 
    [CV]  features__univ_select__k=1, features__pca__n_components=2, svm__C=10, score=0.979167 -   0.0s
    [CV] features__univ_select__k=2, features__pca__n_components=2, svm__C=0.1 
    [CV]  features__univ_select__k=2, features__pca__n_components=2, svm__C=0.1, score=0.980392 -   0.0s
    [CV] features__univ_select__k=2, features__pca__n_components=2, svm__C=0.1 
    [CV]  features__univ_select__k=2, features__pca__n_components=2, svm__C=0.1, score=0.941176 -   0.0s
    [CV] features__univ_select__k=2, features__pca__n_components=2, svm__C=0.1 
    [CV]  features__univ_select__k=2, features__pca__n_components=2, svm__C=0.1, score=0.979167 -   0.0s
    [CV] features__univ_select__k=2, features__pca__n_components=2, svm__C=1 
    [CV]  features__univ_select__k=2, features__pca__n_components=2, svm__C=1, score=1.000000 -   0.0s
    [CV] features__univ_select__k=2, features__pca__n_components=2, svm__C=1 
    [CV]  features__univ_select__k=2, features__pca__n_components=2, svm__C=1, score=0.960784 -   0.0s
    [CV] features__univ_select__k=2, features__pca__n_components=2, svm__C=1 
    [CV]  features__univ_select__k=2, features__pca__n_components=2, svm__C=1, score=0.979167 -   0.0s
    [CV] features__univ_select__k=2, features__pca__n_components=2, svm__C=10 
    [CV]  features__univ_select__k=2, features__pca__n_components=2, svm__C=10, score=0.980392 -   0.0s
    [CV] features__univ_select__k=2, features__pca__n_components=2, svm__C=10 
    [CV]  features__univ_select__k=2, features__pca__n_components=2, svm__C=10, score=0.921569 -   0.0s
    [CV] features__univ_select__k=2, features__pca__n_components=2, svm__C=10 
    [CV]  features__univ_select__k=2, features__pca__n_components=2, svm__C=10, score=1.000000 -   0.0s
    [CV] features__univ_select__k=1, features__pca__n_components=3, svm__C=0.1 
    [CV]  features__univ_select__k=1, features__pca__n_components=3, svm__C=0.1, score=0.980392 -   0.0s
    [CV] features__univ_select__k=1, features__pca__n_components=3, svm__C=0.1 
    [CV]  features__univ_select__k=1, features__pca__n_components=3, svm__C=0.1, score=0.941176 -   0.0s
    [CV] features__univ_select__k=1, features__pca__n_components=3, svm__C=0.1 
    [CV]  features__univ_select__k=1, features__pca__n_components=3, svm__C=0.1, score=0.979167 -   0.0s
    [CV] features__univ_select__k=1, features__pca__n_components=3, svm__C=1 
    [CV]  features__univ_select__k=1, features__pca__n_components=3, svm__C=1, score=1.000000 -   0.0s
    [CV] features__univ_select__k=1, features__pca__n_components=3, svm__C=1 
    [CV]  features__univ_select__k=1, features__pca__n_components=3, svm__C=1, score=0.941176 -   0.0s
    [CV] features__univ_select__k=1, features__pca__n_components=3, svm__C=1 
    [CV]  features__univ_select__k=1, features__pca__n_components=3, svm__C=1, score=0.979167 -   0.0s
    [CV] features__univ_select__k=1, features__pca__n_components=3, svm__C=10 
    [CV]  features__univ_select__k=1, features__pca__n_components=3, svm__C=10, score=1.000000 -   0.0s
    [CV] features__univ_select__k=1, features__pca__n_components=3, svm__C=10 
    [CV]  features__univ_select__k=1, features__pca__n_components=3, svm__C=10, score=0.921569 -   0.0s
    [CV] features__univ_select__k=1, features__pca__n_components=3, svm__C=10 
    [CV]  features__univ_select__k=1, features__pca__n_components=3, svm__C=10, score=1.000000 -   0.0s
    [CV] features__univ_select__k=2, features__pca__n_components=3, svm__C=0.1 
    [CV]  features__univ_select__k=2, features__pca__n_components=3, svm__C=0.1, score=0.980392 -   0.0s
    [CV] features__univ_select__k=2, features__pca__n_components=3, svm__C=0.1 
    [CV]  features__univ_select__k=2, features__pca__n_components=3, svm__C=0.1, score=0.941176 -   0.0s
    [CV] features__univ_select__k=2, features__pca__n_components=3, svm__C=0.1 
    [CV]  features__univ_select__k=2, features__pca__n_components=3, svm__C=0.1, score=0.979167 -   0.0s
    [CV] features__univ_select__k=2, features__pca__n_components=3, svm__C=1 
    [CV]  features__univ_select__k=2, features__pca__n_components=3, svm__C=1, score=1.000000 -   0.0s

    [Parallel(n_jobs=1)]: Done  25 jobs       | elapsed:    0.0s
    [Parallel(n_jobs=1)]: Done  32 jobs       | elapsed:    0.0s
    [Parallel(n_jobs=1)]: Done  41 jobs       | elapsed:    0.0s
    [Parallel(n_jobs=1)]: Done  50 jobs       | elapsed:    0.1s
    

    
    [CV] features__univ_select__k=2, features__pca__n_components=3, svm__C=1 
    [CV]  features__univ_select__k=2, features__pca__n_components=3, svm__C=1, score=0.960784 -   0.0s
    [CV] features__univ_select__k=2, features__pca__n_components=3, svm__C=1 
    [CV]  features__univ_select__k=2, features__pca__n_components=3, svm__C=1, score=0.979167 -   0.0s
    [CV] features__univ_select__k=2, features__pca__n_components=3, svm__C=10 
    [CV]  features__univ_select__k=2, features__pca__n_components=3, svm__C=10, score=1.000000 -   0.0s
    [CV] features__univ_select__k=2, features__pca__n_components=3, svm__C=10 
    [CV]  features__univ_select__k=2, features__pca__n_components=3, svm__C=10, score=0.921569 -   0.0s
    [CV] features__univ_select__k=2, features__pca__n_components=3, svm__C=10 
    [CV]  features__univ_select__k=2, features__pca__n_components=3, svm__C=10, score=1.000000 -   0.0s
    Pipeline(steps=[('features', FeatureUnion(n_jobs=1,
           transformer_list=[('pca', PCA(copy=True, n_components=2, whiten=False)), ('univ_select', SelectKBest(k=2, score_func=<function f_classif at 0x0000000007243488>))],
           transformer_weights=None)), ('svm', SVC(C=1, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0,
      kernel='linear', max_iter=-1, probability=False, random_state=None,
      shrinking=True, tol=0.001, verbose=False))])
    

    [Parallel(n_jobs=1)]: Done  54 out of  54 | elapsed:    0.1s finished
    


```python

```
=======
>>>>>>> origin/master
