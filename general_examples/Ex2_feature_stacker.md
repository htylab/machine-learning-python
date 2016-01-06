
# 通用範例/範例二: Concatenating multiple feature extraction methods

http://scikit-learn.org/stable/auto_examples/feature_stacker.html

這個範例的主要目的
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




```python
# This dataset is way to high-dimensional. Better do PCA:
pca = PCA(n_components=2)

# Maybe some original features where good, too?
selection = SelectKBest(k=1)

# Build estimator from PCA and Univariate selection:

combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])

# Use combined features to transform dataset:
X_features = combined_features.fit(X, y).transform(X)

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

    Fitting 3 folds for each of 18 candidates, totalling 54 fits
    [CV] svm__C=0.1, features__pca__n_components=1, features__univ_select__k=1 
    [CV]  svm__C=0.1, features__pca__n_components=1, features__univ_select__k=1, score=0.960784 -   0.0s
    [CV] svm__C=0.1, features__pca__n_components=1, features__univ_select__k=1 
    [CV]  svm__C=0.1, features__pca__n_components=1, features__univ_select__k=1, score=0.901961 -   0.0s
    [CV] svm__C=0.1, features__pca__n_components=1, features__univ_select__k=1 
    [CV]  svm__C=0.1, features__pca__n_components=1, features__univ_select__k=1, score=0.979167 -   0.0s
    [CV] svm__C=1, features__pca__n_components=1, features__univ_select__k=1 
    [CV]  svm__C=1, features__pca__n_components=1, features__univ_select__k=1, score=0.941176 -   0.0s
    [CV] svm__C=1, features__pca__n_components=1, features__univ_select__k=1 
    [CV]  svm__C=1, features__pca__n_components=1, features__univ_select__k=1, score=0.921569 -   0.0s
    [CV] svm__C=1, features__pca__n_components=1, features__univ_select__k=1 
    [CV]  svm__C=1, features__pca__n_components=1, features__univ_select__k=1, score=0.979167 -   0.0s
    [CV] svm__C=10, features__pca__n_components=1, features__univ_select__k=1 
    [CV]  svm__C=10, features__pca__n_components=1, features__univ_select__k=1, score=0.960784 -   0.0s
    [CV] svm__C=10, features__pca__n_components=1, features__univ_select__k=1 
    [CV]  svm__C=10, features__pca__n_components=1, features__univ_select__k=1, score=0.921569 -   0.0s
    [CV] svm__C=10, features__pca__n_components=1, features__univ_select__k=1 
    [CV]  svm__C=10, features__pca__n_components=1, features__univ_select__k=1, score=0.979167 -   0.0s
    [CV] svm__C=0.1, features__pca__n_components=1, features__univ_select__k=2 
    [CV]  svm__C=0.1, features__pca__n_components=1, features__univ_select__k=2, score=0.960784 -   0.0s
    [CV] svm__C=0.1, features__pca__n_components=1, features__univ_select__k=2 
    [CV]  svm__C=0.1, features__pca__n_components=1, features__univ_select__k=2, score=0.921569 -   0.0s
    [CV] svm__C=0.1, features__pca__n_components=1, features__univ_select__k=2 
    [CV]  svm__C=0.1, features__pca__n_components=1, features__univ_select__k=2, score=0.979167 -   0.0s
    [CV] svm__C=1, features__pca__n_components=1, features__univ_select__k=2 
    [CV]  svm__C=1, features__pca__n_components=1, features__univ_select__k=2, score=0.960784 -   0.0s
    [CV] svm__C=1, features__pca__n_components=1, features__univ_select__k=2 
    [CV]  svm__C=1, features__pca__n_components=1, features__univ_select__k=2, score=0.921569 -   0.0s
    [CV] svm__C=1, features__pca__n_components=1, features__univ_select__k=2 
    [CV]  svm__C=1, features__pca__n_components=1, features__univ_select__k=2, score=1.000000 -   0.0s
    [CV] svm__C=10, features__pca__n_components=1, features__univ_select__k=2 
    [CV]  svm__C=10, features__pca__n_components=1, features__univ_select__k=2, score=0.980392 -   0.0s
    [CV] svm__C=10, features__pca__n_components=1, features__univ_select__k=2 
    [CV]  svm__C=10, features__pca__n_components=1, features__univ_select__k=2, score=0.901961 -   0.0s

    [Parallel(n_jobs=1)]: Done   1 jobs       | elapsed:    0.0s
    [Parallel(n_jobs=1)]: Done   2 jobs       | elapsed:    0.0s
    [Parallel(n_jobs=1)]: Done   5 jobs       | elapsed:    0.0s
    [Parallel(n_jobs=1)]: Done   8 jobs       | elapsed:    0.0s
    [Parallel(n_jobs=1)]: Done  13 jobs       | elapsed:    0.0s
    [Parallel(n_jobs=1)]: Done  18 jobs       | elapsed:    0.0s
    [Parallel(n_jobs=1)]: Done  25 jobs       | elapsed:    0.0s
    

    
    [CV] svm__C=10, features__pca__n_components=1, features__univ_select__k=2 
    [CV]  svm__C=10, features__pca__n_components=1, features__univ_select__k=2, score=1.000000 -   0.0s
    [CV] svm__C=0.1, features__pca__n_components=2, features__univ_select__k=1 
    [CV]  svm__C=0.1, features__pca__n_components=2, features__univ_select__k=1, score=0.960784 -   0.0s
    [CV] svm__C=0.1, features__pca__n_components=2, features__univ_select__k=1 
    [CV]  svm__C=0.1, features__pca__n_components=2, features__univ_select__k=1, score=0.901961 -   0.0s
    [CV] svm__C=0.1, features__pca__n_components=2, features__univ_select__k=1 
    [CV]  svm__C=0.1, features__pca__n_components=2, features__univ_select__k=1, score=0.979167 -   0.0s
    [CV] svm__C=1, features__pca__n_components=2, features__univ_select__k=1 
    [CV]  svm__C=1, features__pca__n_components=2, features__univ_select__k=1, score=0.980392 -   0.0s
    [CV] svm__C=1, features__pca__n_components=2, features__univ_select__k=1 
    [CV]  svm__C=1, features__pca__n_components=2, features__univ_select__k=1, score=0.941176 -   0.0s
    [CV] svm__C=1, features__pca__n_components=2, features__univ_select__k=1 
    [CV]  svm__C=1, features__pca__n_components=2, features__univ_select__k=1, score=0.979167 -   0.0s
    [CV] svm__C=10, features__pca__n_components=2, features__univ_select__k=1 
    [CV]  svm__C=10, features__pca__n_components=2, features__univ_select__k=1, score=0.980392 -   0.0s
    [CV] svm__C=10, features__pca__n_components=2, features__univ_select__k=1 
    [CV]  svm__C=10, features__pca__n_components=2, features__univ_select__k=1, score=0.941176 -   0.0s
    [CV] svm__C=10, features__pca__n_components=2, features__univ_select__k=1 
    [CV]  svm__C=10, features__pca__n_components=2, features__univ_select__k=1, score=0.979167 -   0.0s
    [CV] svm__C=0.1, features__pca__n_components=2, features__univ_select__k=2 
    [CV]  svm__C=0.1, features__pca__n_components=2, features__univ_select__k=2, score=0.980392 -   0.0s
    [CV] svm__C=0.1, features__pca__n_components=2, features__univ_select__k=2 
    [CV]  svm__C=0.1, features__pca__n_components=2, features__univ_select__k=2, score=0.941176 -   0.0s
    [CV] svm__C=0.1, features__pca__n_components=2, features__univ_select__k=2 
    [CV]  svm__C=0.1, features__pca__n_components=2, features__univ_select__k=2, score=0.979167 -   0.0s
    [CV] svm__C=1, features__pca__n_components=2, features__univ_select__k=2 
    [CV]  svm__C=1, features__pca__n_components=2, features__univ_select__k=2, score=1.000000 -   0.0s
    [CV] svm__C=1, features__pca__n_components=2, features__univ_select__k=2 
    [CV]  svm__C=1, features__pca__n_components=2, features__univ_select__k=2, score=0.960784 -   0.0s
    [CV] svm__C=1, features__pca__n_components=2, features__univ_select__k=2 
    [CV]  svm__C=1, features__pca__n_components=2, features__univ_select__k=2, score=0.979167 -   0.0s
    [CV] svm__C=10, features__pca__n_components=2, features__univ_select__k=2 
    [CV]  svm__C=10, features__pca__n_components=2, features__univ_select__k=2, score=0.980392 -   0.0s
    [CV] svm__C=10, features__pca__n_components=2, features__univ_select__k=2 
    [CV]  svm__C=10, features__pca__n_components=2, features__univ_select__k=2, score=0.921569 -   0.0s
    [CV] svm__C=10, features__pca__n_components=2, features__univ_select__k=2 
    [CV]  svm__C=10, features__pca__n_components=2, features__univ_select__k=2, score=1.000000 -   0.0s
    [CV] svm__C=0.1, features__pca__n_components=3, features__univ_select__k=1 
    [CV]  svm__C=0.1, features__pca__n_components=3, features__univ_select__k=1, score=0.980392 -   0.0s
    [CV] svm__C=0.1, features__pca__n_components=3, features__univ_select__k=1 
    [CV]  svm__C=0.1, features__pca__n_components=3, features__univ_select__k=1, score=0.941176 -   0.0s
    [CV] svm__C=0.1, features__pca__n_components=3, features__univ_select__k=1 
    [CV]  svm__C=0.1, features__pca__n_components=3, features__univ_select__k=1, score=0.979167 -   0.0s
    [CV] svm__C=1, features__pca__n_components=3, features__univ_select__k=1 
    [CV]  svm__C=1, features__pca__n_components=3, features__univ_select__k=1, score=1.000000 -   0.0s
    [CV] svm__C=1, features__pca__n_components=3, features__univ_select__k=1 
    [CV]  svm__C=1, features__pca__n_components=3, features__univ_select__k=1, score=0.941176 -   0.0s
    [CV] svm__C=1, features__pca__n_components=3, features__univ_select__k=1 
    [CV]  svm__C=1, features__pca__n_components=3, features__univ_select__k=1, score=0.979167 -   0.0s
    [CV] svm__C=10, features__pca__n_components=3, features__univ_select__k=1 
    [CV]  svm__C=10, features__pca__n_components=3, features__univ_select__k=1, score=1.000000 -   0.0s
    [CV] svm__C=10, features__pca__n_components=3, features__univ_select__k=1 
    [CV]  svm__C=10, features__pca__n_components=3, features__univ_select__k=1, score=0.921569 -   0.0s
    [CV] svm__C=10, features__pca__n_components=3, features__univ_select__k=1 
    [CV]  svm__C=10, features__pca__n_components=3, features__univ_select__k=1, score=1.000000 -   0.0s
    [CV] svm__C=0.1, features__pca__n_components=3, features__univ_select__k=2 
    [CV]  svm__C=0.1, features__pca__n_components=3, features__univ_select__k=2, score=0.980392 -   0.0s
    [CV] svm__C=0.1, features__pca__n_components=3, features__univ_select__k=2 
    [CV]  svm__C=0.1, features__pca__n_components=3, features__univ_select__k=2, score=0.941176 -   0.0s
    [CV] svm__C=0.1, features__pca__n_components=3, features__univ_select__k=2 
    [CV]  svm__C=0.1, features__pca__n_components=3, features__univ_select__k=2, score=0.979167 -   0.0s
    [CV] svm__C=1, features__pca__n_components=3, features__univ_select__k=2 
    [CV]  svm__C=1, features__pca__n_components=3, features__univ_select__k=2, score=1.000000 -   0.0s
    [CV] svm__C=1, features__pca__n_components=3, features__univ_select__k=2 

    [Parallel(n_jobs=1)]: Done  32 jobs       | elapsed:    0.0s
    [Parallel(n_jobs=1)]: Done  41 jobs       | elapsed:    0.0s
    [Parallel(n_jobs=1)]: Done  50 jobs       | elapsed:    0.1s
    

    
    [CV]  svm__C=1, features__pca__n_components=3, features__univ_select__k=2, score=0.960784 -   0.0s
    [CV] svm__C=1, features__pca__n_components=3, features__univ_select__k=2 
    [CV]  svm__C=1, features__pca__n_components=3, features__univ_select__k=2, score=0.979167 -   0.0s
    [CV] svm__C=10, features__pca__n_components=3, features__univ_select__k=2 
    [CV]  svm__C=10, features__pca__n_components=3, features__univ_select__k=2, score=1.000000 -   0.0s
    [CV] svm__C=10, features__pca__n_components=3, features__univ_select__k=2 
    [CV]  svm__C=10, features__pca__n_components=3, features__univ_select__k=2, score=0.921569 -   0.0s
    [CV] svm__C=10, features__pca__n_components=3, features__univ_select__k=2 
    [CV]  svm__C=10, features__pca__n_components=3, features__univ_select__k=2, score=1.000000 -   0.0s
    Pipeline(steps=[('features', FeatureUnion(n_jobs=1,
           transformer_list=[('pca', PCA(copy=True, n_components=2, whiten=False)), ('univ_select', SelectKBest(k=2, score_func=<function f_classif at 0x00000000075A1BF8>))],
           transformer_weights=None)), ('svm', SVC(C=1, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0,
      kernel='linear', max_iter=-1, probability=False, random_state=None,
      shrinking=True, tol=0.001, verbose=False))])
    

    [Parallel(n_jobs=1)]: Done  54 out of  54 | elapsed:    0.1s finished
    


```python

```


```python

```
