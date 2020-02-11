# **Nearest Neighbors Classification**
https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html#sphx-glr-auto-examples-neighbors-plot-classification-py

此範例使用Nearest Neighbors Classification (簡稱NNC 近鄰分類法)將數據分類，並繪製出各類別的decision boundaries(決策邊界)

NNC計算的基礎是「物以類聚」，換句話說，同類型的資料應該會聚集在一起，若以座標中的點來表示，則這些點的距離應該會比較接近。因此，對於一筆未被標籤的資料，我們只要找出在訓練集中和此筆資料最接近的點，就可以判定此筆資料的類別應該和最接近的點的類別是一樣的。NNC是一個較直覺的分類法，在測試各種分類器時，常被當成是基礎的分類器，以便和其他更複雜的分類器進行效能比較。

## (一)引入函式庫

* numpy : 產生陣列數值
* matplotlib.pyplot : 用來繪製影像
* matplotlib.colors import ListedColormap : 匯入用來生成圖上的顏色表
* sklearn import neighbors, datasets : 匯入NNC及資料集
```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
```
## (二)匯入資料集

```python
n_neighbors = 15 # 用於NNC函式內的變數

# import some data to play with
iris = datasets.load_iris() # 匯入資料集

# we only take the first two features. We could avoid this ugly
# slicing by using a two-dim dataset
X = iris.data[:, :2] # 只取資料的前2項數據
y = iris.target # 分類的標籤

h = .02  # step size in the mesh
```
## (三)繪製結果圖

* neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None, **kwargs)
1. n_neighbors : 近鄰查詢的鄰居數
2. weights : 用於預測的權重函數。'uniform' 每個點都被平均加權 'distance' 權重是點之間距離的倒數。在這種情況下，查詢點的近鄰比遠處的近鄰具有更大的影響
3. algorithm{‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’} : 用於計算近鄰的算法。
* np.meshgrid() : 從給定的座標向量回傳座標矩陣
```python
# Create color maps
cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])

for weights in ['uniform', 'distance']:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights) # 使用NNC函式
    clf.fit(X, y) # 擬合資料集

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1 # 設定mesh x的大小邊界
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1 # 設定mesh y的大小邊界
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) # 預測資料的分類

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))

plt.show()
```
![](https://github.com/sdgary56249128/machine-learning-python/blob/master/Nearest_Neighbors/sphx_glr_plot_classification_001.png)
![](https://github.com/sdgary56249128/machine-learning-python/blob/master/Nearest_Neighbors/sphx_glr_plot_classification_002.png)

https://scikit-learn.org/stable/_downloads/fb5fbc2d9b876b776e016c37233e76fd/plot_classification.py
```python
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

n_neighbors = 15

# import some data to play with
iris = datasets.load_iris()

# we only take the first two features. We could avoid this ugly
# slicing by using a two-dim dataset
X = iris.data[:, :2]
y = iris.target

h = .02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])

for weights in ['uniform', 'distance']:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))

plt.show()
```
