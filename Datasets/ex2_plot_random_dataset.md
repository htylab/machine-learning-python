# Datasets

## 機器學習資料集/ 範例二: Plot randomly generated classification dataset


https://scikit-learn.org/stable/auto_examples/datasets/plot_random_dataset.html

這個範例實現了 `datasets.make_classification` `datasets.make_blobs` 以及 `datasets.make_gaussian_quantiles` 的函數運用



## (一)Make classification
對於`make_classification`的函數，隨機生成n種不同的分類數據集，每個類別具有不同數量的信息特徵和群聚。

```python
plt.title("One informative feature, one cluster per class", fontsize='small')
X1, Y1 = make_classification(n_features=2, n_redundant=0, n_informative=1,
                             n_clusters_per_class=1)
plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1,
            s=25, edgecolor='k')
```
![](https://github.com/JENNSHIUAN/machine-learning-python/blob/master/Datasets/ex2_fig1.JPG)

針對不同數量的信息特徵和群聚會產生不同結果

![](https://github.com/JENNSHIUAN/machine-learning-python/blob/master/Datasets/ex2_fig3.JPG) 
![](https://github.com/JENNSHIUAN/machine-learning-python/blob/master/Datasets/ex2_fig4.JPG)
![](https://github.com/JENNSHIUAN/machine-learning-python/blob/master/Datasets/ex2_fig5.JPG)

## (二)Make blobs
對於`make_blobs`的函數，會產生同向心性的高斯分布群。

```python
plt.title("Three blobs", fontsize='small')
X1, Y1 = make_blobs(n_features=2, centers=3)
plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1,
            s=25, edgecolor='k')
```

![](https://github.com/JENNSHIUAN/machine-learning-python/blob/master/Datasets/ex2_fig2.JPG)

## (三)Make gaussian quantiles
對於`make_gaussian_quantiles`的函數，用分位數生成各向同性的高斯並標記樣本。

```python
X1, Y1 = make_gaussian_quantiles(n_features=2, n_classes=3)
plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1,
            s=25, edgecolor='k')
```
![](https://github.com/JENNSHIUAN/machine-learning-python/blob/master/Datasets/ex2_fig6.JPG)

## (四)完整程式碼
Python source code:plot_random_dataset.py

https://scikit-learn.org/stable/_downloads/9534d593e925347a4e0eee78c7d5b226/plot_random_dataset.py
```python
print(__doc__)

import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles

plt.figure(figsize=(8, 8))
plt.subplots_adjust(bottom=.05, top=.9, left=.05, right=.95)

plt.subplot(321)
plt.title("One informative feature, one cluster per class", fontsize='small')
X1, Y1 = make_classification(n_features=2, n_redundant=0, n_informative=1,
                             n_clusters_per_class=1)
plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1,
            s=25, edgecolor='k')

plt.subplot(322)
plt.title("Two informative features, one cluster per class", fontsize='small')
X1, Y1 = make_classification(n_features=2, n_redundant=0, n_informative=2,
                             n_clusters_per_class=1)
plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1,
            s=25, edgecolor='k')

plt.subplot(323)
plt.title("Two informative features, two clusters per class",
          fontsize='small')
X2, Y2 = make_classification(n_features=2, n_redundant=0, n_informative=2)
plt.scatter(X2[:, 0], X2[:, 1], marker='o', c=Y2,
            s=25, edgecolor='k')

plt.subplot(324)
plt.title("Multi-class, two informative features, one cluster",
          fontsize='small')
X1, Y1 = make_classification(n_features=2, n_redundant=0, n_informative=2,
                             n_clusters_per_class=1, n_classes=3)
plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1,
            s=25, edgecolor='k')

plt.subplot(325)
plt.title("Three blobs", fontsize='small')
X1, Y1 = make_blobs(n_features=2, centers=3)
plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1,
            s=25, edgecolor='k')

plt.subplot(326)
plt.title("Gaussian divided into three quantiles", fontsize='small')
X1, Y1 = make_gaussian_quantiles(n_features=2, n_classes=3)
plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1,
            s=25, edgecolor='k')

plt.show()
```
