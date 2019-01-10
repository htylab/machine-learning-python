# **範例二:A demo of the mean-shift clustering algorithm**

https://scikit-learn.org/stable/auto_examples/cluster/plot_mean_shift.html#sphx-glr-auto-examples-cluster-plot-mean-shift-py

此範例展示一種強建的特徵空間分析法

1. 利用 make_blobs 來建立所需的樣本點
2. 利用均值漂移算法找到各類質心集合
3. 通過找到給定樣本的最近質心來給新樣本上標籤


## (一)引入函式庫

引入函式如下:

1. numpy : 產生陣列數值
2. matplotlib.pyplot : 用來繪製影像
3. sklearn.cluster import MeanShift, estimate_bandwidth : MeanShift:發現樣本的平滑密度中的點 ; estimate_bandwidth:計算要用於maen-shift演算法的頻寬
4. sklearn.datasets.samples_generator import make_blobs : 產生用於clustering的等向高斯分布點
5. itertools import cycle : 產生一個迭代器，對迭代器中的元素反覆執行

```python
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs
```

```python
# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]
X, _ = make_blobs(n_samples=10000, centers=centers, cluster_std=0.6)
```
根據提供的3個中心點，產生各10000個等向高斯的點


## (二)Clustering
```python
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)
```
estimate_bandwidth 算出的 bandwidth 會用來作為提供 RBF krenel 的參數，用在 MeanShift 的 bandwidth 參數裡面
RBF kernel : 主要用於線性不可分的情形，將資料投射到更高維的空間，讓其變得可以線性分割
做聚集後就可得各類別的中心點，以及各點的label

```python
# Plot result
import matplotlib.pyplot as plt
from itertools import cycle

plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
```
colors : 在這用作圖形顏色切換
plt.plot(X[my_members, 0], X[my_members, 1], col + '.') : 畫出個別label的點
plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,markeredgecolor='k', markersize=14) : 畫出個別label的中心
最後秀出結果圖

![](https://github.com/kenny024241/machine-learning-python/blob/master/Clustering/ex2.png)


## (三)完整程式碼
Python source code:plot_mean_shift.py

https://scikit-learn.org/stable/_downloads/plot_mean_shift.py
```python
"""
=============================================
A demo of the mean-shift clustering algorithm
=============================================

Reference:

Dorin Comaniciu and Peter Meer, "Mean Shift: A robust approach toward
feature space analysis". IEEE Transactions on Pattern Analysis and
Machine Intelligence. 2002. pp. 603-619.

"""
print(__doc__)

import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs

# #############################################################################
# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]
X, _ = make_blobs(n_samples=10000, centers=centers, cluster_std=0.6)

# #############################################################################
# Compute clustering with MeanShift

# The following bandwidth can be automatically detected using
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)

# #############################################################################
# Plot result
import matplotlib.pyplot as plt
from itertools import cycle

plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
```
