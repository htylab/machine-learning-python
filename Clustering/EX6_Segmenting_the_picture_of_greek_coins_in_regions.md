# **範例六:Segmenting the picture of greek coins in regions**

https://scikit-learn.org/stable/auto_examples/cluster/plot_coin_segmentation.html#sphx-glr-auto-examples-cluster-plot-coin-segmentation-py

此範例用Spectral Clustering以分割圖中的硬幣

1. 利用coins()匯入圖片
2. 利用spectral_clustering做切割
3. 最後將結果可視化

## (一)引入函式庫

引入函式如下:

1. time : 提供各種與時間相關函數
2. numpy : 產生陣列數值
3. scipy.ndimage.filters import gaussian_filter : 做gaussian filter
4. matplotlib.pyplot : 用來繪製影像
5. skimage.data import coins : 匯入龐貝城的希臘硬幣
6. skimage.transform import rescale : 用來縮放圖片
7. sklearn.feature_extraction import image : 將每個像素的梯度關係圖像化
8. sklearn.cluster import spectral_clustering : 將影像正規化切割

```python
import time
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from skimage.data import coins
from skimage.transform import rescale

from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering

orig_coins = coins()

smoothened_coins = gaussian_filter(orig_coins, sigma=2)
rescaled_coins = rescale(smoothened_coins, 0.2, mode="reflect")
```
coins() : 匯入一張303x384的影像
用 rescale 將圖片 resize 成原圖的20%(61x77)來加快處理，並根據 mode 選擇 padding 方式


## (二)Clustering
```python
# Convert the image into a graph with the value of the gradient on the edges.
graph = image.img_to_graph(rescaled_coins)
```
img_to_graph : 用來處理邊緣的權重與每個像速間的梯度關聯

```python
beta = 10
eps = 1e-6
graph.data = np.exp(-beta * graph.data / graph.data.std()) + eps
```
beta越小，實際圖像會分割的越獨立，當 beta = 1 時，會類似Voronoi Diagram演算法的概念

```python
N_REGIONS = 24

for assign_labels in ('kmeans', 'discretize'):
    t0 = time.time()
    labels = spectral_clustering(graph, n_clusters=N_REGIONS,
                                 assign_labels=assign_labels, random_state=42)
    t1 = time.time()
    labels = labels.reshape(rescaled_coins.shape)

    plt.figure(figsize=(5, 5))
    plt.imshow(rescaled_coins, cmap=plt.cm.gray)
    for l in range(N_REGIONS):
        plt.contour(labels == l,
                    colors=[plt.cm.nipy_spectral(l / float(N_REGIONS))])
    plt.xticks(())
    plt.yticks(())
    title = 'Spectral clustering: %s, %.2fs' % (assign_labels, (t1 - t0))
    print(title)
    plt.title(title)
plt.show()
```
用spectral_clustering將連在一起的部分切開，而spectral_clustering中的各項參數設定如下:
* graph: 必須是一個矩陣且大小為nxn的形式
* n_clusters: 需要提取出的群集數
* random_state: 偽隨機數產生器，用於初始化特徵向量分解計算
* assign_labels:選擇assign label的方法(kmeans or discretize)

用plt.contour畫出等高線，同個label會被框在同個圈內


![](https://github.com/kenny024241/machine-learning-python/blob/master/Clustering/ex6-1.png)  ![](https://github.com/kenny024241/machine-learning-python/blob/master/Clustering/ex6-2.png)


## (三)完整程式碼
Python source code:plot_coin_segmentation.py

https://scikit-learn.org/stable/_downloads/plot_coin_segmentation.py
```python
"""
================================================
Segmenting the picture of greek coins in regions
================================================

This example uses :ref:`spectral_clustering` on a graph created from
voxel-to-voxel difference on an image to break this image into multiple
partly-homogeneous regions.

This procedure (spectral clustering on an image) is an efficient
approximate solution for finding normalized graph cuts.

There are two options to assign labels:

* with 'kmeans' spectral clustering will cluster samples in the embedding space
  using a kmeans algorithm
* whereas 'discrete' will iteratively search for the closest partition
  space to the embedding space.
"""
print(__doc__)

# Author: Gael Varoquaux <gael.varoquaux@normalesup.org>, Brian Cheung
# License: BSD 3 clause

import time

import numpy as np
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from skimage.data import coins
from skimage.transform import rescale

from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering


# load the coins as a numpy array
orig_coins = coins()

# Resize it to 20% of the original size to speed up the processing
# Applying a Gaussian filter for smoothing prior to down-scaling
# reduces aliasing artifacts.
smoothened_coins = gaussian_filter(orig_coins, sigma=2)
rescaled_coins = rescale(smoothened_coins, 0.2, mode="reflect")

# Convert the image into a graph with the value of the gradient on the
# edges.
graph = image.img_to_graph(rescaled_coins)

# Take a decreasing function of the gradient: an exponential
# The smaller beta is, the more independent the segmentation is of the
# actual image. For beta=1, the segmentation is close to a voronoi
beta = 10
eps = 1e-6
graph.data = np.exp(-beta * graph.data / graph.data.std()) + eps

# Apply spectral clustering (this step goes much faster if you have pyamg
# installed)
N_REGIONS = 24

#############################################################################
# Visualize the resulting regions

for assign_labels in ('kmeans', 'discretize'):
    t0 = time.time()
    labels = spectral_clustering(graph, n_clusters=N_REGIONS,
                                 assign_labels=assign_labels, random_state=42)
    t1 = time.time()
    labels = labels.reshape(rescaled_coins.shape)

    plt.figure(figsize=(5, 5))
    plt.imshow(rescaled_coins, cmap=plt.cm.gray)
    for l in range(N_REGIONS):
        plt.contour(labels == l,
                    colors=[plt.cm.nipy_spectral(l / float(N_REGIONS))])
    plt.xticks(())
    plt.yticks(())
    title = 'Spectral clustering: %s, %.2fs' % (assign_labels, (t1 - t0))
    print(title)
    plt.title(title)
plt.show()
```

