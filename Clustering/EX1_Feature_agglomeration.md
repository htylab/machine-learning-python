# **範例一:Feature agglomeration**

https://scikit-learn.org/stable/auto_examples/cluster/plot_digits_agglomeration.html#sphx-glr-auto-examples-cluster-plot-digits-agglomeration-py

此範例是用FeatureAgglomeration來做特徵聚集

1. 利用 sklearn.datasets.load_digits() 來讀取內建資料庫
2. 利用 FeatureAgglomeration : 將相似特徵聚集並降維，來減少特徵數量，避免特徵過多的問題


## (一)引入函式庫

引入函式如下:

1. numpy : 產生陣列數值
2. matplotlib.pyplot : 用來繪製影像
3. sklearn import datasets, cluster : datasets : 用來匯入內建之手寫數字資料庫 ; cluster : 其內收集非監督clustering演算法
4. sklearn.feature_extraction.image import grid_to_graph : 定義資料的結構


```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets, cluster
from sklearn.feature_extraction.image import grid_to_graph
```

```python
# The digits dataset
digits = datasets.load_digits()
images = digits.images
```

使用 datasets.load_digits() 將資料存入， digits 為一個dict型別資料，我們可以用以下指令來看一下資料的內容。

```python
for key,value in digits.items() :
    try:
        print (key,value.shape)
    except:
        print (key)
```

| 顯示 | 說明 |
| -- | -- |
| ('images', (1797L, 8L, 8L))| 共有 1797 張影像，影像大小為 8x8 |
| ('data', (1797L, 64L)) | data 則是將8x8的矩陣攤平成64個元素之一維向量 |
| ('target_names', (10L,)) | 說明10種分類之對應 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] |
| DESCR | 資料之描述 |
| ('target', (1797L,))| 記錄1797張影像各自代表那一個數字 |

```python
X = np.reshape(images, (len(images), -1))
```
將1797x8x8的圖片拉長，變成1797x64

## (二)特徵聚集
```python
connectivity = grid_to_graph(*images[0].shape)

agglo = cluster.FeatureAgglomeration(connectivity=connectivity,
                                     n_clusters=32)
                                     
agglo.fit(X)
```
grid_to_graph : 做出像素連接的矩陣
FeatureAgglomeration : 將相似特徵聚集並降維，來減少特徵數量

```python
X_reduced = agglo.transform(X)

X_restored = agglo.inverse_transform(X_reduced)
```
transform : 根據上面 n_clusters 的值做轉換，得出[n_samples, n_features_new]新的特徵值
inverse_transform : 將其轉換回原本的特徵數(64)對應的特徵值

```python
images_restored = np.reshape(X_restored, images.shape)
plt.figure(1, figsize=(4, 3.5))
plt.clf()
```
plt.clf() : 保留figure但是清除內容，可以讓這figure重複使用

最後用下面程式碼將圖秀出來
```python
plt.subplots_adjust(left=.01, right=.99, bottom=.01, top=.91)
for i in range(4):
    plt.subplot(3, 4, i + 1)
    plt.imshow(images[i], cmap=plt.cm.gray, vmax=16, interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
    if i == 1:
        plt.title('Original data')
    plt.subplot(3, 4, 4 + i + 1)
    plt.imshow(images_restored[i], cmap=plt.cm.gray, vmax=16,
               interpolation='nearest')
    if i == 1:
        plt.title('Agglomerated data')
    plt.xticks(())
    plt.yticks(())

plt.subplot(3, 4, 10)
plt.imshow(np.reshape(agglo.labels_, images[0].shape),
           interpolation='nearest', cmap=plt.cm.nipy_spectral)
plt.xticks(())
plt.yticks(())
plt.title('Labels')
plt.show()
```

![](https://github.com/kenny024241/machine-learning-python/raw/master/Clustering/ex1.png)


## (三)完整程式碼
Python source code:plot_digits_agglomeration.py

https://scikit-learn.org/stable/_downloads/plot_digits_agglomeration.py

```python
print(__doc__)

# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets, cluster
from sklearn.feature_extraction.image import grid_to_graph

digits = datasets.load_digits()
images = digits.images
X = np.reshape(images, (len(images), -1))
connectivity = grid_to_graph(*images[0].shape)

agglo = cluster.FeatureAgglomeration(connectivity=connectivity,
                                     n_clusters=32)

agglo.fit(X)
X_reduced = agglo.transform(X)

X_restored = agglo.inverse_transform(X_reduced)
images_restored = np.reshape(X_restored, images.shape)
plt.figure(1, figsize=(4, 3.5))
plt.clf()
plt.subplots_adjust(left=.01, right=.99, bottom=.01, top=.91)
for i in range(4):
    plt.subplot(3, 4, i + 1)
    plt.imshow(images[i], cmap=plt.cm.gray, vmax=16, interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
    if i == 1:
        plt.title('Original data')
    plt.subplot(3, 4, 4 + i + 1)
    plt.imshow(images_restored[i], cmap=plt.cm.gray, vmax=16,
               interpolation='nearest')
    if i == 1:
        plt.title('Agglomerated data')
    plt.xticks(())
    plt.yticks(())

plt.subplot(3, 4, 10)
plt.imshow(np.reshape(agglo.labels_, images[0].shape),
           interpolation='nearest', cmap=plt.cm.nipy_spectral)
plt.xticks(())
plt.yticks(())
plt.title('Labels')
plt.show()
```


