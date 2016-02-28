# **範例十二:Spectral clustering for image segmentation**

http://scikit-learn.org/stable/auto_examples/cluster/plot_segmentation_toy.html

此範例是利用Spectral clustering來區別重疊的圓圈，將重疊的圓圈分為個體。

1. 建立一個100x100的影像包含四個不同半徑的圓
2. 透過```np.indices```改變影像顏色複雜度
3. 用```spectral_clustering```區分出各個不同區域特徵


## (一)引入函式庫
引入函式庫如下：
1. ```numpy```:產生陣列數值
2. ```matplotlib.pyplot```:用來繪製影像
3. ```sklearn.feature_extraction import image```:將每個像素的梯度關係圖像化
4. ```sklearn.cluster import spectral_clustering```:將影像正規化切割


```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering```


## (二)建立要被區分的重疊圓圈影像

* 產生一個大小為輸入值得矩陣(此範例為100x100)，其內部值為沿著座標方向遞增(如:0,1,...)的值。


```python
l = 100
x, y = np.indices((l, l))```

* 建立四個圓圈的圓心座標並給定座標值
* 給定四個圓圈的半徑長度
* 將圓心座標與半徑結合產生四個圓圈圖像

```python
center1 = (28, 24)
center2 = (40, 50)
center3 = (67, 58)
center4 = (24, 70)

radius1, radius2, radius3, radius4 = 16, 14, 15, 14

circle1 = (x - center1[0]) ** 2 + (y - center1[1]) ** 2 < radius1 ** 2
circle2 = (x - center2[0]) ** 2 + (y - center2[1]) ** 2 < radius2 ** 2
circle3 = (x - center3[0]) ** 2 + (y - center3[1]) ** 2 < radius3 ** 2
circle4 = (x - center4[0]) ** 2 + (y - center4[1]) ** 2 < radius4 ** 2
```
* 將上一段產生的四個圓圈影像合併為```img```使其成為一體的物件
* ```mask```為布林形式的```img```
* ```img```為浮點數形式的```img```
* 用亂數產生的方法將整張影像作亂數處理


```python
# 4 circles
img = circle1 + circle2 + circle3 + circle4
mask = img.astype(bool)
img = img.astype(float)

img += 1 + 0.2 * np.random.randn(*img.shape)```



接著將產生好的影像化為可使用```spectral_clustering```的影像

* ```image.img_to_graph``` 用來處理邊緣的權重與每個像速間的梯度關聯有關
* 用類似Voronoi Diagram演算法的概念來處理影像

```python
graph = image.img_to_graph(img, mask=mask)

graph.data = np.exp(-graph.data / graph.data.std())
```
最後用```spectral_clustering```將連在一起的部分切開，而```spectral_clustering```中的各項參數設定如下:
* ```graph```: 必須是一個矩陣且大小為nxn的形式
* ```n_clusters=4```: 需要提取出的群集數
* ```eigen_solver='arpack'```: 解特徵值的方式

開一張新影像```label_im```用來展示```spectral_clustering```切開後的分類結果

```python
labels = spectral_clustering(graph, n_clusters=4, eigen_solver='arpack')
label_im = -np.ones(mask.shape)
label_im[mask] = labels

plt.matshow(img)
plt.matshow(label_im)
```
![](http://scikit-learn.org/stable/_images/plot_segmentation_toy_001.png)
![](http://scikit-learn.org/stable/_images/plot_segmentation_toy_002.png)


## (三)完整程式碼
Python source code:plot_segmentation_toy.py

http://scikit-learn.org/stable/_downloads/plot_segmentation_toy.py


```python
print(__doc__)

# Authors:  Emmanuelle Gouillart <emmanuelle.gouillart@normalesup.org>
#           Gael Varoquaux <gael.varoquaux@normalesup.org>
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering

###############################################################################
l = 100
x, y = np.indices((l, l))

center1 = (28, 24)
center2 = (40, 50)
center3 = (67, 58)
center4 = (24, 70)

radius1, radius2, radius3, radius4 = 16, 14, 15, 14

circle1 = (x - center1[0]) ** 2 + (y - center1[1]) ** 2 < radius1 ** 2
circle2 = (x - center2[0]) ** 2 + (y - center2[1]) ** 2 < radius2 ** 2
circle3 = (x - center3[0]) ** 2 + (y - center3[1]) ** 2 < radius3 ** 2
circle4 = (x - center4[0]) ** 2 + (y - center4[1]) ** 2 < radius4 ** 2

###############################################################################
# 4 circles
img = circle1 + circle2 + circle3 + circle4
mask = img.astype(bool)
img = img.astype(float)

img += 1 + 0.2 * np.random.randn(*img.shape)

# Convert the image into a graph with the value of the gradient on the
# edges.
graph = image.img_to_graph(img, mask=mask)

# Take a decreasing function of the gradient: we take it weakly
# dependent from the gradient the segmentation is close to a voronoi
graph.data = np.exp(-graph.data / graph.data.std())

# Force the solver to be arpack, since amg is numerically
# unstable on this example
labels = spectral_clustering(graph, n_clusters=4, eigen_solver='arpack')
label_im = -np.ones(mask.shape)
label_im[mask] = labels

plt.matshow(img)
plt.matshow(label_im)

###############################################################################
# 2 circles
img = circle1 + circle2
mask = img.astype(bool)
img = img.astype(float)

img += 1 + 0.2 * np.random.randn(*img.shape)

graph = image.img_to_graph(img, mask=mask)
graph.data = np.exp(-graph.data / graph.data.std())

labels = spectral_clustering(graph, n_clusters=2, eigen_solver='arpack')
label_im = -np.ones(mask.shape)
label_im[mask] = labels

plt.matshow(img)
plt.matshow(label_im)

plt.show()```