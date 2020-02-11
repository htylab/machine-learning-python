# **Plot Hierarchical Clustering Dendrogram**
https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html#sphx-glr-auto-examples-cluster-plot-agglomerative-dendrogram-py

此範例使用AgglomerativeClustering(聚集聚類)和scipy中的樹狀圖法繪製Hierarchical Clustering Dendrogram(分層式聚類樹狀圖)

Hierarchical Clustering Dendrogram透過一種階層架構的方式，將資料分層反覆進行分群或聚集，以產生最後的樹狀結構，常見的方式有兩種：

* 如果採用聚集的方式，階層式分群法可由樹狀結構的底部開始，將資料或聚集逐次合併
* 如果採用分群的方式，則由樹狀結構的頂端開始，將聚集逐次分群
## (一)引入函式庫

* numpy : 產生陣列數值
* matplotlib.pyplot : 用來繪製影像
* scipy.cluster.hierarchy import dendrogram : 繪製樹狀圖
* sklearn.datasets import load_iris : 匯入資料集
* sklearn.cluster import AgglomerativeClustering : 匯入聚集聚類演算法

```python
import numpy as np

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
```
## (二)定義函式

* numpy.zeros() : 創建空矩陣
* numpy.column_stack() : 將一維陣列堆疊成二維陣列
* scipy.cluster.hierarchy.dendrogram(Z, p=30, truncate_mode=None, color_threshold=None, get_leaves=True, orientation='top', labels=None, count_sort=False, distance_sort=False, show_leaf_counts=True, no_plot=False, no_labels=False, leaf_font_size=None, leaf_rotation=None, leaf_label_func=None, show_contracted=False, link_color_func=None, ax=None, above_threshold_color='b')

Z : 決定群聚距離的定義方式 

1. 'single-linkage agglomerative algorithm'(單一連結聚合演算法)：群聚間的距離可以定義為不同群聚中最接近兩點間的距離
2. 'complete-linkage agglomerative algorithm'(完整連結聚合演算法)：群聚間的距離定義為不同群聚中最遠兩點間的距離
3. 'average-linkage agglomerative algorithm'(平均連結聚合演算法)：群聚間的距離則定義為不同群聚間各點與各點間距離總和的平均
4. 'Ward's method'(沃德法)：群聚間的距離定義為在將兩群合併後，各點到合併後的群中心的距離平方和
```python
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram 創建連結矩陣並繪製樹狀圖

    # create the counts of samples under each node 計算各節點下的樣本數量
    counts = np.zeros(model.children_.shape[0]) # 創建空矩陣空間儲存計數
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float) # 

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
```
## (三)繪製圖片

* uster.AgglomerativeClustering(n_clusters=2, affinity='euclidean', memory=None, connectivity=None, compute_full_tree='auto', linkage='ward', distance_threshold=None)

1. n_clusters : 要查找的群集數。如果distance_threshold不為None，則必須為None
2. affinity : 用於計算鏈接的度量
3. memory :暫存樹計算的輸出
4. connectivity : 連接矩陣
5. compute_full_tree : 在n_clusters處盡早停止構建樹。如果聚集的數量與樣本數相比不小的話，對於減少計算時間很有用。當有指定連接矩陣時，此選項才有作用。需要注意的是，當改變群集數量並使用暫存時，將樹完整計算完可能是有利於結果分群
6. linkage : 連結方式，與上面Z相同
7. distance_threshold : 設定閥值，鏈接距離等於高於該值時，群集將不會合併。
```python
iris = load_iris() # 匯入資料集
X = iris.data

# setting distance_threshold=0 ensures we compute the full tree.
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

model = model.fit(X)
plt.title('Hierarchical Clustering Dendrogram')
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode='level', p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()
```
![](https://github.com/sdgary56249128/machine-learning-python/blob/master/Clustering/sphx_glr_plot_agglomerative_dendrogram_001.png)
## (四)完整程式碼
https://scikit-learn.org/stable/_downloads/6c3126e55d97d68efdd8da229311ac00/plot_agglomerative_dendrogram.py
```python
import numpy as np

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


iris = load_iris()
X = iris.data

# setting distance_threshold=0 ensures we compute the full tree.
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

model = model.fit(X)
plt.title('Hierarchical Clustering Dendrogram')
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode='level', p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()
```
