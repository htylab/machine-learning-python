# 維基百科主要的特徵向量:Wikipedia principal eigenvector

https://scikit-learn.org/stable/auto_examples/applications/wikipedia_principal_eigenvector.html

強調圖中節點相對重要性的一種經典方法是計算鄰接矩陣的主要特徵向量，以便將每個特徵向量的分量值作為中心性分數分配給每個節點：

https://en.wikipedia.org/wiki/Eigenvector_centrality

在圖中的網頁和連結上，這些值被Google稱為```PageRank```分數，
本範例的目的為分析維基百科文章內部的鏈接圖，以根據此特徵向量中心性按相對重要性對文章進行排名。

計算主特徵向量的傳統方法是使用冪迭代方法：

https://en.wikipedia.org/wiki/Power_iteration

在這要感謝Martinsson的隨機SVD算法，才能夠在```scikit-learn```中實現計算。
此範例的數據是從```DBpedia```轉儲中獲取，```DBpedia``` 是一項從維基百科裡萃取結構化內容的專案計畫，
這些計畫所得的結構化資訊，也將放在網際網路中公開讓人取閱。

## (一)引入函式庫
引入函式庫如下：
1. ```bz2 import BZ2File```:用bzip2格式進行壓縮
2. ```import os```:調用操作系統命令查詢文件
3. ```import datetime```:計算日期
4. ```import time```:計算時間
5. ```numpy as np```:產生陣列數值
6. ```from scipy import sparse```:產生稀疏矩陣
7. ```from joblib import Memory```:將資料進行暫存
8. ```sklearn.decomposition import randomized_svd```:計算分解的隨機SVD
9. ```urllib.request import urlopen```:開啟URL網址

## (二)載入壓縮檔

```os.path.exists```將會判斷是否存在以下兩個壓縮檔(bzip2格式):
* ```redirects_en.nt.bz2```
* ```page_links_en.nt.bz2```

```python
redirects_url = "http://downloads.dbpedia.org/3.5.1/en/redirects_en.nt.bz2"
redirects_filename = redirects_url.rsplit("/", 1)[1]

page_links_url = "http://downloads.dbpedia.org/3.5.1/en/page_links_en.nt.bz2"
page_links_filename = page_links_url.rsplit("/", 1)[1]

resources = [
    (redirects_url, redirects_filename),
    (page_links_url, page_links_filename),
]

for url, filename in resources:
    if not os.path.exists(filename):
        print("Downloading data from '%s', please wait..." % url)
        opener = urlopen(url)
        open(filename, 'wb').write(opener.read())
        print()
```
## (三)函數設置


```Transitive Closure```中文譯作遞移閉包，用來紀錄由一點能不能走到另一點的關係，如果能走到，則兩點之間以邊相連。

```python
def get_redirects(redirects_filename):
    """分析重定向後，建立出遞移閉包的圖"""
    redirects = {}
    print("Parsing the NT redirect file")
    for l, line in enumerate(BZ2File(redirects_filename)):
        split = line.split()
        if len(split) != 4:
            print("ignoring malformed line: " + line)
            continue
        redirects[short_name(split[0])] = short_name(split[2])
        if l % 1000000 == 0:
            print("[%s] line: %08d" % (datetime.now().isoformat(), l))

    # 計算遞移閉包
    print("Computing the transitive closure of the redirect relation")
    for l, source in enumerate(redirects.keys()):
        transitive_target = None
        target = redirects[source]
        seen = {source}
        while True:
            transitive_target = target
            target = redirects.get(target)
            if target is None or target in seen:
                break
            seen.add(target)
        redirects[source] = transitive_target
        if l % 1000000 == 0:
            print("[%s] line: %08d" % (datetime.now().isoformat(), l))

    return redirects
```
## (四)鄰接矩陣

首先，先介紹何謂```稀疏矩陣```，對於一個矩陣而言，
若數值為零的元素遠遠多於非零元素的個數，且非零元素分佈沒有規律時，這樣的矩陣被稱作稀疏矩陣。
此函數提取鄰接的圖作為稀疏矩陣(sparse matrix)，
並使用稀疏矩陣作為鄰接矩陣。

其中運用scipy的```sparse.lil_matrix```建立一個稀疏矩陣。
```python
 X = sparse.lil_matrix((len(index_map), len(index_map)), dtype=np.float32)
 ```
 定義鄰接矩陣之函數:
 
```python
def get_adjacency_matrix(redirects_filename, page_links_filename, limit=None):
    """Extract the adjacency graph as a scipy sparse matrix

    Redirects are resolved first.

    Returns X, the scipy sparse adjacency matrix, redirects as python
    dict from article names to article names and index_map a python dict
    from article names to python int (article indexes).
    """

    print("Computing the redirect map")
    redirects = get_redirects(redirects_filename)

    print("Computing the integer index map")
    index_map = dict()
    links = list()
    for l, line in enumerate(BZ2File(page_links_filename)):
        split = line.split()
        if len(split) != 4:
            print("ignoring malformed line: " + line)
            continue
        i = index(redirects, index_map, short_name(split[0]))
        j = index(redirects, index_map, short_name(split[2]))
        links.append((i, j))
        if l % 1000000 == 0:
            print("[%s] line: %08d" % (datetime.now().isoformat(), l))

        if limit is not None and l >= limit - 1:
            break

    print("Computing the adjacency matrix")
    X = sparse.lil_matrix((len(index_map), len(index_map)), dtype=np.float32)
    for i, j in links:
        X[i, j] = 1.0
    del links
    print("Converting to CSR representation")
    X = X.tocsr()
    print("CSR conversion done")
    return X, redirects, index_map
```
其中回傳三個值:
* X：計算出來的稀疏矩陣。
* redirects：python 的字典型態，存取文章名稱。
* index_map：python 的字典型態，存取文章名稱及索引。

為了能夠在RAM中工作，5M個連結後停止。

```python
X, redirects, index_map = get_adjacency_matrix(
    redirects_filename, page_links_filename, limit=5000000)
names = {i: name for name, i in index_map.items()}
```
## (五)計算奇異向量


使用```randomized_svd```計算SVD(奇異值分解)
```python
print("Computing the principal singular vectors using randomized_svd")
t0 = time()
U, s, V = randomized_svd(X, 5, n_iter=3)
print("done in %0.3fs" % (time() - t0))

# 印出與維基百科相關的強元件的名稱
# 主奇異向量應與最高特徵向量相似
print("Top wikipedia pages according to principal singular vectors")
pprint([names[i] for i in np.abs(U.T[0]).argsort()[-10:]])
pprint([names[i] for i in np.abs(V[0]).argsort()[-10:]])
```
Output：


![](https://github.com/JENNSHIUAN/machine-learning-python/blob/master/application/wikipedia_principal_eigenvector/wikipedia_fig1.JPG) 




## (六)計算主要特徵向量之分數

定義中心性分數(centrality scores)之函數，用Power 迭代法計算主要特徵向量。
這個方法也適用於知名的```Google PageRank```上，並實施於NetworkX project(BSD授權條款)
其版權：
* Aric Hagberg <hagberg@lanl.gov>
* Dan Schult <dschult@colgate.edu>
* Pieter Swart <swart@lanl.gov>

```python
def centrality_scores(X, alpha=0.85, max_iter=100, tol=1e-10):

    n = X.shape[0]
    X = X.copy()
    incoming_counts = np.asarray(X.sum(axis=1)).ravel()

    print("Normalizing the graph")
    for i in incoming_counts.nonzero()[0]:
        X.data[X.indptr[i]:X.indptr[i + 1]] *= 1.0 / incoming_counts[i]
    dangle = np.asarray(np.where(np.isclose(X.sum(axis=1), 0),
                                 1.0 / n, 0)).ravel()

    scores = np.full(n, 1. / n, dtype=np.float32)  # initial guess
    for i in range(max_iter):
        print("power iteration #%d" % i)
        prev_scores = scores
        scores = (alpha * (scores * X + np.dot(dangle, prev_scores))
                  + (1 - alpha) * prev_scores.sum() / n)
        # check convergence: normalized l_inf norm
        scores_max = np.abs(scores).max()
        if scores_max == 0.0:
            scores_max = 1.0
        err = np.abs(scores - prev_scores).max() / scores_max
        print("error: %0.6f" % err)
        if err < n * tol:
            return scores

    return scores

print("Computing principal eigenvector score using a power iteration method")
t0 = time()
scores = centrality_scores(X, max_iter=100)
print("done in %0.3fs" % (time() - t0))
pprint([names[i] for i in np.abs(scores).argsort()[-10:]])
```


Output：


![](https://github.com/JENNSHIUAN/machine-learning-python/blob/master/application/wikipedia_principal_eigenvector/wikipedia_fig2.JPG) 

## (七)完整程式碼
Python source code:wikipedia_principal_eigenvector.py

https://scikit-learn.org/stable/_downloads/637afdd681404c733540858401aadf5c/wikipedia_principal_eigenvector.py
```python
# Author: Olivier Grisel <olivier.grisel@ensta.org>
# License: BSD 3 clause

from bz2 import BZ2File
import os
from datetime import datetime
from pprint import pprint
from time import time

import numpy as np

from scipy import sparse

from joblib import Memory

from sklearn.decomposition import randomized_svd
from urllib.request import urlopen


print(__doc__)

# #############################################################################
# Where to download the data, if not already on disk
redirects_url = "http://downloads.dbpedia.org/3.5.1/en/redirects_en.nt.bz2"
redirects_filename = redirects_url.rsplit("/", 1)[1]

page_links_url = "http://downloads.dbpedia.org/3.5.1/en/page_links_en.nt.bz2"
page_links_filename = page_links_url.rsplit("/", 1)[1]

resources = [
    (redirects_url, redirects_filename),
    (page_links_url, page_links_filename),
]

for url, filename in resources:
    if not os.path.exists(filename):
        print("Downloading data from '%s', please wait..." % url)
        opener = urlopen(url)
        open(filename, 'wb').write(opener.read())
        print()


# #############################################################################
# 讀取重定向檔案

memory = Memory(cachedir=".")


def index(redirects, index_map, k):
    """重定向之後，找到文章名稱的索引"""
    k = redirects.get(k, k)
    return index_map.setdefault(k, len(index_map))


DBPEDIA_RESOURCE_PREFIX_LEN = len("http://dbpedia.org/resource/")
SHORTNAME_SLICE = slice(DBPEDIA_RESOURCE_PREFIX_LEN + 1, -1)


def short_name(nt_uri):
    """移除 < and > URI 記號以及普遍 URI 開頭"""
    return nt_uri[SHORTNAME_SLICE]


def get_redirects(redirects_filename):
    """分析重定向後，建立出遞移閉包的圖"""
    redirects = {}
    print("Parsing the NT redirect file")
    for l, line in enumerate(BZ2File(redirects_filename)):
        split = line.split()
        if len(split) != 4:
            print("ignoring malformed line: " + line)
            continue
        redirects[short_name(split[0])] = short_name(split[2])
        if l % 1000000 == 0:
            print("[%s] line: %08d" % (datetime.now().isoformat(), l))

    # 計算遞移閉包
    print("Computing the transitive closure of the redirect relation")
    for l, source in enumerate(redirects.keys()):
        transitive_target = None
        target = redirects[source]
        seen = {source}
        while True:
            transitive_target = target
            target = redirects.get(target)
            if target is None or target in seen:
                break
            seen.add(target)
        redirects[source] = transitive_target
        if l % 1000000 == 0:
            print("[%s] line: %08d" % (datetime.now().isoformat(), l))

    return redirects


# disabling joblib as the pickling of large dicts seems much too slow
#@memory.cache
def get_adjacency_matrix(redirects_filename, page_links_filename, limit=None):
    """Extract the adjacency graph as a scipy sparse matrix

    Redirects are resolved first.

    Returns X, the scipy sparse adjacency matrix, redirects as python
    dict from article names to article names and index_map a python dict
    from article names to python int (article indexes).
    """

    print("Computing the redirect map")
    redirects = get_redirects(redirects_filename)

    print("Computing the integer index map")
    index_map = dict()
    links = list()
    for l, line in enumerate(BZ2File(page_links_filename)):
        split = line.split()
        if len(split) != 4:
            print("ignoring malformed line: " + line)
            continue
        i = index(redirects, index_map, short_name(split[0]))
        j = index(redirects, index_map, short_name(split[2]))
        links.append((i, j))
        if l % 1000000 == 0:
            print("[%s] line: %08d" % (datetime.now().isoformat(), l))

        if limit is not None and l >= limit - 1:
            break

    print("Computing the adjacency matrix")
    X = sparse.lil_matrix((len(index_map), len(index_map)), dtype=np.float32)
    for i, j in links:
        X[i, j] = 1.0
    del links
    print("Converting to CSR representation")
    X = X.tocsr()
    print("CSR conversion done")
    return X, redirects, index_map


# 為了能夠在RAM中工作，5M個連結後停止。
X, redirects, index_map = get_adjacency_matrix(
    redirects_filename, page_links_filename, limit=5000000)
names = {i: name for name, i in index_map.items()}

print("Computing the principal singular vectors using randomized_svd")
t0 = time()
U, s, V = randomized_svd(X, 5, n_iter=3)
print("done in %0.3fs" % (time() - t0))

# 印出與維基百科相關的強元件的名稱
# 主奇異向量應與最高特徵向量相似
print("Top wikipedia pages according to principal singular vectors")
pprint([names[i] for i in np.abs(U.T[0]).argsort()[-10:]])
pprint([names[i] for i in np.abs(V[0]).argsort()[-10:]])


def centrality_scores(X, alpha=0.85, max_iter=100, tol=1e-10):
    """Power iteration computation of the principal eigenvector

    This method is also known as Google PageRank and the implementation
    is based on the one from the NetworkX project (BSD licensed too)
    with copyrights by:

      Aric Hagberg <hagberg@lanl.gov>
      Dan Schult <dschult@colgate.edu>
      Pieter Swart <swart@lanl.gov>
    """
    n = X.shape[0]
    X = X.copy()
    incoming_counts = np.asarray(X.sum(axis=1)).ravel()

    print("Normalizing the graph")
    for i in incoming_counts.nonzero()[0]:
        X.data[X.indptr[i]:X.indptr[i + 1]] *= 1.0 / incoming_counts[i]
    dangle = np.asarray(np.where(np.isclose(X.sum(axis=1), 0),
                                 1.0 / n, 0)).ravel()

    scores = np.full(n, 1. / n, dtype=np.float32)  # initial guess
    for i in range(max_iter):
        print("power iteration #%d" % i)
        prev_scores = scores
        scores = (alpha * (scores * X + np.dot(dangle, prev_scores))
                  + (1 - alpha) * prev_scores.sum() / n)
        # check convergence: normalized l_inf norm
        scores_max = np.abs(scores).max()
        if scores_max == 0.0:
            scores_max = 1.0
        err = np.abs(scores - prev_scores).max() / scores_max
        print("error: %0.6f" % err)
        if err < n * tol:
            return scores

    return scores

print("Computing principal eigenvector score using a power iteration method")
t0 = time()
scores = centrality_scores(X, max_iter=100)
print("done in %0.3fs" % (time() - t0))
pprint([names[i] for i in np.abs(scores).argsort()[-10:]])
```
