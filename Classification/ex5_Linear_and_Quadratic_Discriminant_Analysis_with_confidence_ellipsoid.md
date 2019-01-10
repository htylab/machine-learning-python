
##分類法/範例五：Linear and Quadratic Discriminant Analysis with confidence ellipsoid
## 線性判別以及二次判別的比較

http://scikit-learn.org/stable/auto_examples/classification/plot_lda_qda.html



## (一)資料產生function

這個範例引入的套件，主要特點在：
1. `scipy.linalg`:線性代數相關函式，這裏主要使用到linalg.eigh 特徵值相關問題
2. `matplotlib.colors`: 用來處理繪圖時的色彩分佈
3. `LinearDiscriminantAnalysis`:線性判別演算法
4. `QuadraticDiscriminantAnalysis`:二次判別演算法


```python
%matplotlib inline
from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
```

接下來是設定一個線性變化的colormap，`LinearSegmentedColormap(name, segmentdata)` 預設會傳回一個256個值的數值顏色對應關係。用一個具備有三個項目的dict變數`segmentdata`來設定。以`'red': [(0, 1, 1), (1, 0.7, 0.7)]`來解釋，就是我們希望數值由0到1的過程，紅色通道將由1線性變化至0.7。


```python
cmap = colors.LinearSegmentedColormap(
    'red_blue_classes',
    {'red': [(0, 1, 1), (1, 0.7, 0.7)],
     'green': [(0, 0.7, 0.7), (1, 0.7, 0.7)],
     'blue': [(0, 0.7, 0.7), (1, 1, 1)]})
plt.cm.register_cmap(cmap=cmap)
```

我們可以用以下程式碼來觀察。當輸入數值為`np.arange(0,1.1,0.1)`也就是0,0.1...,1.0 時RGB數值的變化情形。


```python
values = np.arange(0,1.1,0.1)
cmap_values = mpl.cm.get_cmap('red_blue_classes')(values)
import pandas as pd
pd.set_option('precision',2)
df=pd.DataFrame(np.hstack((values.reshape(11,1),cmap_values)))
df.columns = ['Value', 'R', 'G', 'B', 'Alpha']
print(df)
```

        Value    R    G    B  Alpha
    0     0.0  1.0  0.7  0.7      1
    1     0.1  1.0  0.7  0.7      1
    2     0.2  0.9  0.7  0.8      1
    3     0.3  0.9  0.7  0.8      1
    4     0.4  0.9  0.7  0.8      1
    5     0.5  0.8  0.7  0.9      1
    6     0.6  0.8  0.7  0.9      1
    7     0.7  0.8  0.7  0.9      1
    8     0.8  0.8  0.7  0.9      1
    9     0.9  0.7  0.7  1.0      1
    10    1.0  0.7  0.7  1.0      1


接著我們產生兩組資料, 每組資料有 600筆資料，2個特徵 `X: 600x2`以及2個類別 `y:600 (前300個元素為0，餘下為1)`：
1. `dataset_fixed_cov()`:2個類別的特徵具備有相同共變數(covariance)
2. `dataset_fixed_cov()`:2個類別的特徵具備有不同之共變數
差異落在X資料的產生`np.dot(np.random.randn(n, dim), C)` 與 `np.dot(np.random.randn(n, dim), C.T)`的不同。`np.dot(np.random.randn(n, dim), C)`會產生`300x2`之矩陣，其亂數產生的範圍可交由`C`矩陣來控制。在`dataset_fixed_cov()`中，前後300筆資料產生之範圍皆由`C`來調控。我們可以在最下方的結果圖示看到上排影像(相同共變數)的資料分佈無論是紅色(代表類別1)以及藍色(代表類別2)其分佈形狀相似。而下排影像(不同共變數)，分佈形狀則不同。圖示中，橫軸及縱軸分別表示第一及第二個特徵，讀者可以試著將 `0.83`這個數字減少或是將`C.T`改成`C`，看看最後結果圖形有了什麼改變？


```python
def dataset_fixed_cov():
    '''Generate 2 Gaussians samples with the same covariance matrix'''
    n, dim = 300, 2
    np.random.seed(0)
    C = np.array([[0., -0.23], [0.83, .23]])
    X = np.r_[np.dot(np.random.randn(n, dim), C),
              np.dot(np.random.randn(n, dim), C) + np.array([1, 1])] #利用 + np.array([1, 1]) 產生類別間的差異
    y = np.hstack((np.zeros(n), np.ones(n))) #產生300個零及300個1並連接起來
    return X, y

def dataset_cov():
    '''Generate 2 Gaussians samples with different covariance matrices'''
    n, dim = 300, 2
    np.random.seed(0)
    C = np.array([[0., -1.], [2.5, .7]]) * 2.
    X = np.r_[np.dot(np.random.randn(n, dim), C),
              np.dot(np.random.randn(n, dim), C.T) + np.array([1, 4])]
    y = np.hstack((np.zeros(n), np.ones(n)))
    return X, y
```

## (二)繪圖函式

1. 找出 True Positive及False Negative 之辨認點
2. 以紅色及藍色分別表示分類為 0及1的資料點，而以深紅跟深藍來表示誤判資料
3. 以`lda.predict_proba()`畫出分類的機率分佈(請參考範例三)

(為了方便在ipython notebook環境下顯示，下面函式有經過微調)


```python
def plot_data(lda, X, y, y_pred, fig_index):
    splot = plt.subplot(2, 2, fig_index)
    if fig_index == 1:
        plt.title('Linear Discriminant Analysis',fontsize=28)
        plt.ylabel('Data with fixed covariance',fontsize=28)
    elif fig_index == 2:
        plt.title('Quadratic Discriminant Analysis',fontsize=28)
    elif fig_index == 3:
        plt.ylabel('Data with varying covariances',fontsize=28)

    # 步驟一：找出 True Positive及False postive 之辨認點

    tp = (y == y_pred)  # True Positive
    tp0, tp1 = tp[y == 0], tp[y == 1] #tp0 代表分類為0且列為 True Positive之資料點
    X0, X1 = X[y == 0], X[y == 1]
    X0_tp, X0_fp = X0[tp0], X0[~tp0]
    X1_tp, X1_fp = X1[tp1], X1[~tp1]

    # 步驟二：以紅藍來畫出分類資料，以深紅跟深藍來表示誤判資料

    # class 0: dots
    plt.plot(X0_tp[:, 0], X0_tp[:, 1], 'o', color='red')
    plt.plot(X0_fp[:, 0], X0_fp[:, 1], '.', color='#990000')  # dark red

    # class 1: dots
    plt.plot(X1_tp[:, 0], X1_tp[:, 1], 'o', color='blue')
    plt.plot(X1_fp[:, 0], X1_fp[:, 1], '.', color='#000099')  # dark blue

    #步驟三：畫出分類的機率分佈(請參考範例三)
    # class 0 and 1 : areas
    nx, ny = 200, 100
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                         np.linspace(y_min, y_max, ny))
    Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z[:, 1].reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap='red_blue_classes',
                   norm=colors.Normalize(0., 1.))
    plt.contour(xx, yy, Z, [0.5], linewidths=2., colors='k')

    # means
    plt.plot(lda.means_[0][0], lda.means_[0][1],
             'o', color='black', markersize=10)
    plt.plot(lda.means_[1][0], lda.means_[1][1],
             'o', color='black', markersize=10)

    return splot
```


```python
def plot_ellipse(splot, mean, cov, color):
    v, w = linalg.eigh(cov)
    u = w[0] / linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    angle = 180 * angle / np.pi  # convert to degrees
    # filled Gaussian at 2 standard deviation
    ell = mpl.patches.Ellipse(mean, 2 * v[0] ** 0.5, 2 * v[1] ** 0.5,
                              180 + angle, color=color)
    ell.set_clip_box(splot.bbox)
    ell.set_alpha(0.5)
    splot.add_artist(ell)
    splot.set_xticks(())
    splot.set_yticks(())
```

## (三)測試資料並繪圖


```python
def plot_lda_cov(lda, splot):
    plot_ellipse(splot, lda.means_[0], lda.covariance_, 'red')
    plot_ellipse(splot, lda.means_[1], lda.covariance_, 'blue')


def plot_qda_cov(qda, splot):
    plot_ellipse(splot, qda.means_[0], qda.covariances_[0], 'red')
    plot_ellipse(splot, qda.means_[1], qda.covariances_[1], 'blue')

###############################################################################
figure = plt.figure(figsize=(30,20), dpi=300)
for i, (X, y) in enumerate([dataset_fixed_cov(), dataset_cov()]):
    # Linear Discriminant Analysis
    lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
    y_pred = lda.fit(X, y).predict(X)
    splot = plot_data(lda, X, y, y_pred, fig_index=2 * i + 1)
    plot_lda_cov(lda, splot)
    plt.axis('tight')

    # Quadratic Discriminant Analysis
    qda = QuadraticDiscriminantAnalysis(store_covariances=True)
    y_pred = qda.fit(X, y).predict(X)
    splot = plot_data(qda, X, y, y_pred, fig_index=2 * i + 2)
    plot_qda_cov(qda, splot)
    plt.axis('tight')
plt.suptitle('Linear Discriminant Analysis vs Quadratic Discriminant Analysis',fontsize=28)
plt.show()
```


![png](images/ex5_output_13_0.png)


Python source code: plot_lda_qda.py

http://scikit-learn.org/stable/_downloads/plot_lda_qda.py


```python
print(__doc__)

from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

###############################################################################
# colormap
cmap = colors.LinearSegmentedColormap(
    'red_blue_classes',
    {'red': [(0, 1, 1), (1, 0.7, 0.7)],
     'green': [(0, 0.7, 0.7), (1, 0.7, 0.7)],
     'blue': [(0, 0.7, 0.7), (1, 1, 1)]})
plt.cm.register_cmap(cmap=cmap)


###############################################################################
# generate datasets
def dataset_fixed_cov():
    '''Generate 2 Gaussians samples with the same covariance matrix'''
    n, dim = 300, 2
    np.random.seed(0)
    C = np.array([[0., -0.23], [0.83, .23]])
    X = np.r_[np.dot(np.random.randn(n, dim), C),
              np.dot(np.random.randn(n, dim), C) + np.array([1, 1])]
    y = np.hstack((np.zeros(n), np.ones(n)))
    return X, y


def dataset_cov():
    '''Generate 2 Gaussians samples with different covariance matrices'''
    n, dim = 300, 2
    np.random.seed(0)
    C = np.array([[0., -1.], [2.5, .7]]) * 2.
    X = np.r_[np.dot(np.random.randn(n, dim), C),
              np.dot(np.random.randn(n, dim), C.T) + np.array([1, 4])]
    y = np.hstack((np.zeros(n), np.ones(n)))
    return X, y


###############################################################################
# plot functions
def plot_data(lda, X, y, y_pred, fig_index):
    splot = plt.subplot(2, 2, fig_index)
    if fig_index == 1:
        plt.title('Linear Discriminant Analysis')
        plt.ylabel('Data with fixed covariance')
    elif fig_index == 2:
        plt.title('Quadratic Discriminant Analysis')
    elif fig_index == 3:
        plt.ylabel('Data with varying covariances')

    tp = (y == y_pred)  # True Positive
    tp0, tp1 = tp[y == 0], tp[y == 1]
    X0, X1 = X[y == 0], X[y == 1]
    X0_tp, X0_fp = X0[tp0], X0[~tp0]
    X1_tp, X1_fp = X1[tp1], X1[~tp1]

    # class 0: dots
    plt.plot(X0_tp[:, 0], X0_tp[:, 1], 'o', color='red')
    plt.plot(X0_fp[:, 0], X0_fp[:, 1], '.', color='#990000')  # dark red

    # class 1: dots
    plt.plot(X1_tp[:, 0], X1_tp[:, 1], 'o', color='blue')
    plt.plot(X1_fp[:, 0], X1_fp[:, 1], '.', color='#000099')  # dark blue

    # class 0 and 1 : areas
    nx, ny = 200, 100
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                         np.linspace(y_min, y_max, ny))
    Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z[:, 1].reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap='red_blue_classes',
                   norm=colors.Normalize(0., 1.))
    plt.contour(xx, yy, Z, [0.5], linewidths=2., colors='k')

    # means
    plt.plot(lda.means_[0][0], lda.means_[0][1],
             'o', color='black', markersize=10)
    plt.plot(lda.means_[1][0], lda.means_[1][1],
             'o', color='black', markersize=10)

    return splot


def plot_ellipse(splot, mean, cov, color):
    v, w = linalg.eigh(cov)
    u = w[0] / linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    angle = 180 * angle / np.pi  # convert to degrees
    # filled Gaussian at 2 standard deviation
    ell = mpl.patches.Ellipse(mean, 2 * v[0] ** 0.5, 2 * v[1] ** 0.5,
                              180 + angle, color=color)
    ell.set_clip_box(splot.bbox)
    ell.set_alpha(0.5)
    splot.add_artist(ell)
    splot.set_xticks(())
    splot.set_yticks(())


def plot_lda_cov(lda, splot):
    plot_ellipse(splot, lda.means_[0], lda.covariance_, 'red')
    plot_ellipse(splot, lda.means_[1], lda.covariance_, 'blue')


def plot_qda_cov(qda, splot):
    plot_ellipse(splot, qda.means_[0], qda.covariances_[0], 'red')
    plot_ellipse(splot, qda.means_[1], qda.covariances_[1], 'blue')

###############################################################################
for i, (X, y) in enumerate([dataset_fixed_cov(), dataset_cov()]):
    # Linear Discriminant Analysis
    lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
    y_pred = lda.fit(X, y).predict(X)
    splot = plot_data(lda, X, y, y_pred, fig_index=2 * i + 1)
    plot_lda_cov(lda, splot)
    plt.axis('tight')

    # Quadratic Discriminant Analysis
    qda = QuadraticDiscriminantAnalysis(store_covariances=True)
    y_pred = qda.fit(X, y).predict(X)
    splot = plot_data(qda, X, y, y_pred, fig_index=2 * i + 2)
    plot_qda_cov(qda, splot)
    plt.axis('tight')
plt.suptitle('Linear Discriminant Analysis vs Quadratic Discriminant Analysis')
plt.show()
```


```python

```

## (一)引入函式庫

引入函式如下:

1. numpy : 產生陣列數值
2. matplotlib.pyplot : 用來繪製影像
3. sklearn import datasets, cluster : datasets : 用來繪入內建之手寫數字資料庫 ; cluster : 其內收集非監督clustering演算法
4. sklearn.feature_extraction.image import grid_to_graph : 定義資料的結構

