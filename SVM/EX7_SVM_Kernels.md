# **範例七: SVM-Kernels**

https://scikit-learn.org/stable/auto_examples/svm/plot_svm_kernels.html#sphx-glr-auto-examples-svm-plot-svm-kernels-py

此範例展示三種不同 SVM Kernel 所呈現的結果，其中 Polynomial 及 RBF 常用來處理非線性之分類

1. Linear
2. Polynomial
3. RBF(Radio Basis Function)

## (一)引入函式庫

引入函式如下:

1. `numpy` : 產生陣列數值
2. `matplotlib.pyplot` : 用來繪製影像
3. `sklearn.svm` : SVM 支持向量機之演算法物件

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
```
```python
# Our dataset and targets
X = np.c_[(.4, -.7),
          (-1.5, -1),
          (-1.4, -.9),
          (-1.3, -1.2),
          (-1.1, -.2),
          (-1.2, -.4),
          (-.5, 1.2),
          (-1.5, 2.1),
          (1, 1),
          # --
          (1.3, .8),
          (1.2, .5),
          (.2, -2),
          (.5, -2.4),
          (.2, -2.3),
          (0, -2.7),
          (1.3, 2.1)].T
Y = [0] * 8 + [1] * 8
```
首先，第一步先自行產生出資料點X及目標值Y

## (二)Different SVM Kernels
```python
# figure number
fignum = 1

# fit the model
for kernel in ('linear', 'poly', 'rbf'):
    clf = svm.SVC(kernel=kernel, gamma=2)
    clf.fit(X, Y)

    # plot the line, the points, and the nearest vectors to the plane
    plt.figure(fignum, figsize=(4, 3))
    plt.clf()

    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                facecolors='none', zorder=10, edgecolors='k')
    plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired,
                edgecolors='k')

    plt.axis('tight')
    x_min = -3
    x_max = 3
    y_min = -3
    y_max = 3

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.figure(fignum, figsize=(4, 3))
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                levels=[-.5, 0, .5])

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.xticks(())
    plt.yticks(())
    fignum = fignum + 1
plt.show()
```
接著利用`svm.SVC`去設定 SVM 的相關參數，包括Kernel類型及Gamma值等等，其中Gamma值是只有當Kernel為'rbf','poly','sigmoid'時才需設定，設定完 SVM 後利用`fit`這個指令去訓練我們的資料和目標值。

最後，利用`plt.scatter`用來畫散點圖的指令幫助我們畫出資料點及標示出 support vector 的資料點位置，在下方結果圖中，標示為雙圈的資料點就是 support vector，接著利用`plt.contour`來畫出 SVM 的 decision boundary，在結果圖中以實線表示；虛線部分則為 margin 的範圍。

`plt.scatter`參數設定:

* s : Size 大小
* c : Color 設定
* zorder : 繪圖的順序，類似圖層概念，zorder 值越大代表越上層

`plt.contour`參數設定:

* colors : 設定等高線的顏色
* linestyles : 設定線所要呈現的方式
* levels : 該條等高線所代表的值

關於 support vector，margin 及 decision boundary 的相關介紹可以參考另外一篇 SVM 的教學 *EX9: SVM Margins Example* 中會有說明。

下面分別為 Linear, Polynomial, RBF 三種kernel產生出來的結果圖 :

![Linear Kernel](https://github.com/I-Yun/machine-learning-python/blob/master/SVM/linear.PNG "Linear Kernel") ![Polynomial Kernel](https://github.com/I-Yun/machine-learning-python/blob/master/SVM/poly.PNG "Polynomial Kernel") ![RBF Kernel](https://github.com/I-Yun/machine-learning-python/blob/master/SVM/rbf.PNG "RBF Kernel")

當然在 SVM 的 Kernel 選擇上不只能夠使用內建的 Kernel Type ，也可以自行設計 Kernel，在 SVM 的教學 *EX4: SVM with custom kernel* 中會有範例說明。

## (三)完整程式碼

Python source code: plot_svm_kernels.py

https://scikit-learn.org/stable/_downloads/plot_svm_kernels.py

iPython source code: plot_svm_kernels.ipynb

https://scikit-learn.org/stable/_downloads/plot_svm_kernels.ipynb

```python
"""
=========================================================
SVM-Kernels
=========================================================

Three different types of SVM-Kernels are displayed below.
The polynomial and RBF are especially useful when the
data-points are not linearly separable.


"""
print(__doc__)


# Code source: Gaël Varoquaux
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


# Our dataset and targets
X = np.c_[(.4, -.7),
          (-1.5, -1),
          (-1.4, -.9),
          (-1.3, -1.2),
          (-1.1, -.2),
          (-1.2, -.4),
          (-.5, 1.2),
          (-1.5, 2.1),
          (1, 1),
          # --
          (1.3, .8),
          (1.2, .5),
          (.2, -2),
          (.5, -2.4),
          (.2, -2.3),
          (0, -2.7),
          (1.3, 2.1)].T
Y = [0] * 8 + [1] * 8

# figure number
fignum = 1

# fit the model
for kernel in ('linear', 'poly', 'rbf'):
    clf = svm.SVC(kernel=kernel, gamma=2)
    clf.fit(X, Y)

    # plot the line, the points, and the nearest vectors to the plane
    plt.figure(fignum, figsize=(4, 3))
    plt.clf()

    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                facecolors='none', zorder=10, edgecolors='k')
    plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired,
                edgecolors='k')

    plt.axis('tight')
    x_min = -3
    x_max = 3
    y_min = -3
    y_max = 3

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.figure(fignum, figsize=(4, 3))
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                levels=[-.5, 0, .5])

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.xticks(())
    plt.yticks(())
    fignum = fignum + 1
plt.show()
```
