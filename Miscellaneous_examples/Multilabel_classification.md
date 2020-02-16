# **Multilabel classification**
https://scikit-learn.org/stable/auto_examples/plot_multilabel.html#sphx-glr-auto-examples-plot-multilabel-py

Multi-Label(多標籤) vs Multi-Class(多分類) :
-
*  一部電影可以分為普遍級、保護級、輔導級、限制級，那這部電影只會屬於其中一類，這就是 multi-class
*  一部電影可以同時有很多種的類型例如喜劇、劇情、浪漫等等，這就是 multi-label

範例 :
-
模擬multi-label document(多標籤檔案)的分類問題，數據集是依照下面的方式隨機生成的:

1. pick the number of labels: n ~ Poisson(n_labels)
2. n times, choose a class c: c ~ Multinomial(theta)
3. pick the document length: k ~ Poisson(length)
4. k times, choose a word: w ~ Multinomial(theta_c)

* Poisson distribution(帕松分布) : 適合於描述單位時間內隨機事件發生的次數的機率分布

* Multinomial distribution(多項式分布) : 多項分布是二項分布的延伸。例如，二項分布的典型範例為扔硬幣，正面槽上的機率為p， 重複扔n次，k次為正面的機率即是一個二項分布的機率。把二項分布公式推廣到多種狀態，就得到多項式分布。

透過上面的方法，剔除採樣的目的是為了確保n(label數)可以大於2，而且文件的長度不等於0。同樣，也排除已經選過的類別。備標註為2種類別的檔案會以雙重顏色的圈圈表示。

為了進行可視化，藉由PCA (Principal Component Analysis 主成分分析) 和CCA (Canonical Correlation Analysis 典型相關分析) 找到前兩個主要成分將數據projecting(投影)後來執行分類。使用sklearn.multiclass.OneVsRestClassifier，metaclassifier(元分類器)使用兩個帶有線性內核的SVC來學習每個類別的discriminative model(判別模型)。

* PCA用於執行unsupervised(無監督)的降維，而CCA用於執行supervised(監督)的降維。

## (一)引入函式庫

* numpy : 產生陣列數值
* matplotlib.pyplot : 用來繪製影像
* sklearn.datasets import make_multilabel_classification : 生成隨機的多標籤分類問題
* sklearn.svm import SVC : 匯入Support Vector Classification
* sklearn.decomposition import PCA : 匯入Principal Component Analysis
* sklearn.cross_decomposition import CCA : 匯入Canonical Correlation Analysis

```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
```

## (二)定義繪製hyperplane函式

* np.linspace() : 回傳指定區間內的相同間隔的數字

```python
def plot_hyperplane(clf, min_x, max_x, linestyle, label):
    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min_x - 5, max_x + 5)  # make sure the line is long enough
    yy = a * xx - (clf.intercept_[0]) / w[1]
    plt.plot(xx, yy, linestyle, label=label)
```

## (三)定義繪製圖片函式

* PCA(n_components=None, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=None)

1. n_components : 要保留的成分數，此範例是保留2項

* CCA(n_components=2, scale=True, max_iter=500, tol=1e-06, copy=True)

1. n_components : 要保留的成分數，此範例是保留2項
2. scale : 是否縮放數據

* OneVsRestClassifier(estimator, n_jobs=None): 一對一（OvR）的多類/多標籤策略

1. estimator : 估計對象，此範例使用SVC

```python
def plot_subfigure(X, Y, subplot, title, transform):
    if transform == "pca":
        X = PCA(n_components=2).fit_transform(X)
    elif transform == "cca":
        X = CCA(n_components=2).fit(X, Y).transform(X)
    else:
        raise ValueError

    min_x = np.min(X[:, 0])
    max_x = np.max(X[:, 0])

    min_y = np.min(X[:, 1])
    max_y = np.max(X[:, 1])

    classif = OneVsRestClassifier(SVC(kernel='linear'))
    classif.fit(X, Y)

    plt.subplot(2, 2, subplot)
    plt.title(title)

    zero_class = np.where(Y[:, 0])
    one_class = np.where(Y[:, 1])
    plt.scatter(X[:, 0], X[:, 1], s=40, c='gray', edgecolors=(0, 0, 0))
    plt.scatter(X[zero_class, 0], X[zero_class, 1], s=160, edgecolors='b',
                facecolors='none', linewidths=2, label='Class 1')
    plt.scatter(X[one_class, 0], X[one_class, 1], s=80, edgecolors='orange',
                facecolors='none', linewidths=2, label='Class 2')

    plot_hyperplane(classif.estimators_[0], min_x, max_x, 'k--',
                    'Boundary\nfor class 1')
    plot_hyperplane(classif.estimators_[1], min_x, max_x, 'k-.',
                    'Boundary\nfor class 2')
    plt.xticks(())
    plt.yticks(())

    plt.xlim(min_x - .5 * max_x, max_x + .5 * max_x)
    plt.ylim(min_y - .5 * max_y, max_y + .5 * max_y)
    if subplot == 2:
        plt.xlabel('First principal component')
        plt.ylabel('Second principal component')
        plt.legend(loc="upper left")


plt.figure(figsize=(8, 6))
```

## (四)呼叫函式並輸出圖片

```python
X, Y = make_multilabel_classification(n_classes=2, n_labels=1,
                                      allow_unlabeled=True,
                                      random_state=1)

plot_subfigure(X, Y, 1, "With unlabeled samples + CCA", "cca")
plot_subfigure(X, Y, 2, "With unlabeled samples + PCA", "pca")

X, Y = make_multilabel_classification(n_classes=2, n_labels=1,
                                      allow_unlabeled=False,
                                      random_state=1)

plot_subfigure(X, Y, 3, "Without unlabeled samples + CCA", "cca")
plot_subfigure(X, Y, 4, "Without unlabeled samples + PCA", "pca")

plt.subplots_adjust(.04, .02, .97, .94, .09, .2)
plt.show()
```

* 在圖中，“未標記樣本”並不意味著我們不知道標記（如在半監督學習中一樣），而是樣本根本沒有標記。

![](https://github.com/sdgary56249128/machine-learning-python/blob/master/Miscellaneous_examples/sphx_glr_plot_multilabel_001.png)

## (五)完整程式碼

https://scikit-learn.org/stable/_downloads/39d4a835d597f9ae7842ba4a877fd5b1/plot_multilabel.py

```python
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA


def plot_hyperplane(clf, min_x, max_x, linestyle, label):
    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min_x - 5, max_x + 5)  # make sure the line is long enough
    yy = a * xx - (clf.intercept_[0]) / w[1]
    plt.plot(xx, yy, linestyle, label=label)


def plot_subfigure(X, Y, subplot, title, transform):
    if transform == "pca":
        X = PCA(n_components=2).fit_transform(X)
    elif transform == "cca":
        X = CCA(n_components=2).fit(X, Y).transform(X)
    else:
        raise ValueError

    min_x = np.min(X[:, 0])
    max_x = np.max(X[:, 0])

    min_y = np.min(X[:, 1])
    max_y = np.max(X[:, 1])

    classif = OneVsRestClassifier(SVC(kernel='linear'))
    classif.fit(X, Y)

    plt.subplot(2, 2, subplot)
    plt.title(title)

    zero_class = np.where(Y[:, 0])
    one_class = np.where(Y[:, 1])
    plt.scatter(X[:, 0], X[:, 1], s=40, c='gray', edgecolors=(0, 0, 0))
    plt.scatter(X[zero_class, 0], X[zero_class, 1], s=160, edgecolors='b',
                facecolors='none', linewidths=2, label='Class 1')
    plt.scatter(X[one_class, 0], X[one_class, 1], s=80, edgecolors='orange',
                facecolors='none', linewidths=2, label='Class 2')

    plot_hyperplane(classif.estimators_[0], min_x, max_x, 'k--',
                    'Boundary\nfor class 1')
    plot_hyperplane(classif.estimators_[1], min_x, max_x, 'k-.',
                    'Boundary\nfor class 2')
    plt.xticks(())
    plt.yticks(())

    plt.xlim(min_x - .5 * max_x, max_x + .5 * max_x)
    plt.ylim(min_y - .5 * max_y, max_y + .5 * max_y)
    if subplot == 2:
        plt.xlabel('First principal component')
        plt.ylabel('Second principal component')
        plt.legend(loc="upper left")


plt.figure(figsize=(8, 6))

X, Y = make_multilabel_classification(n_classes=2, n_labels=1,
                                      allow_unlabeled=True,
                                      random_state=1)

plot_subfigure(X, Y, 1, "With unlabeled samples + CCA", "cca")
plot_subfigure(X, Y, 2, "With unlabeled samples + PCA", "pca")

X, Y = make_multilabel_classification(n_classes=2, n_labels=1,
                                      allow_unlabeled=False,
                                      random_state=1)

plot_subfigure(X, Y, 3, "Without unlabeled samples + CCA", "cca")
plot_subfigure(X, Y, 4, "Without unlabeled samples + PCA", "pca")

plt.subplots_adjust(.04, .02, .97, .94, .09, .2)
plt.show()
```
