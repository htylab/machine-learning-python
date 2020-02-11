# **Decision boundary of label propagation versus SVM on the Iris dataset**
https://scikit-learn.org/stable/auto_examples/semi_supervised/plot_label_propagation_versus_svm_iris.html#sphx-glr-auto-examples-semi-supervised-plot-label-propagation-versus-svm-iris-py

此範例是比較藉由Label Propagation(標籤傳播法)及SVM(支持向量機)對iris dataset(鳶尾花卉數據集)生成的decision boundary(決策邊界)
* Label Propagation屬於一種Semi-supervised learning(半監督學習)

## (一)引入函式庫

* numpy : 產生陣列數值
* matplotlib.pyplot : 用來繪製影像
* sklearn import datasets : 匯入資料集
* sklearn import svm : 匯入支持向量機
* sklearn.semi_supervised import LabelSpreading : 匯入標籤傳播算法
```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import svm
from sklearn.semi_supervised import LabelSpreading
```
## (二)讀取資料集

* numpy.random.RandomState(seed=None) : 產生隨機數
* datasets.load_iris() : 將資料及存入，iris為一個dict型別資料
* X代表從iris資料內讀取前兩項數據，分別表示萼片的長度及寬度
* y代表iris所屬的class
* np.copy() : 複製數據進行操作，避免修改原本的檔案
* [rng.rand(len(y)) < 0.3] = -1 代表隨機生成150個0~1區間的值，並將小於0.3的值轉為-1 (len(y)為150)
```python
rng = np.random.RandomState(0)
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target
h = .02 # 設定用於mesh的step
y_30 = np.copy(y)
y_30[rng.rand(len(y)) < 0.3] = -1
y_50 = np.copy(y)
y_50[rng.rand(len(y)) < 0.5] = -1
```
* LabelSpreading().fit() : 進行標籤傳播法並擬合數據集
* svm.SVC().fit() : 進行SVC(support vectors classification)並擬合數據集
* 分別用不同比例已被標籤的數據和未被標籤的數據進行標籤傳播，並與SVM的結果進行對比
```python
# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
ls30 = (LabelSpreading().fit(X, y_30), y_30)
ls50 = (LabelSpreading().fit(X, y_50), y_50)
ls100 = (LabelSpreading().fit(X, y), y)
rbf_svc = (svm.SVC(kernel='rbf', gamma=.5).fit(X, y), y)
```
## (三)繪製比較圖

* min()、max() : 決定x與y的範圍
* np.meshgrid() : 從給定的座標向量回傳座標矩陣
* 這裡分別是以x,y的最大、最小值加減1並以h=0.02的間隔來繪製
```python
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
```
* 定義各圖片標題
```python
titles = ['Label Spreading 30% data',
          'Label Spreading 50% data',
          'Label Spreading 100% data',
          'SVC with rbf kernel']
```
* 為了繪製圖片，設定一個為dict型態的color_map，將4種label分別給予不同顏色

最後用下面的程式將所有點繪製出來
```python
color_map = {-1: (1, 1, 1), 0: (0, 0, .9), 1: (1, 0, 0), 2: (.8, .6, 0)}

for i, (clf, y_train) in enumerate((ls30, ls50, ls100, rbf_svc)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    plt.subplot(2, 2, i + 1)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    plt.axis('off')

    # Plot also the training points
    colors = [color_map[y] for y in y_train]
    plt.scatter(X[:, 0], X[:, 1], c=colors, edgecolors='black')

    plt.title(titles[i])

plt.suptitle("Unlabeled points are colored white", y=0.1)
plt.show()
```
* 顯示了即使只有少部分被標籤的數據，Label Propagation也能很好的學習產生decision boundary
![](https://github.com/sdgary56249128/machine-learning-python/blob/master/Semi-Supervised_Classification/plot_label_propagation_versus_svm_iris_001.png)
## (四)完整程式碼

https://scikit-learn.org/stable/_downloads/97a366ef6b2f7394d4eb409814bf4842/plot_label_propagation_versus_svm_iris.py
```python
print(__doc__)

# Authors: Clay Woolam <clay@woolam.org>
# License: BSD

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn.semi_supervised import LabelSpreading

rng = np.random.RandomState(0)

iris = datasets.load_iris()

X = iris.data[:, :2]
y = iris.target

# step size in the mesh
h = .02

y_30 = np.copy(y)
y_30[rng.rand(len(y)) < 0.3] = -1
y_50 = np.copy(y)
y_50[rng.rand(len(y)) < 0.5] = -1
# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
ls30 = (LabelSpreading().fit(X, y_30), y_30)
ls50 = (LabelSpreading().fit(X, y_50), y_50)
ls100 = (LabelSpreading().fit(X, y), y)
rbf_svc = (svm.SVC(kernel='rbf', gamma=.5).fit(X, y), y)

# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# title for the plots
titles = ['Label Spreading 30% data',
          'Label Spreading 50% data',
          'Label Spreading 100% data',
          'SVC with rbf kernel']

color_map = {-1: (1, 1, 1), 0: (0, 0, .9), 1: (1, 0, 0), 2: (.8, .6, 0)}

for i, (clf, y_train) in enumerate((ls30, ls50, ls100, rbf_svc)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    plt.subplot(2, 2, i + 1)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    plt.axis('off')

    # Plot also the training points
    colors = [color_map[y] for y in y_train]
    plt.scatter(X[:, 0], X[:, 1], c=colors, edgecolors='black')

    plt.title(titles[i])

plt.suptitle("Unlabeled points are colored white", y=0.1)
plt.show()
```
