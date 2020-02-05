# **Decision boundary of label propagation versus SVM on the Iris dataset**
https://scikit-learn.org/stable/auto_examples/semi_supervised/plot_label_propagation_versus_svm_iris.html#sphx-glr-auto-examples-semi-supervised-plot-label-propagation-versus-svm-iris-py

此範例是比較藉由Label Propagation(標籤傳播法)及SVM(支持向量機)對iris dataset(鳶尾花卉數據集)生成的decision boundary(決策邊界)

Label Propagation屬於一種Semi-supervised learning(半監督學習)

顯示了即使只有少部分被標籤的數據，Label Propagation也能很好的學習產生decision boundary


## (一)引入函式庫

1. numpy : 產生陣列數值
2. matplotlib.pyplot : 用來繪製影像
3. sklearn import datasets : 匯入資料集
4. sklearn import svm : 匯入支持向量機
5. sklearn.semi_supervised import LabelSpreading : 匯入標籤傳播算法

```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import svm
from sklearn.semi_supervised import LabelSpreading
```
使用numpy.random.RandomState(seed=None)產生隨機數
```python
rng = np.random.RandomState(0)
```
使用datasets.load_iris()將資料及存入，iris為一個dict型別資料
```python
iris = datasets.load_iris()
```
X代表從iris資料內讀取前兩項數據，分別表示萼片的長度及寬度
y代表iris的class
```python
X = iris.data[:, :2]
y = iris.target
```
設定用於mesh的step
```python
h = .02
```

```python
y_30 = np.copy(y)
y_30[rng.rand(len(y)) < 0.3] = -1
y_50 = np.copy(y)
y_50[rng.rand(len(y)) < 0.5] = -1
```










