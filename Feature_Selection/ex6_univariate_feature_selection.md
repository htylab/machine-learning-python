##特徵選擇/範例六: Univariate Feature Selection

http://scikit-learn.org/stable/auto_examples/feature_selection/plot_feature_selection.html


此範例示範單變量特徵的選擇。鳶尾花資料中會加入數個雜訊特徵(不具影響力的特徵資訊)並且選擇單變量特徵。選擇過程會畫出每個特徵的 p-value 與其在支持向量機中的權重。可以從圖表中看出主要影響力特徵的選擇會選出具有主要影響力的特徵，並且這些特徵會在支持向量機有相當大的權重。
在本範例的所有特徵中，只有最前面的四個特徵是對目標有意義的。我們可以看到這些特徵的單變量特徵評分很高。而支持向量機會賦予最主要的權重到這些具影響力的特徵之一，但也會挑選剩下的特徵來做判斷。在支持向量機增加權重之前就確定那些特徵較具有影響力，從而增加辨識率。

1. 資料集：鳶尾花
2. 特徵：萼片(sepal)之長與寬以及花瓣(petal)之長與寬
3. 預測目標：共有三種鳶尾花 setosa, versicolor, virginica
4. 機器學習方法：線性分類
5. 探討重點：使用單變量選擇(`SelectPercentile`)挑出訓練特徵，與直接將所有訓練特徵輸入的分類器做比較
6. 關鍵函式： `sklearn.feature_selection.SelectPercentile`


### (一)修改原本的鳶尾花資料

用`datasets.load_iris()`讀取鳶尾花的資料做為具有影響力的特徵，並以`np.random.uniform`建立二十個隨機資料做為不具影響力的特徵，並合併做為訓練樣本。
```###############################################################################
# import some data to play with

# The iris dataset
iris = datasets.load_iris()

# Some noisy data not correlated
E = np.random.uniform(0, 0.1, size=(len(iris.data), 20))

# Add the noisy data to the informative features
X = np.hstack((iris.data, E))
y = iris.target
```

### (二)使用f-value作為判斷的基準來找主要影響力特徵

以`SelectPercentile`作單變量特徵的計算，以F-test(`f_classif`)來做為選擇的統計方式，挑選函式輸出結果大於百分之十的特徵。並將計算出來的單便量特徵分數結果做正規化，以便比較每特徵在使用單變量計算與未使用單變量計算的差別。
```
###############################################################################
# Univariate feature selection with F-test for feature scoring
# We use the default selection function: the 10% most significant features
selector = SelectPercentile(f_classif, percentile=10)
selector.fit(X, y)
scores = -np.log10(selector.pvalues_)
scores /= scores.max()
plt.bar(X_indices - .45, scores, width=.2,
        label=r'Univariate score ($-Log(p_{value})$)', color='g')
```
### (三)找出不計算單變量特徵的分類權重

以所有特徵資料，以線性核函數丟入支持向量分類機，找出各特徵的權重。
```
###############################################################################
# Compare to the weights of an SVM
clf = svm.SVC(kernel='linear')
clf.fit(X, y)

svm_weights = (clf.coef_ ** 2).sum(axis=0)
svm_weights /= svm_weights.max()

plt.bar(X_indices - .25, svm_weights, width=.2, label='SVM weight', color='r')
```
### (四)找出以單變量特徵選出的分類權重

以單變量特徵選擇選出的特徵，做為分類的訓練特徵，差別在於訓練的特徵資料是使用`selector.transform(X)`將`SelectPercentile`選擇的結果讀取出來，並算出以單變量特徵選擇做預先選擇後，該分類器的判斷權重。
```
clf_selected = svm.SVC(kernel='linear')
clf_selected.fit(selector.transform(X), y)

svm_weights_selected = (clf_selected.coef_ ** 2).sum(axis=0)
svm_weights_selected /= svm_weights_selected.max()

plt.bar(X_indices[selector.get_support()] - .05, svm_weights_selected,
        width=.2, label='SVM weights after selection', color='b')
```

### (五)原始碼出處
Python source code: [plot_feature_selection.py](http://scikit-learn.org/stable/_downloads/plot_feature_selection.py)

```Python
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets, svm
from sklearn.feature_selection import SelectPercentile, f_classif

###############################################################################
# import some data to play with

# The iris dataset
iris = datasets.load_iris()

# Some noisy data not correlated
E = np.random.uniform(0, 0.1, size=(len(iris.data), 20))

# Add the noisy data to the informative features
X = np.hstack((iris.data, E))
y = iris.target

###############################################################################
plt.figure(1)
plt.clf()

X_indices = np.arange(X.shape[-1])

###############################################################################
# Univariate feature selection with F-test for feature scoring
# We use the default selection function: the 10% most significant features
selector = SelectPercentile(f_classif, percentile=10)
selector.fit(X, y)
scores = -np.log10(selector.pvalues_)
scores /= scores.max()
plt.bar(X_indices - .45, scores, width=.2,
        label=r'Univariate score ($-Log(p_{value})$)', color='g')

###############################################################################
# Compare to the weights of an SVM
clf = svm.SVC(kernel='linear')
clf.fit(X, y)

svm_weights = (clf.coef_ ** 2).sum(axis=0)
svm_weights /= svm_weights.max()

plt.bar(X_indices - .25, svm_weights, width=.2, label='SVM weight', color='r')

clf_selected = svm.SVC(kernel='linear')
clf_selected.fit(selector.transform(X), y)

svm_weights_selected = (clf_selected.coef_ ** 2).sum(axis=0)
svm_weights_selected /= svm_weights_selected.max()

plt.bar(X_indices[selector.get_support()] - .05, svm_weights_selected,
        width=.2, label='SVM weights after selection', color='b')


plt.title("Comparing feature selection")
plt.xlabel('Feature number')
plt.yticks(())
plt.axis('tight')
plt.legend(loc='upper right')
plt.show()
```
