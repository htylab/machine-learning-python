##特徵選擇/範例六: Univariate Feature Selection

http://scikit-learn.org/stable/auto_examples/feature_selection/plot_feature_selection.html


此範例示範主要影響力特徵的選擇。鳶尾花資料中會加入數個雜訊(不具影響力的特徵資訊)並且選擇主要影響力的特徵。選擇過程會畫出每個特徵的 p-value 與其在支持向量機中的權重。可以從圖表中看出主要影響力特徵的選擇會選出具有主要影響力的特徵，並且這些特徵會在支持向量機有相當大的權重。在所有的特徵裡面，只有最前面的四個特徵是對目標有意義的。我們可以看到在主要影響力特徵的選擇中，這些特徵的評分會相當高。而支持向量機會認定其中一個特徵有很大的權重，但也會用一些不具有影響力的特徵來做選擇判斷。在支持向量機增加權重之前就確定那些特徵較具有影響力，從而增加辨識率。


1. 若資料的標籤非常明確，但樣本只是隨機、順序的改變，使得同樣的樣本數目變多。
2. 因此在本範例中，介紹如何以交叉驗證算出資料的score與p-value

### (一)修改原本的鳶尾花資料

### (二)使用f-value作為判斷的基準來找主要影響力特徵
函式 SelectPercentile
### (三)設定選取參數的門檻值


### (四)原始碼出處
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
