##特徵選擇/範例四: Feature selection using SelectFromModel and LassoCV

http://scikit-learn.org/stable/auto_examples/feature_selection/plot_select_from_model_boston.html

此範例是示範以`LassoCV`來挑選特徵，Lasso是一種用來計算稀疏矩陣的線性模形。在某些情況下是非常有用的，因為在此演算過程中會以較少數的特徵來找最佳解，基於參數有相依性的情況下，使變數的數目有效的縮減。因此，Lasso法以及它的變形式可算是壓縮參數關係基本方法。在某些情況下，此方法可以準確的偵測非零權重的值。

Lasso最佳化的目標函數:

![](http://scikit-learn.org/stable/_images/math/5ff15825a85204658e3e5aa6e3b5952b8f709c27.png)

1. 以`LassoCV`法來計算目標資訊性特徵數目較少的資料
2. 用`SelectFromModel`設定特徵重要性的門檻值來選擇特徵
3. 提高`SelectFromModel`的`.threshold`使目標資訊性特徵數逼近預期的數目



### (一)取得波士頓房產資料
```
from sklearn.datasets import load_boston
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV

# Load the boston dataset.
boston = load_boston()
X, y = boston['data'], boston['target']
```

### (二)使用LassoCV功能來篩選具有影響力的特徵
1. 由於資料的類型為連續數字，選用LassoCV來做最具有代表性的特徵選取。
2. 當設定好門檻值，並做訓練後，可以用transform(X)取得計算過後，被認為是具有影響力的特徵以及對應的樣本，可以由其列的數目知道總影響力特徵有幾個。
3. 後面使用了增加門檻值來達到限制最後特徵數目的
4. 使用門檻值來決定後來選取的參數，其說明在下一個標題。
5. 需要用後設轉換

### (三)設定選取參數的門檻值
```
while n_features > 2:
    sfm.threshold += 0.1
    X_transform = sfm.transform(X)
    n_features = X_transform.shape[1]
```

### (四)原始碼之出處
Python source code: [plot_select_from_model_boston.py](http://scikit-learn.org/stable/_downloads/plot_select_from_model_boston.py)

```Python
# Author: Manoj Kumar <mks542@nyu.edu>
# License: BSD 3 clause

print(__doc__)

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_boston
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV

# Load the boston dataset.
boston = load_boston()
X, y = boston['data'], boston['target']

# We use the base estimator LassoCV since the L1 norm promotes sparsity of features.
clf = LassoCV()

# Set a minimum threshold of 0.25
sfm = SelectFromModel(clf, threshold=0.25)
sfm.fit(X, y)
n_features = sfm.transform(X).shape[1]

# Reset the threshold till the number of features equals two.
# Note that the attribute can be set directly instead of repeatedly
# fitting the metatransformer.
while n_features > 2:
    sfm.threshold += 0.1
    X_transform = sfm.transform(X)
    n_features = X_transform.shape[1]

# Plot the selected two features from X.
plt.title(
    "Features selected from Boston using SelectFromModel with "
    "threshold %0.3f." % sfm.threshold)
feature1 = X_transform[:, 0]
feature2 = X_transform[:, 1]
plt.plot(feature1, feature2, 'r.')
plt.xlabel("Feature number 1")
plt.ylabel("Feature number 2")
plt.ylim([np.min(feature2), np.max(feature2)])
plt.show()
```
