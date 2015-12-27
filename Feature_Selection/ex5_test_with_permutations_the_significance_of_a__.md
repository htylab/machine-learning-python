# 特徵選擇 Feature Selection 
##範例五: [Test with permutations the significance of a classification score](http://scikit-learn.org/stable/auto_examples/feature_selection/plot_permutation_test_for_classification.html)


(還看不太懂)

1. 若資料的標籤非常明確，但樣本只是隨機、順序的改變，使得同樣的樣本數目變多。
2. 因此在本範例中，介紹如何以交叉驗證算出資料的score與p-value



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
### (一)取得波士頓房產資料

### (二)使用LassoCV功能來篩選具有影響力的特徵
由於資料的類型為連續數字，選用LassoCV來做最具有代表性的特徵選取。
當設定好門檻值，並做訓練後，可以用transform(X)取得計算過後，被認為是具有影響力的特徵以及對應的樣本，可以由其列的數目知道總影響力特徵有幾個。
後面使用了增加門檻值來達到限制最後特徵數目的
使用門檻值來決定後來選取的參數，其說明在下一個標題。
### (三)設定選取參數的門檻值



