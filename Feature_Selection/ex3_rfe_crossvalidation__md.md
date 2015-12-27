# 特徵選擇 Feature Selection 
##範例三: [Recursive feature elimination with cross-validation](http://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html#example-feature-selection-plot-rfe-with-cross-validation-py)

[待整理資料] REFCV比REF多一個交叉比對的分數(grid_scores_)，代表選擇多少特徵後的準確率。但REFCV不用像REF要給定選擇多少特徵，而是會依照交叉比對的分數而自動選擇訓練的特徵數。

1. 以疊代方式計算模型
2. 以交叉驗證來取得影響力特徵


Python source code: [plot_rfe_digits.py](http://scikit-learn.org/stable/_downloads/plot_rfe_with_cross_validation.py)

```Python
print(__doc__)

import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification

# Build a classification task using 3 informative features
X, y = make_classification(n_samples=1000, n_features=25, n_informative=3,
                           n_redundant=2, n_repeated=0, n_classes=8,
                           n_clusters_per_class=1, random_state=0)

# Create the RFE object and compute a cross-validated score.
svc = SVC(kernel="linear")
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(y, 2),
              scoring='accuracy')
rfecv.fit(X, y)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
```
### (一)以疊代排序特徵影響力，並以交叉驗證來選出具有實際影響力的特徵

```Python
class sklearn.feature_selection.RFECV(estimator, step=1, cv=None, scoring=None, estimator_params=None, verbose=0)[source]
```

參數
* estimator
* step
* cv: 若無輸入，預設為3-fold的交叉驗證。輸入整數i，則做i-fold交叉驗證。若為物件，則以該物件做為交叉驗證產生器。
* scoring
* estimator_params
* verbose

輸出
* n\_features_: 預測有影響力的特徵的總數目
* support_: 有影響力的特徵遮罩，可以用來挑出哪些特徵
* ranking_: 各特徵的影響力程度
* grid_scores_: 從最有影響力的特徵開始加入，計算使用多少個特徵對應得到的準確率。
* estimator_

以RFECV設定好的功能物件，即可用以做訓練的動作。其結果可由n_features_得知有幾樣特徵是具有實際影響力。並可以由grid_scores_看出特徵的多寡如何影響準確率。
此功能需要設定交叉驗證的形式，本範例是以交叉驗證產生器做為輸入，其功能介紹如下。

### (二)交叉驗證的設定

```Python
class sklearn.cross_validation.StratifiedKFold(y, n_folds=3, shuffle=False, random_state=None)
```
參數
* y: 輸入的標籤資料
* n_folds: 需要分為多少組子項目
* shuffle: 在分割一單位前，是否需要隨機取樣
* random_state: 若要隨機取樣，可由此設定該以哪種形是隨機取樣，若無輸入則預設以numpy RNG 做隨機取樣。

該功能將輸入的y以設定好的子項目數目n_folds來做切割，

### (三)畫出具有影響力特徵對應準確率的圖

下圖的曲線表示選擇多少個特徵來做訓練，會得到多少的準確率。

![](http://scikit-learn.org/stable/_images/plot_rfe_with_cross_validation_001.png)

可以看到選擇三個最具有影響力的特徵時，交叉驗證的準確率高達81.8%。與建立模擬資料的n_informative=3是相對應的。
