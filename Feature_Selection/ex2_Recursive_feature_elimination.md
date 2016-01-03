# 特徵選擇 Feature Selection 
##範例二: [Recursive feature elimination](http://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_digits.html)

1. 以`load_digits`取得內建的數字辨識資料
2. 以疊代方式計算模型


Python source code: [plot_rfe_digits.py](http://scikit-learn.org/stable/_downloads/plot_rfe_digits.py)

```Python
print(__doc__)

from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt

# Load the digits dataset
digits = load_digits()
X = digits.images.reshape((len(digits.images), -1))
y = digits.target

# Create the RFE object and rank each pixel
svc = SVC(kernel="linear", C=1)
rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
rfe.fit(X, y)
ranking = rfe.ranking_.reshape(digits.images[0].shape)

# Plot pixel ranking
plt.matshow(ranking)
plt.colorbar()
plt.title("Ranking of pixels with RFE")
plt.show()
```
---
### (一)產生內建的數字辨識資料

```Python
sklearn.datasets.load_digits(n_class=10)
```
數位數字資料是解析度為8*8的手寫數字影像，總共有1797筆資料。預設為0~9十種數字類型，亦可由n_class來設定要取得多少種數字類型。

輸出的資料包含
1. ‘data’, 特徵資料(1797*64)
2. ‘images’, 影像資料(1797*8*8) 
3. ‘target’, 資料標籤(1797) 
4. ‘target_names’, 選取出的標籤列表(與n_class給定的長度一樣) 
5. ‘DESCR’, 此資料庫的描述

可以參考Classification的Ex1

### (二)以疊代方式計算模型

```Python
class sklearn.feature_selection.RFE(estimator, n_features_to_select=None, step=1, estimator_params=None, verbose=0)
```
以排除最不具影響力的特徵，做特徵的影響力排序。

參數    
* estimator:
* n_features_to_select:
* step:
* estimator_params:
* verbose:

回傳值
* n_features_:
* support_:
* ranking_: 特徵的影響力程度
* estimator_:

當我們以RFE指令建立好功能物件後，就可以用該功能物件做訓練，並以該物件中的ranking_物件取得特徵的影響力程度。

![](http://scikit-learn.org/stable/_images/plot_rfe_digits_001.png)

---
待整理資料:

rfe.estimator\_.coef_可以用來看係數，但是不知道為甚麼長度與想像中的不太一樣。[*](http://stackoverflow.com/questions/34204898/how-to-get-the-coefficients-from-rfe-using-sklearn)

希望能加入解釋RFE是如何判斷影響係數大小的解釋，下面為RFE的原始碼，看看他是怎麼把係數拿出來找最不具影響力的係數
```Python
        # Elimination
        while np.sum(support_) > n_features_to_select:
            # Remaining features
            features = np.arange(n_features)[support_]

            # Rank the remaining features
            estimator = clone(self.estimator)
            if self.estimator_params:
                estimator.set_params(**self.estimator_params)
            if self.verbose > 0:
                print("Fitting estimator with %d features." % np.sum(support_))

            estimator.fit(X[:, features], y)

            # Get coefs
            if hasattr(estimator, 'coef_'):
                coefs = estimator.coef_
            elif hasattr(estimator, 'feature_importances_'):
                coefs = estimator.feature_importances_
            else:
                raise RuntimeError('The classifier does not expose '
                                   '"coef_" or "feature_importances_" '
                                   'attributes')

            # Get ranks
            if coefs.ndim > 1:
                ranks = np.argsort(safe_sqr(coefs).sum(axis=0))
            else:
                ranks = np.argsort(safe_sqr(coefs))

            # for sparse case ranks is matrix
            ranks = np.ravel(ranks)

            # Eliminate the worse features
            threshold = min(step, np.sum(support_) - n_features_to_select)

            # Compute step score on the previous selection iteration
            # because 'estimator' must use features
            # that have not been eliminated yet
            if step_score:
                self.scores_.append(step_score(estimator, features))
            support_[features[ranks][:threshold]] = False
            ranking_[np.logical_not(support_)] += 1
```

看起來係數是這樣子算出來的
```Python
            # Get ranks
            if coefs.ndim > 1:
                ranks = np.argsort(safe_sqr(coefs).sum(axis=0))
            else:
                ranks = np.argsort(safe_sqr(coefs))
```

並且以以下程式碼，刪掉最不具影響力的特徵

```Python
            # Eliminate the worse features
            threshold = min(step, np.sum(support_) - n_features_to_select)
```