# 特徵選擇/範例一: Pipeline Anova SVM

http://scikit-learn.org/stable/auto_examples/feature_selection/feature_selection_pipeline.html

此機器學習範例示範佇列的使用，依照順序執行ANOVA挑選主要特徵，並且使用C-SVM來計算特徵的權重與預測。

1. 使用 `make_classification` 建立模擬資料
2. 使用 `SelectKBest` 設定要用哪種目標函式，以挑出可提供信息的特徵
3. 使用 `SVC` 設定支持向量機為分類計算以及其核函數
4. 用 `make_pipeline` 合併 SelectKBest物件 與 SVC物件
5. 用 `fit` 做訓練，並且以 `predict` 來做預測


---
### (一)建立模擬資料

在選擇特徵之前需要有整理好的特徵與目標資料。在此範例中，將以`make_classification`功能建立特徵與目標。該功能可以依照使用者想模擬的情況，建立含有不同特性的模擬資料，像是總特徵數目，其中有幾項特徵含有目標資訊性、目標聚集的程度、目標分為幾類等等的特性。


```python
# import some data to play with
X, y = samples_generator.make_classification(
    n_features=20, n_informative=3, n_redundant=0, n_classes=4,
    n_clusters_per_class=2)
```
在本範例，我們將X建立為一個有20個特徵的資料，其中有3種特徵具有目標資訊性，0個特徵是由目標資訊性特徵所產生的線性組合，目標分為4類，而每個分類的目標分布為2個群集。


### (二)選擇最好的特徵

在機器學習的訓練之前，可以藉由統計或指定評分函數，算出特徵與目標之間的關係，並挑選出最具有關係的特徵作為訓練的素材，而不直接使用所有特徵做為訓練的素材。

其中一種方法是統計特徵與目標之間的F-score做為評估分數，再挑選F-score最高的幾個特徵作為訓練素材。我們可以用 `SelectKBest()` 來建立該功能的運算物件。

```python
# ANOVA SVM-C
# 1) anova filter, take 3 best ranked features
anova_filter = SelectKBest(f_regression, k=3)
```
`SelectKBest()`的第一項參數須給定評分函數，在本範例是設定為`f_regression` 。第二項參數代表選擇評估分數最高的3個特徵做為訓練的素材。建立完成後，即可用物件內的方法`.fit_transform(X,y)` 來提取被選出來的特徵。

### (三)以佇列方式來設定支持向量機分類法運算物件

Scikit-lenarn的支持向量機分類涵式庫提供使用簡單易懂的指令，只要用 `SVC()` 建立運算物件後，便可以用運算物件內的方法 `.fit()` 與 `.predict()` 來做訓練與預測。

本範例在建立運算物件後，不直接用`SelectKBest().fit_transform()` 提出訓練素材。而是以 `make_pipeline()`合併先前設定好的兩個運算物件。再執行`.fit()` 與 `.predict()`來完成訓練與預測的動作。

```python
# 2) svm
clf = svm.SVC(kernel='linear')

anova_svm = make_pipeline(anova_filter, clf)
anova_svm.fit(X, y)
anova_svm.predict(X)
```
當我們以佇列建立好的運算物件，就可以直接給定所有的特徵資料與目標資料做訓練與預測。在訓練過程中，會依照給定的特徵素材數目從特徵資料中挑出特徵素材。預測時，也會從預測資料中挑出對應特徵素材的資料來做預測判斷。

若是將`SelectKBest()`與 `SVC()`物件分開來執行，當 `SVC()`物件在做學習時給定的特徵即為被選出來的特徵素材數目。那預測的時候也必須從預測資料中，挑出被`SelectKBest()`選出來的特徵來給`SVC()`做預測。

---

## (四)原始碼

### Python source code: [feature_selection_pipeline.py](http://scikit-learn.org/stable/_downloads/feature_selection_pipeline.py)

```python
from sklearn import svm
from sklearn.datasets import samples_generator
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import make_pipeline

# import some data to play with
X, y = samples_generator.make_classification(
    n_features=20, n_informative=3, n_redundant=0, n_classes=4,
    n_clusters_per_class=2)

# ANOVA SVM-C
# 1) anova filter, take 3 best ranked features
anova_filter = SelectKBest(f_regression, k=3)
# 2) svm
clf = svm.SVC(kernel='linear')

anova_svm = make_pipeline(anova_filter, clf)
anova_svm.fit(X, y)
anova_svm.predict(X)
```

## (五)函式用法
###[`make_classification()`](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html) 的參數



```Python
sklearn.datasets.make_classification(   n_samples=100,
                                        n_features=20,
                                        n_informative=2,
                                        n_redundant=2,
                                        n_repeated=0,
                                        n_classes=2,
                                        n_clusters_per_class=2,    
                                        weights=None,
                                        flip_y=0.01,
                                        class_sep=1.0,
                                        hypercube=True,
                                        shift=0.0,
                                        scale=1.0,
                                        shuffle=True,
                                        random_state=None)
```

參數:
* n_samples :
* n_fratures : 總特徵數目
* n_informative: 有意義的特徵數目
* n_redundant : 產生有意義特徵的隨機線性組合
* n_repeated
* n_classes: 共分類為幾類
* n_clusters_per_class: 一個類群有幾個群組分布
* weights :
* flip_y :
* class_sep :
* hypercube :
* shift :
* scale :
* shuffle :
* random_state :

輸出:
* X : 特徵矩陣資料
* Y : 對應目標資料

類似的功能:

[make_blobs](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html#sklearn.datasets.make_blobs)

[make_gaussian_quantiles](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_gaussian_quantiles.html#sklearn.datasets.make_gaussian_quantiles)


---

###[`SelectKBest()`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html#sklearn.feature_selection.SelectKBest) 的參數


SelectKBest 的使用:
* 選擇最好的特徵(目標函式, 特徵個數)
* 目標函式:  測試X與Y之間關係，須提供F score與p-value
* 特徵個數: 最好的特徵個數

f_regression 的使用：

* f_regression(X,y)
* 輸入X與y
* 輸出F score與p-value
