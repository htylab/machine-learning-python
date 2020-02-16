# **IsolationForest example**

https://scikit-learn.org/stable/auto_examples/ensemble/plot_isolation_forest.html#sphx-glr-auto-examples-ensemble-plot-isolation-forest-py

此範例介紹IsolationForest(隔離森林、孤立森林)的使用方式及其效果，使用IsolationForest會回傳每個樣本的異常分數

IsolationForest是用於異常檢測的unsupervised learning(無監督學習)算法，適合用於大規模連續數據(網路資安和流量異常、金融機構)，其工作原理是隔離異常樣本(可以理解為分布稀疏且離密度高的群體較遠的點)

和RandomForest(隨機森林)類似，但在建立iTree時，每次選擇劃分條件及劃分點時都是隨機的，而不是根據樣本內容或是樣本相關資訊

在建立iTree的過程中，如果一些樣本很快就到達了leaf節點(即leaf到root的距離d很短)，那就很有可能是異常點。因為那些路徑d比較短的樣本，都是距離主要的樣本中心比較遠的點。因此可以透過計算樣本在所有樹中的平均路徑長度來尋找異常點

## (一)引入函式庫

* numpy : 產生陣列數值
* matplotlib.pyplot : 用來繪製影像
* sklearn.ensemble import IsolationForest : 匯入隔離森林算法

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
```

## (二)產生訓練樣本

* np.random.RandomState(seed) : 產生偽隨機數，當seed值相同時，產生的數值為一樣
* np.r_[] : 將數據沿第一個軸相連接
* rng.uniform() : 隨機數產生

```python
rng = np.random.RandomState(42)

# Generate train data
X = 0.3 * rng.randn(100, 2)   # 生成100筆基礎資料
X_train = np.r_[X + 2, X - 2] # 將+,-2的資料相連接成為一筆(200,2)
# Generate some regular novel observations
X = 0.3 * rng.randn(20, 2)    # 生成20筆新的正常資料
X_test = np.r_[X + 2, X - 2]  # 將+,-2的資料相連接成為一筆(40,2)
# Generate some abnormal novel observations
X_outliers = rng.uniform(low=-4, high=4, size=(20, 2)) # 生成20筆新的異常資料，藉由亂數產生
```

## (三)IsolationForest model

* IsolationForest(n_estimators=100, max_samples='auto', contamination='auto', max_features=1.0, bootstrap=False, n_jobs=None, behaviour='deprecated', random_state=None, verbose=0, warm_start=False)

1. n_estimators : 森林中樹的棵樹
2. max_samples : 每棵樹中的樣本數量
3. contamination : 設置樣本中異常
4. max_features : 每顆樹中特徵個數或比例
5. random_state : 隨機數與random_seed作用相同

* fit() : 擬合資料
* predict() : 預測資料

```python
# fit the Model
clf = IsolationForest(max_samples=100, random_state=rng)
clf.fit(X_train) 
y_pred_train = clf.predict(X_train) 
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)
```

## (四)繪製結果

* np.meshgrid() : 從給定的座標向量回傳座標矩陣
* np.linspace(start, stop, num) : 回傳指定間格內的數值
* numpy.c_[] : 將數據沿第二個軸相連接
* plt.contourf() : 繪製輪廓
* plt.scatter() : 繪製x與y的散點圖，其中標記大小和顏色不同
最後用下面的程式將所有點繪製出來

```python
xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("IsolationForest")
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', 
                 s=20, edgecolor='k') # 100筆正常基礎資料標示為白色
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green',
                 s=20, edgecolor='k') # 20筆新的正常資料標示為綠色
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red',
                s=20, edgecolor='k')  # 20筆新的異常資料標示為紅色
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend([b1, b2, c],
           ["training observations",
            "new regular observations", "new abnormal observations"],
           loc="upper left")
plt.show()
```

![](https://github.com/sdgary56249128/machine-learning-python/blob/master/Ensemble_methods/sphx_glr_plot_isolation_forest_001.png)

## (五)完整程式碼

https://scikit-learn.org/stable/_downloads/a48f0894575e256740089d572cff3acd/plot_isolation_forest.py

```python
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

rng = np.random.RandomState(42)

# Generate train data
X = 0.3 * rng.randn(100, 2)
X_train = np.r_[X + 2, X - 2]
# Generate some regular novel observations
X = 0.3 * rng.randn(20, 2)
X_test = np.r_[X + 2, X - 2]
# Generate some abnormal novel observations
X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))

# fit the model
clf = IsolationForest(max_samples=100, random_state=rng)
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)

# plot the line, the samples, and the nearest vectors to the plane
xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("IsolationForest")
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white',
                 s=20, edgecolor='k')
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green',
                 s=20, edgecolor='k')
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red',
                s=20, edgecolor='k')
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend([b1, b2, c],
           ["training observations",
            "new regular observations", "new abnormal observations"],
           loc="upper left")
plt.show()
```
