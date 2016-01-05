##範例一: Plotting Cross-Validated Predictions
http://scikit-learn.org/stable/auto_examples/plot_cv_predict.html

這個範例可以看出用`cross_val_predict`的預測誤差

## (一)引入函式庫及內建測試資料庫

引入之函式庫如下

1. `matplotlib.pyplot`: 用來繪製影像
2. `sklearn.datasets`: 用來繪入內建測試資料庫
3. `sklearn.cross_validation import cross_val_predict` 
4. `sklearn.linear_model` 




使用 `datasets.load_digits()` 將資料存入， `data` 為一個dict型別資料，我們看一下資料的內容。

```python
from sklearn.datasets import fetch_olivetti_faces
data = fetch_olivetti_faces()
targets = data.target
data = data.images.reshape((len(data.images), -1))
```

| 顯示 | 說明 |
| -- | -- |
| ('images', (400, 64, 64))| 共有40個人，每個人各有10張影像，共有 400 張影像，影像大小為 64x64 |
| ('data', (400, 4096)) | data 則是將64x64的矩陣攤平成4096個元素之一維向量 |
| ('targets', (400,)) | 說明400張圖與40個人之分類對應 0-39，記錄每張影像是哪一個人 |
| DESCR | 資料之描述 |


前面30個人當訓練資料，之後當測試資料
```python
train = data[targets < 30]
test = data[targets >= 30]
```
測試影像從100張亂數選5張出來，變數`test`的大小變成(5,4096)
```python
# Test on a subset of people
n_faces = 5
rng = check_random_state(4)
face_ids = rng.randint(test.shape[0], size=(n_faces, ))
test = test[face_ids, :]
```

把每張訓練影像和測試影像都切割成上下兩部分:

X_人臉上半部分，
Y_人臉下半部分。
```python
n_pixels = data.shape[1]
X_train = train[:, :np.ceil(0.5 * n_pixels)]  
y_train = train[:, np.floor(0.5 * n_pixels):]  
X_test = test[:, :np.ceil(0.5 * n_pixels)]
y_test = test[:, np.floor(0.5 * n_pixels):]
```



```
from sklearn import datasets
from sklearn.cross_validation import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt

lr = linear_model.LinearRegression()
boston = datasets.load_boston()
y = boston.target

# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validated:
predicted = cross_val_predict(lr, boston.data, y, cv=10)

fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
```