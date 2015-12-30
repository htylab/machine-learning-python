##範例七: Face completion with a multi-output estimators
http://scikit-learn.org/stable/auto_examples/plot_multioutput_face_completion.html

這個範例用來展示scikit-learn 如何用 extremely randomized trees, k nearest neighbors, linear regression and ridge regression 演算法來完成人臉估測。





## (一)引入函式庫及內建影像資料庫

引入之函式庫如下

1. numpy
2. matplotlib.pyplot: 用來繪製影像
2. sklearn.datasets: 用來繪入內建之影像資料庫
3. sklearn.utils.validation import check_random_state
4. from sklearn.ensemble import ExtraTreesRegresso
5. from sklearn.neighbors import KNeighborsRegressor
6. from sklearn.linear_model import LinearRegression
7. from sklearn.linear_model import RidgeCV



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

把每張訓練影像和測試影像都切割成上下兩部分，
X_人臉上半部分，
Y_人臉下半部分。
```python
n_pixels = data.shape[1]
X_train = train[:, :np.ceil(0.5 * n_pixels)]  
y_train = train[:, np.floor(0.5 * n_pixels):]  
X_test = test[:, :np.ceil(0.5 * n_pixels)]
y_test = test[:, np.floor(0.5 * n_pixels):]
```