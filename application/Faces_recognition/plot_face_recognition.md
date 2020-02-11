# 用特徵臉及SVM進行人臉辨識實例:Faces recognition example using eigenfaces and SVMs

https://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html

本範例所使用的資料庫主要採集於LFW人臉資料庫

http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)

採取資料集中最具有代表性的人做預測，以下為預測結果:

![](https://github.com/JENNSHIUAN/machine-learning-python/blob/master/application/Faces_recognition/Face_fig1.JPG)

## (一)引入函式庫
引入函式庫如下：
1. ```time```:計算時間
2. ```logging```:具有除錯功能
3. ```matplotlib.pyplot```:用來繪製影像
4. ```sklearn.model_selection import train_test_split```:將資料集隨機分配成訓練集和測試集
5. ```sklearn.model_selection import GridSearchCV```:搜索指定參數的估計值
6. ```sklearn.datasets import fetch_lfw_people```:載入LFW人臉資料庫
7. ```sklearn.metrics import classification_report```:建立文字報告，顯示主要的分類矩陣
8. ```sklearn.metrics import confusion_matrix```:計算混淆矩陣以評估分類的準確性
9. ```sklearn.decomposition import PCA```:進行主成分分析
10. ```sklearn.svm import SVC```:載入用於分類的向量支持模型

## (二)載入LFW人臉資料庫

將資料以```numpy array```形式存進```lfw_people```中，
其中```min_faces_per_person=70 ```指提取的數據集將僅保留具有至少70個不同圖片的人的圖片。

```python
# 下載資料(如果並未下載於電腦中)

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
```

此範例中共有1288張影像，每張影像大小為62 x 47像素
```python
# 查詢影像的大小(為了畫圖)
n_samples, h, w = lfw_people.images.shape

# 為了機器學習，我們直接使用這兩個資料(這個模型忽略了相對像素的位置信息)
X = lfw_people.data
n_features = X.shape[1]

# 要預測的標籤是該人的ID
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)
```
將資料集隨機分配成訓練集和測試集
```python
# 分成訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)
```

## (三)對於人臉資料計算PCA

計算人臉資料集中的PCA(特徵臉)，視為未標籤的資料:使用非監督式提取降維。


```python
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)
```
* ```svd_solver='randomized'```：用Halko方法運行隨機SVD
* ```whiten=True```：將```components```向量乘以n_samples的平方根並除以奇異值，以確保具有不相關的輸出。

```python
n_components = 150
print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))
t0 = time() #計時
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))

eigenfaces = pca.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
#進行降維
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))
```
## (四)訓練SVM分類模型
SVM模型有兩個非常重要的參數C與gamma。
<br />C:懲罰係數，即對誤差的寬容度。c越高，說明越不能容忍出現誤差，容易過擬合。
<br />gamma:選擇RBF函數作為kernel後，該函數自帶的一個參數。隱含地決定了數據映射到新的特徵空間後的分佈。
```python
print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(
    SVC(kernel='rbf', class_weight='balanced'), param_grid
)
```
* ```skernel='rbf'```：使用（高斯）徑向基函數
* ```param_grid```：給定SVM參數
```python
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)
```
## (五)對測試集中進行預測

使用```y_pred = clf.predict(X_test_pca)```，對測試集進行預測。
```python
print("Predicting people's names on the test set")
t0 = time()
#用最佳發現的參數對評估器進行預測。
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))

print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
```

## (六)使用matplotlib對預測進行評估
```python
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """為了畫出人像的函數"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# 在部分測試集中繪製預測結果
def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w)

# 繪製最有意義的特徵臉
eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()
```
![](https://github.com/JENNSHIUAN/machine-learning-python/blob/master/application/Faces_recognition/Face_fig2.JPG)
## Total Output:

![](https://github.com/JENNSHIUAN/machine-learning-python/blob/master/application/Faces_recognition/Face_fig3.JPG)

## (七)完整程式碼
Python source code:plot_face_recognition.py

https://scikit-learn.org/stable/_downloads/fcbed4be5eadd64ee8f4961f64b1904c/plot_face_recognition.py

```python
from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC




print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


# #############################################################################
# Download the data, if not already on disk and load it as numpy arrays

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)


# #############################################################################
# Split into a training set and a test set using a stratified k fold

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)


# #############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
n_components = 150

print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))

eigenfaces = pca.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))


# #############################################################################
# Train a SVM classification model

print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(
    SVC(kernel='rbf', class_weight='balanced'), param_grid
)
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)


# #############################################################################
# Quantitative evaluation of the model quality on the test set

print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))

print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))


# #############################################################################
# Qualitative evaluation of the predictions using matplotlib

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# plot the result of the prediction on a portion of the test set

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w)

# plot the gallery of the most significative eigenfaces

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()
```
