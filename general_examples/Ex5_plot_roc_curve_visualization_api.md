# **範例五:ROC Curve with Visualization API**

https://scikit-learn.org/stable/auto_examples/plot_roc_curve_visualization_api.html

Scikit-learn定義了一個簡單的API，用於創建機器學習的可視化。該API的主要功能是無需重新計算即可進行快速繪圖和視覺調整。在此範例中，我們將通過比較ROC曲線來展示如何使用可視化API。


## (一)載入資料以及訓練SVC

首先，我們載入`load_wine`，它主要為一個典型且簡單的多分類資料庫，並將它轉換為二進位制的分類問題。

```python
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_roc_curve
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

X, y = load_wine(return_X_y=True)
y = y == 2
```
對於訓練資料訓練一個SVC。

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
svc = SVC(random_state=42)
svc.fit(X_train, y_train)
```

## (二)繪製ROC曲線

使用`sklearn.metrics.plot_roc_curve`來繪製ROC曲線，回傳的`svc_disp`對象使我們可以在以後的圖中繼續使用已經計算出的ROC曲線。

```python
svc_disp = plot_roc_curve(svc, X_test, y_test)
plt.show()
```
![](images/ex5_fig1.png)

## (三)訓練一個隨機森林並且繪製ROC曲線

我們訓練一個隨機森林分類器並繪製出ROC曲線來比較先前用SVC繪製的ROC曲線，值得注意的是，`svc_disp`使用`plot`繪製曲線，而無需重新計算ROC曲線本身的值。
此外，我們將alpha = 0.8傳遞給繪圖函數以調整曲線的alpha值。

```python
rfc = RandomForestClassifier(n_estimators=10, random_state=42)
rfc.fit(X_train, y_train)
ax = plt.gca()
rfc_disp = plot_roc_curve(rfc, X_test, y_test, ax=ax, alpha=0.8)
svc_disp.plot(ax=ax, alpha=0.8)
plt.show()
```
![](images/ex5_fig2.png)

## (四)完整程式碼

Python source code:plot_roc_curve_visualization_api.py
<br />https://scikit-learn.org/stable/_downloads/e2f118a17ad70541f445f35934fdbb99/plot_roc_curve_visualization_api.py

```python
================================
ROC Curve with Visualization API
================================
Scikit-learn defines a simple API for creating visualizations for machine
learning. The key features of this API is to allow for quick plotting and
visual adjustments without recalculation. In this example, we will demonstrate
how to use the visualization API by comparing ROC curves.
"""
print(__doc__)

##############################################################################
# Load Data and Train a SVC
# -------------------------
# First, we load the wine dataset and convert it to a binary classification
# problem. Then, we train a support vector classifier on a training dataset.
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_roc_curve
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

X, y = load_wine(return_X_y=True)
y = y == 2

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
svc = SVC(random_state=42)
svc.fit(X_train, y_train)

##############################################################################
# Plotting the ROC Curve
# ----------------------
# Next, we plot the ROC curve with a single call to
# :func:`sklearn.metrics.plot_roc_curve`. The returned `svc_disp` object allows
# us to continue using the already computed ROC curve for the SVC in future
# plots.
svc_disp = plot_roc_curve(svc, X_test, y_test)
plt.show()

##############################################################################
# Training a Random Forest and Plotting the ROC Curve
# --------------------------------------------------------
# We train a random forest classifier and create a plot comparing it to the SVC
# ROC curve. Notice how `svc_disp` uses
# :func:`~sklearn.metrics.RocCurveDisplay.plot` to plot the SVC ROC curve
# without recomputing the values of the roc curve itself. Furthermore, we
# pass `alpha=0.8` to the plot functions to adjust the alpha values of the
# curves.
rfc = RandomForestClassifier(n_estimators=10, random_state=42)
rfc.fit(X_train, y_train)
ax = plt.gca()
rfc_disp = plot_roc_curve(rfc, X_test, y_test, ax=ax, alpha=0.8)
svc_disp.plot(ax=ax, alpha=0.8)
plt.show()
```
