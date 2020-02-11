## 半監督式分類法/範例4 : Label Propagation digits active learning

本範例目的：
* 展示active learning(主動學習)進行以label propagation(標籤傳播法)學習辨識手寫數字

## 一、Active Learning 主動學習
在實際應用上，通常我們獲得到的數據，有一大部分是未標籤的，如果要套用在常用的分類法上，最直接的想法是標籤所有的數據，但一一標籤所有數據是非常耗時耗工的，因此，在面對未標籤的數據遠多於有標籤的數據之情況下，可以透過active learning，主動的挑選一些數據進行標籤。
Active learning分成兩部分：
* 從已標籤的數據中隨機抽取一小部分作為訓練集，訓練出一個分類模型
* 透過迭代，將分類器預測出來的結果再進行訓練。

## 二、引入函式與模型
* stats用來進行統計與分析
* LabelSpreading為半監督式學習的模型
* confusion_matrix為混淆矩陣
* classification_report用於觀察預測和實際數值的差異，包含precision、recall、f1-score及support

```python
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from sklearn import datasets
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import classification_report, confusion_matrix
```

## 三、建立dataset
* Dataset取自sklearn.datasets.load_digits，內容為0~9的手寫數字，共有1797筆
* 使用其中的330筆進行訓練(y_train)，其中40筆為labeled，其餘290筆為unlabeled(標為-1)
* 迭代的次數設定為5次
* scikit learn網站中的範例程式敘述為10筆labeled，但原始程式碼為40筆，因此在這邊以原始碼為主

```python
digits = datasets.load_digits()
rng = np.random.RandomState(0)
indices = np.arange(len(digits.data))
rng.shuffle(indices)

X = digits.data[indices[:330]]
y = digits.target[indices[:330]]
images = digits.images[indices[:330]]

n_total_samples = len(y)
n_labeled_points = 40
max_iterations = 5

unlabeled_indices = np.arange(n_total_samples)[n_labeled_points:]
f = plt.figure()
```

## 四、利用Active learning進行模型訓練與預測
* 以下程式為每一次迭代所做的過程(for迴圈的內容)
* 每一次迭代都利用訓練過後的模型進行預測，得到predicted_labels，並與true_labels計算混淆矩陣與classification report

```python
if len(unlabeled_indices) == 0:
    print("No unlabeled items left to label.")
    break
y_train = np.copy(y)
y_train[unlabeled_indices] = -1

lp_model = LabelSpreading(gamma=0.25, max_iter=20)
lp_model.fit(X, y_train)

predicted_labels = lp_model.transduction_[unlabeled_indices]
true_labels = y[unlabeled_indices]

cm = confusion_matrix(true_labels, predicted_labels,
                      labels=lp_model.classes_)

print("Iteration %i %s" % (i, 70 * "_"))
print("Label Spreading model: %d labeled & %d unlabeled (%d total)"
      % (n_labeled_points, n_total_samples - n_labeled_points,
         n_total_samples))

print(classification_report(true_labels, predicted_labels))

print("Confusion matrix")
print(cm)
```

* 利用stats進行數據的統計，找出前5筆預測最不佳的結果，將其預測的label與true label和圖像顯示出來
* 每一次迭代的最後挑出上述的5筆預測最不佳的結果，進行下一次的迭代時，把相對應的true label替換給y_train測試集裡面，其餘(第40筆之後的數據)的label依然給予-1表示unlabeled

```python
# compute the entropies of transduced label distributions
pred_entropies = stats.distributions.entropy(
    lp_model.label_distributions_.T)

# select up to 5 digit examples that the classifier is most uncertain about
uncertainty_index = np.argsort(pred_entropies)[::-1]
uncertainty_index = uncertainty_index[
    np.in1d(uncertainty_index, unlabeled_indices)][:5]

# keep track of indices that we get labels for
delete_indices = np.array([], dtype=int)

# for more than 5 iterations, visualize the gain only on the first 5
if i < 5:
    f.text(.05, (1 - (i + 1) * .183),
           "model %d\n\nfit with\n%d labels" %
           ((i + 1), i * 5 + 40), size=10)
for index, image_index in enumerate(uncertainty_index):
    image = images[image_index]

    # for more than 5 iterations, visualize the gain only on the first 5
    if i < 5:
        sub = f.add_subplot(5, 5, index + 1 + (5 * i))
        sub.imshow(image, cmap=plt.cm.gray_r, interpolation='none')
        sub.set_title("predict: %i\ntrue: %i" % (
            lp_model.transduction_[image_index], y[image_index]), size=10)
        sub.axis('off')

    # labeling 5 points, remote from labeled set
    delete_index, = np.where(unlabeled_indices == image_index)
    delete_indices = np.concatenate((delete_indices, delete_index))

unlabeled_indices = np.delete(unlabeled_indices, delete_indices)
n_labeled_points += len(uncertainty_index)
```

* 下列程式屬於for迴圈外圍

```python
f.suptitle("Active learning with Label Propagation.\nRows show 5 most "
           "uncertain labels to learn with the next model.", y=1.15)
plt.subplots_adjust(left=0.2, bottom=0.03, right=0.9, top=0.9, wspace=0.2,
                    hspace=0.85)
plt.show()
```

* 以下即為每一次迭代的結果，可以看到每一次迭代之後，micro avg逐漸上升

Out:

      Iteration 0 ______________________________________________________________________
      Label Spreading model: 40 labeled & 290 unlabeled (330 total)
                    precision    recall  f1-score   support

                 0       1.00      1.00      1.00        22
                 1       0.78      0.69      0.73        26
                 2       0.93      0.93      0.93        29
                 3       1.00      0.89      0.94        27
                 4       0.92      0.96      0.94        23
                 5       0.96      0.70      0.81        33
                 6       0.97      0.97      0.97        35
                 7       0.94      0.91      0.92        33
                 8       0.62      0.89      0.74        28
                 9       0.73      0.79      0.76        34

         micro avg       0.87      0.87      0.87       290
         macro avg       0.89      0.87      0.87       290
      weighted avg       0.88      0.87      0.87       290

      Confusion matrix
      [[22  0  0  0  0  0  0  0  0  0]
       [ 0 18  2  0  0  0  1  0  5  0]
       [ 0  0 27  0  0  0  0  0  2  0]
       [ 0  0  0 24  0  0  0  0  3  0]
       [ 0  1  0  0 22  0  0  0  0  0]
       [ 0  0  0  0  0 23  0  0  0 10]
       [ 0  1  0  0  0  0 34  0  0  0]
       [ 0  0  0  0  0  0  0 30  3  0]
       [ 0  3  0  0  0  0  0  0 25  0]
       [ 0  0  0  0  2  1  0  2  2 27]]
      Iteration 1 ______________________________________________________________________
      Label Spreading model: 45 labeled & 285 unlabeled (330 total)
                    precision    recall  f1-score   support

                 0       1.00      1.00      1.00        22
                 1       0.79      1.00      0.88        22
                 2       1.00      0.93      0.96        29
                 3       1.00      1.00      1.00        26
                 4       0.92      0.96      0.94        23
                 5       0.96      0.70      0.81        33
                 6       1.00      0.97      0.99        35
                 7       0.94      0.91      0.92        33
                 8       0.77      0.86      0.81        28
                 9       0.73      0.79      0.76        34

         micro avg       0.90      0.90      0.90       285
         macro avg       0.91      0.91      0.91       285
      weighted avg       0.91      0.90      0.90       285

      Confusion matrix
      [[22  0  0  0  0  0  0  0  0  0]
       [ 0 22  0  0  0  0  0  0  0  0]
       [ 0  0 27  0  0  0  0  0  2  0]
       [ 0  0  0 26  0  0  0  0  0  0]
       [ 0  1  0  0 22  0  0  0  0  0]
       [ 0  0  0  0  0 23  0  0  0 10]
       [ 0  1  0  0  0  0 34  0  0  0]
       [ 0  0  0  0  0  0  0 30  3  0]
       [ 0  4  0  0  0  0  0  0 24  0]
       [ 0  0  0  0  2  1  0  2  2 27]]
      Iteration 2 ______________________________________________________________________
      Label Spreading model: 50 labeled & 280 unlabeled (330 total)
                    precision    recall  f1-score   support

                 0       1.00      1.00      1.00        22
                 1       0.85      1.00      0.92        22
                 2       1.00      1.00      1.00        28
                 3       1.00      1.00      1.00        26
                 4       0.87      1.00      0.93        20
                 5       0.96      0.70      0.81        33
                 6       1.00      0.97      0.99        35
                 7       0.94      1.00      0.97        32
                 8       0.92      0.86      0.89        28
                 9       0.73      0.79      0.76        34

         micro avg       0.92      0.92      0.92       280
         macro avg       0.93      0.93      0.93       280
      weighted avg       0.93      0.92      0.92       280

      Confusion matrix
      [[22  0  0  0  0  0  0  0  0  0]
       [ 0 22  0  0  0  0  0  0  0  0]
       [ 0  0 28  0  0  0  0  0  0  0]
       [ 0  0  0 26  0  0  0  0  0  0]
       [ 0  0  0  0 20  0  0  0  0  0]
       [ 0  0  0  0  0 23  0  0  0 10]
       [ 0  1  0  0  0  0 34  0  0  0]
       [ 0  0  0  0  0  0  0 32  0  0]
       [ 0  3  0  0  1  0  0  0 24  0]
       [ 0  0  0  0  2  1  0  2  2 27]]
      Iteration 3 ______________________________________________________________________
      Label Spreading model: 55 labeled & 275 unlabeled (330 total)
                    precision    recall  f1-score   support

                 0       1.00      1.00      1.00        22
                 1       0.85      1.00      0.92        22
                 2       1.00      1.00      1.00        27
                 3       1.00      1.00      1.00        26
                 4       0.87      1.00      0.93        20
                 5       0.96      0.87      0.92        31
                 6       1.00      0.97      0.99        35
                 7       1.00      1.00      1.00        31
                 8       0.92      0.86      0.89        28
                 9       0.88      0.85      0.86        33

         micro avg       0.95      0.95      0.95       275
         macro avg       0.95      0.95      0.95       275
      weighted avg       0.95      0.95      0.95       275

      Confusion matrix
      [[22  0  0  0  0  0  0  0  0  0]
       [ 0 22  0  0  0  0  0  0  0  0]
       [ 0  0 27  0  0  0  0  0  0  0]
       [ 0  0  0 26  0  0  0  0  0  0]
       [ 0  0  0  0 20  0  0  0  0  0]
       [ 0  0  0  0  0 27  0  0  0  4]
       [ 0  1  0  0  0  0 34  0  0  0]
       [ 0  0  0  0  0  0  0 31  0  0]
       [ 0  3  0  0  1  0  0  0 24  0]
       [ 0  0  0  0  2  1  0  0  2 28]]
      Iteration 4 ______________________________________________________________________
      Label Spreading model: 60 labeled & 270 unlabeled (330 total)
                    precision    recall  f1-score   support

                 0       1.00      1.00      1.00        22
                 1       0.96      1.00      0.98        22
                 2       1.00      0.96      0.98        27
                 3       0.96      1.00      0.98        25
                 4       0.86      1.00      0.93        19
                 5       0.96      0.87      0.92        31
                 6       1.00      0.97      0.99        35
                 7       1.00      1.00      1.00        31
                 8       0.92      0.96      0.94        25
                 9       0.88      0.85      0.86        33

         micro avg       0.96      0.96      0.96       270
         macro avg       0.95      0.96      0.96       270
      weighted avg       0.96      0.96      0.96       270

      Confusion matrix
      [[22  0  0  0  0  0  0  0  0  0]
       [ 0 22  0  0  0  0  0  0  0  0]
       [ 0  0 26  1  0  0  0  0  0  0]
       [ 0  0  0 25  0  0  0  0  0  0]
       [ 0  0  0  0 19  0  0  0  0  0]
       [ 0  0  0  0  0 27  0  0  0  4]
       [ 0  1  0  0  0  0 34  0  0  0]
       [ 0  0  0  0  0  0  0 31  0  0]
       [ 0  0  0  0  1  0  0  0 24  0]
       [ 0  0  0  0  2  1  0  0  2 28]]


![png](ex4_output_result.png)

上圖的結果即為Active Learning訓練過程的結果，第一次迭代以330筆的資料進行訓練，其中包含40筆labeled的資料與290 unlabeled的資料，再對unlabeled的資料做預測，將預測出來的結果中，5個預測最不佳的結果顯示出來，即第一列的5張圖，將這5筆資料的從測試集中強制變為true label的結果，再下一次迭代中，labeled的資料就變成45筆，unlabeled的資料為285筆，總和為330筆的資料進行第二次的訓練，以此類推，因此可以看到，每一次訓練，labeled的資料會5筆、5筆的增加。

## 五、原始碼列表
Python source code: plot_label_propagation_digits_active_learning.py

https://scikit-learn.org/stable/auto_examples/semi_supervised/plot_label_propagation_digits_active_learning.html

```python
print(__doc__)

# Authors: Clay Woolam <clay@woolam.org>
# License: BSD

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from sklearn import datasets
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import classification_report, confusion_matrix

digits = datasets.load_digits()
rng = np.random.RandomState(0)
indices = np.arange(len(digits.data))
rng.shuffle(indices)

X = digits.data[indices[:330]]
y = digits.target[indices[:330]]
images = digits.images[indices[:330]]

n_total_samples = len(y)
n_labeled_points = 40
max_iterations = 5

unlabeled_indices = np.arange(n_total_samples)[n_labeled_points:]
f = plt.figure()

for i in range(max_iterations):
    if len(unlabeled_indices) == 0:
        print("No unlabeled items left to label.")
        break
    y_train = np.copy(y)
    y_train[unlabeled_indices] = -1

    lp_model = LabelSpreading(gamma=0.25, max_iter=20)
    lp_model.fit(X, y_train)

    predicted_labels = lp_model.transduction_[unlabeled_indices]
    true_labels = y[unlabeled_indices]

    cm = confusion_matrix(true_labels, predicted_labels,
                          labels=lp_model.classes_)

    print("Iteration %i %s" % (i, 70 * "_"))
    print("Label Spreading model: %d labeled & %d unlabeled (%d total)"
          % (n_labeled_points, n_total_samples - n_labeled_points,
             n_total_samples))

    print(classification_report(true_labels, predicted_labels))

    print("Confusion matrix")
    print(cm)

    # compute the entropies of transduced label distributions
    pred_entropies = stats.distributions.entropy(
        lp_model.label_distributions_.T)

    # select up to 5 digit examples that the classifier is most uncertain about
    uncertainty_index = np.argsort(pred_entropies)[::-1]
    uncertainty_index = uncertainty_index[
        np.in1d(uncertainty_index, unlabeled_indices)][:5]

    # keep track of indices that we get labels for
    delete_indices = np.array([], dtype=int)

    # for more than 5 iterations, visualize the gain only on the first 5
    if i < 5:
        f.text(.05, (1 - (i + 1) * .183),
               "model %d\n\nfit with\n%d labels" %
               ((i + 1), i * 5 + 10), size=10)
    for index, image_index in enumerate(uncertainty_index):
        image = images[image_index]

        # for more than 5 iterations, visualize the gain only on the first 5
        if i < 5:
            sub = f.add_subplot(5, 5, index + 1 + (5 * i))
            sub.imshow(image, cmap=plt.cm.gray_r, interpolation='none')
            sub.set_title("predict: %i\ntrue: %i" % (
                lp_model.transduction_[image_index], y[image_index]), size=10)
            sub.axis('off')

        # labeling 5 points, remote from labeled set
        delete_index, = np.where(unlabeled_indices == image_index)
        delete_indices = np.concatenate((delete_indices, delete_index))

    unlabeled_indices = np.delete(unlabeled_indices, delete_indices)
    n_labeled_points += len(uncertainty_index)

f.suptitle("Active learning with Label Propagation.\nRows show 5 most "
           "uncertain labels to learn with the next model.", y=1.15)
plt.subplots_adjust(left=0.2, bottom=0.03, right=0.9, top=0.9, wspace=0.2,
                    hspace=0.85)
plt.show()
```
