# **範例四:Plot randomly generated multilabel dataset**

https://scikit-learn.org/stable/auto_examples/datasets/plot_random_multilabel_dataset.html

這個範例示範了如何使用` make_multilabel_classification`函數，每個樣本都包含兩個特徵的計數（總共最多50個），
這兩個特徵在兩個類別的每個類別中的分佈不同。


點的標記如下，其中Y表示類別是否存在：

![](https://github.com/JENNSHIUAN/machine-learning-python/blob/master/Datasets/ex4_fig1.JPG)

設定分類的顏色

```python
COLORS = np.array(['!',
                   '#FF3333',  # red
                   '#0198E1',  # blue
                   '#BF5FFF',  # purple
                   '#FCD116',  # yellow
                   '#FF7216',  # orange
                   '#4DBD33',  # green
                   '#87421F'   # brown
                   ])
```

從0~1024中隨機設定種子，使用相同的隨機種子多次調用`make_ml_clf`，確保相同的分佈

```python
RANDOM_SEED = np.random.randint(2 ** 10)
```

## (一)Make multilabel classification
使用`make_ml_clf`生成隨機的多標籤分類，其中回傳四個變數:
<br />X 表示產生的樣本
<br />Y 表示標籤的集合
<br />p_c 表示每個分類被選中的機率
<br />p_w_c 表示給定每一個分類，特徵被選中的機率

```python
 X, Y, p_c, p_w_c = make_ml_clf(n_samples=150, n_features=2,
                                  n_classes=n_classes, n_labels=n_labels,
                                  length=length, allow_unlabeled=False,
                                  return_distributions=True,
                                  random_state=RANDOM_SEED)

ax.scatter(X[:, 0], X[:, 1], color=COLORS.take((Y * [1, 2, 4]
                                                    ).sum(axis=1)),
              marker='.'
```

星號標記每個類別的預期樣本；它的大小反映了選擇該類別標籤的可能性。

```python
 ax.scatter(p_w_c[0] * length, p_w_c[1] * length,
               marker='*', linewidth=.5, edgecolor='black',
               s=20 + 1500 * p_c ** 2,
               color=COLORS.take([1, 2, 4]))
```

## (二)顯示圖形與結果
![](https://github.com/JENNSHIUAN/machine-learning-python/blob/master/Datasets/ex4_fig2.JPG)

![](https://github.com/JENNSHIUAN/machine-learning-python/blob/master/Datasets/ex4_fig3.JPG)


請注意，由於此範例過於簡化：特徵的數量通常會比“文檔長度”大得多，而此範例的文檔長度比特徵量大得多。也就是說`n_classes> n_features`，特徵要分辨特定分類的機率相對小得很多。

## (三)完整程式碼

Python source code:plot_random_multilabel_dataset.py

https://scikit-learn.org/stable/_downloads/e35860bbf32dbc6fb903781f623874e3/plot_random_multilabel_dataset.py
```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_multilabel_classification as make_ml_clf

print(__doc__)

COLORS = np.array(['!',
                   '#FF3333',  # red
                   '#0198E1',  # blue
                   '#BF5FFF',  # purple
                   '#FCD116',  # yellow
                   '#FF7216',  # orange
                   '#4DBD33',  # green
                   '#87421F'   # brown
                   ])

# Use same random seed for multiple calls to make_multilabel_classification to
# ensure same distributions
RANDOM_SEED = np.random.randint(2 ** 10)


def plot_2d(ax, n_labels=1, n_classes=3, length=50):
    X, Y, p_c, p_w_c = make_ml_clf(n_samples=150, n_features=2,
                                   n_classes=n_classes, n_labels=n_labels,
                                   length=length, allow_unlabeled=False,
                                   return_distributions=True,
                                   random_state=RANDOM_SEED)

    ax.scatter(X[:, 0], X[:, 1], color=COLORS.take((Y * [1, 2, 4]
                                                    ).sum(axis=1)),
               marker='.')
    ax.scatter(p_w_c[0] * length, p_w_c[1] * length,
               marker='*', linewidth=.5, edgecolor='black',
               s=20 + 1500 * p_c ** 2,
               color=COLORS.take([1, 2, 4]))
    ax.set_xlabel('Feature 0 count')
    return p_c, p_w_c


_, (ax1, ax2) = plt.subplots(1, 2, sharex='row', sharey='row', figsize=(8, 4))
plt.subplots_adjust(bottom=.15)

p_c, p_w_c = plot_2d(ax1, n_labels=1)
ax1.set_title('n_labels=1, length=50')
ax1.set_ylabel('Feature 1 count')

plot_2d(ax2, n_labels=3)
ax2.set_title('n_labels=3, length=50')
ax2.set_xlim(left=0, auto=True)
ax2.set_ylim(bottom=0, auto=True)

plt.show()

print('The data was generated from (random_state=%d):' % RANDOM_SEED)
print('Class', 'P(C)', 'P(w0|C)', 'P(w1|C)', sep='\t')
for k, p, p_w in zip(['red', 'blue', 'yellow'], p_c, p_w_c.T):
    print('%s\t%0.2f\t%0.2f\t%0.2f' % (k, p, p_w[0], p_w[1]))
```
