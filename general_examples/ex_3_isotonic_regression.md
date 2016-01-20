# 通用範例/範例三: Isotonic Regression

http://scikit-learn.org/stable/auto_examples/plot_isotonic_regression.html

迴歸函數採用遞增函數。

這個範例的主要目的：

比較

* Isotonic Fit
* Linear Fit


# (一) Regression「迴歸」
「迴歸」就是找一個函數，盡量符合手邊的一堆數據。此函數稱作「迴歸函數」。

# (二) Linear Regression「線性迴歸」
迴歸函數採用線性函數。誤差採用平方誤差。

二維數據，迴歸函數是直線。

![](images/Isotonic Regression_figure_1.png)


# (三) Isotonic Regression「保序迴歸」
具有分段迴歸的效果。迴歸函數採用遞增函數。

採用平方誤差，時間複雜度 O(N) 。

![](images/Isotonic Regression_figure_2.png)
