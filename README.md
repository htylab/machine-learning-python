#機器學習：使用Python

這份文件的目的是要提供Python 之機器學習套件 scikit-learn (http://scikit-learn.org/) 的中文使用說明。一開始的主要目標是詳細說明scikit-learn套件中的[範例程式](http://scikit-learn.org/stable/auto_examples/index.html )的使用流程以及相關函式的使用方法。目前使用版本為 scikit-learn version 0.18 以上


本書原始資料在 Github 上公開，歡迎大家共同參與維護： [https://github.com/htygithub/machine-learning-python](https://github.com/htygithub/machine-learning-python)。

## 本文件主要的版本發展
* 0.0: 2015/12/21
    * 開始本文件「機器學習：使用Python」的撰寫
    * 初期以scikit-learn套件的範例介紹為主軸
* 0.1: 2016/4/15
    * 「機器學習：使用Python」文件
    *  Contributor: 陳巧寧、曾裕勝、黃騰毅 、蔡奕甫
    *  新增章節: Classification, Clustering, cross_decomposition, Datasets, feature_selection, general_examples
    *  新增 introduction: 說明簡易的Anaconda安裝，以及利用數字辨識範例來入門機器學習的方法
    *  第 10,000個 pageview 達成
![](images/pg10000.PNG)
* 0.2: 2016/8/30
    *  新增應用章節，Contributor: 吳尚真
    *  增修章節: Classification, Datasets, feature_selection, general_examples
* 0.3: 2017/2/16
    *  新增應用章節，Contributor: 楊采玲、歐育年
    *  增修章節: Neural_Network, Decision tree
    *  2016年，使用者約四萬人次，頁面流量約15萬次。
![](images/2016year.PNG)
##  Scikit-learn 套件

[Scikit-learn](http://scikit-learn.org/) 是一個機器學習領域的開源套件。整個專案起始於2007年，由[David Cournapeau](https://github.com/cournape)所執行的[Google Summer of Code](https://developers.google.com/open-source/gsoc/)計畫。2010年之後，則由法國國家資訊暨自動化研究院（[INRIA]( http://www.inria.fr)繼續主導及後續的支援與開發。近幾年(2013-2015)則由INRIA支持[Olivier Grisel](http://ogrisel.com)全職負責該套件的維護工作。從開發者的角度切入，會發現Scikit-learn使用的邏輯設計極其簡單。往往能將繁雜的機器學習過程簡化到幾個步驟完成。Python上與機器學習相關的套件數量相當多。為何首選Scikit-learn進行介紹呢？其實，一個開源套件的選擇，最簡易的方法即是審視其**貢獻者(contributor)數量**、**版本(commit)數量**及最近的更新日期。下圖為2016/1/3跨年夜後，筆者於[Scikit-learn官方專案網站](https://github.com/scikit-learn/scikit-learn)所擷取的畫面。我們可以發現最新的版本更新是四小時前，且貢獻者及版本數量分別來到531人及20,331個。由此可知，截至2016年為止，這個專案仍然非常積極的在運作著。相比其他眾多機器學習套件，Scikit-learn無論是貢獻者數量或是版本數量皆是最多的。因此，本文件以此作為介紹機器學習的切入點。未來，我們希望能介紹更多的機器學習套件及背後的機器學習理論，也歡迎有志之士共同參與維護。

![](images/sklearn_intro.PNG)
