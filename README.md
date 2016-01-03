#機器學習：使用Python

這份文件的目的是要提供Python 之機器學習套件 scikit-learn (http://scikit-learn.org/) 的中文使用說明。一開始的主要目標是詳細說明scikit-learn套件中的[範例程式](http://scikit-learn.org/stable/auto_examples/index.html )的使用流程以及相關函式的使用方法。目前使用版本為 scikit-learn version 0.17 以上


本書原始資料在 Github 上公開，歡迎大家共同參與維護： [https://github.com/htygithub/machine-learning-python](https://github.com/htygithub/machine-learning-python)。

## 本文件主要的版本發展
* 0.0: 2015/12/21
    * 開始本文件的撰寫
    * 初期以scikit-learn套件的範例介紹為主軸

##  Scikit-learn 套件

Scikit-learn (http://scikit-learn.org/) 是一個機器學習領域的開源套件。整個專案起始於 2007年由David Cournapeau所執行的`Google Summer of Code` 計畫。而2010年之後，則由法國國家資訊暨自動化研究院（INRIA, http://www.inria.fr） 繼續主導及後續的支援及開發。近幾年(2013-2015)則由 INRIA 支持 Olivier Grisel (http://ogrisel.com) 全職負責該套件的維護工作。以開發者的角度來觀察，會發現Scikit-learn的整套使用邏輯設計的極其簡單。往往能將繁雜的機器學習理論簡化到一個步驟完成。


