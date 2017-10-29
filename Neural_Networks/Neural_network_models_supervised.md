+<script>
 +MathJax.Hub.Queue(["Typeset",MathJax.Hub]);
 +</script>
# Multi-layer Perceptron(多層感知器)
http://scikit-learn.org/stable/modules/neural_networks_supervised.html

Multi-layer Perceptron (MLP):MLP為一種監督式學習的演算法，藉由 f(⋅):R^m→R^o ，m是輸入時的維度、o是輸出時的維度，藉由輸入特徵 $$X=x_1,x_2,.....,x_m$$ 和目標值Y，此算法將可以使用非線性近似將資料分類或進行迴歸運算。MLP可以在輸入層與輸出層中間插入許多非線性層，如圖1所示:，這是有一層隱藏層的網路。



![](images/multilayerperceptron_network.png)
<center>圖1:包含一層隱藏層的MLP</center>

最左邊那層稱作輸入層，為一個神經元集合 \{x_i|x_1,x_2,...,x_m\}代表輸入的特徵。每個神經元在隱藏層會根據前一層的輸出的結果，做為此層的輸入$$w_1x_1+w_2x_2+...+w_mx_m$$在將總和使用非線性的活化函數做 f(⋅):R→R轉換，例如:[hyperbolic tan function](https://en.wikipedia.org/wiki/Hyperbolic_function#/media/File:Sinh_cosh_tanh.svg)、[Sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function)，最右邊那層為輸出層，會接收最後的隱藏層的輸出在轉換一次成輸出值。

intercepts_為模型訓練後，權重矩陣內包含兩個屬性:$coefs\_$和$intercepts\_$。coefs\_ 此矩陣中第i個指標表示第$i$層與$i+1$層的權重，intercepts_為偏權值(bias)矩陣，此矩陣中第i個指標表示要加在$i+1層$的偏權值。

MLP優點:<br\>
    1.有能力建立非線性的模型<br\>
    2.可以使用$partial\_fit$建立real-time模型<br\>
MLP缺點:<br\>
1.因為[凹函數](https://zh.wikipedia.org/wiki/%E5%87%B9%E5%87%BD%E6%95%B0)擁有大於一個區域最小值，使用不同的初始權重，會讓驗證時的準確率浮動<br\>
2.MLP模型需要調整每層神經元數、層數、疊代次數<br\>
3.MLP對於特徵的預先處理很敏感，建議將特徵X都尺度降至[0,1]或[-1,+1]或讓特徵值降至平均值等於0與變異數等於1的數字區間

## MLP分類器
使用MLP訓練需要使用輸入兩種陣列，一個是特徵X陣列，X陣列包含(樣本數，特徵數)，另一個是Y向量包含目標值(分類標籤)下面將會介紹MLP分類器範例。

### (一)引入函式庫
from sklearn.neural_network import MLPClassifier:引進MLP分類器<br\>
### (二)建立模擬資料與設定分類器參數
建立擁有三種特徵的三筆資料<br\>
X = [[0., 0.,0.], [1., 1.,1.],[2., 2.,2.]] <br\>
將三筆資料的分類標上<br\>
y = [0, 1, 2]<br\>
設定分類器:最佳化參數的演算法，alpha值，隱藏層的層數與每層神經元數: hidden_layer_sizes=(5,3)表示隱藏層有兩層第一層為五個神經元，第二層為三個神經元<br\>
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(5,3), random_state=1)
### (三)訓練網路參數與預測
<br\>將資料丟進分類器，訓練網路參數<br\>
clf.fit(X, y)    
<br\>將要預測的資料丟進網路預測<br\>
clf.predict([[2., 2., 2.], [-1., -2.,0.],[1., 1.,0.]])
<br\>預測結果:array([2, 0, 1])
<br\>結果表示[2., 2., 2.]為第三類，[-1., -2.,0.]為第一類，[1., 1.,0.]為第二類


### 完整程式碼:

```python
from sklearn.neural_network import MLPClassifier

X = [[0., 0.,0.], [1., 1.,1.],[2., 2.,2.]]

y = [0, 1, 2]

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(5,3), random_state=1)

clf.fit(X, y)    

clf.predict([[2., 2., 2.], [-1., -2.,0.],[1., 1.,0.]])
```
