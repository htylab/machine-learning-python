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


'''python


'''