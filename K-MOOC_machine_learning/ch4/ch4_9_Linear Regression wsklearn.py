from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import numpy as np

boston = load_boston()
boston.keys()
print(boston["DESCR"])
boston["data"]
boston["target"]

x_data = boston.data
y_data = boston.target.reshape([boston.target.size,1])

x_data[:3]

from sklearn import preprocessing
minmax_scale = preprocessing.MinMaxScaler().fit(x_data)
x_scaled_data = minmax_scale.transform(x_data)

x_scaled_data[:3]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_scaled_data, y_data, test_size=0.3)

from sklearn import linear_model

regr = linear_model.LinearRegression(fit_intercept=True, # 상수항
                                    normalize=False, # 정규화
                                    copy_X=True, # 데이터 복사해서 분석. 일반적 True
                                    n_jobs=8) # CPU 개수

regr.fit(X_train, y_train)
regr.coef_, regr.intercept_

y_pred = regr.predict(X_test)

from sklearn.metrics import mean_squared_error
mean_squared_error(y_pred, y_test)
