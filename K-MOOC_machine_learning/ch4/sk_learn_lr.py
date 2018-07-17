import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston
boston = load_boston()

boston.keys()

boston.feature_names

df = pd.DataFrame(boston.data, columns=boston.feature_names)
df.head()

X = df.values
y = boston.target

## Linear Regression with Normal Equation
from sklearn.linear_model import LinearRegression
lr_ne = LinearRegression(fit_intercept=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42)

lr_ne.fit(X_train, y_train)
y_hat = lr_ne.predict(X_test)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_hat, y_test)

plt.scatter(y_test, y_hat, s=10)
plt.xlabel("price: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")
plt.show()

lr_ne.coef_
boston.feature_names


## Linear Regression with SGD
from sklearn.linear_model import SGDRegressor
lr_SGD = SGDRegressor()

from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
std_scaler.fit(X)
X = df.values
y = boston.target
X_scaled = std_scaler.transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y,
                        test_size=1/3, random_state=42)

lr_SGD.fit(X_train, y_train)
y_hat = lr_SGD.predict(X_test)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_hat, y_test)
rmse = np.sqrt(mse)


plt.scatter(y_test, y_hat, s=10)
plt.xlabel("price: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")
plt.show()


## Linear Regression with Ridge & Lasso Regression
from sklearn.linear_model import Lasso, Ridge

X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size=1/3, random_state=42)

ridge = Ridge(fit_intercept=True, alpha=0.5)
ridge.fit(X_train, y_train)

lasso = Lasso(fit_intercept=True, alpha=0.5)
lasso.fit(X_train, y_train)

y_hat = ridge.predict(X_test)
mse = mean_squared_error(y_hat, y_test)

plt.scatter(y_test, y_hat, s=10)
plt.xlabel("price: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("[Ridge] Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")
plt.show()

y_hat = lasso.predict(X_test)
mse = mean_squared_error(y_hat, y_test)

plt.scatter(y_test, y_hat, s=10)
plt.xlabel("price: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("[Lasso] Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")
plt.show()

from sklearn.model_selection import KFold

print('Ridge Regression')
print('alpha\t RMSE_train\t RMSE_10cv\n')
alpha = np.linspace(.01,20,50)
t_rmse = np.array([])
cv_rmse = np.array([])

for a in alpha:
    ridge = Ridge(fit_intercept=True, alpha=a)

    # computing the RMSE on training data
    ridge.fit(X_train,y_train)
    p = ridge.predict(X_test)
    err = p-y_test
    total_error = np.dot(err,err)
    rmse_train = np.sqrt(total_error/len(p))

    # computing RMSE using 10-fold cross validation
    kf = KFold(10)
    xval_err = 0
    for train, test in kf.split(X):
        ridge.fit(X[train], y[train])
        p = ridge.predict(X[test])
        err = p - y[test]
        xval_err += np.dot(err,err)
    rmse_10cv = np.sqrt(xval_err/len(X))

    t_rmse = np.append(t_rmse, [rmse_train])
    cv_rmse = np.append(cv_rmse, [rmse_10cv])
    print('{:.3f}\t {:.4f}\t\t {:.4f}'.format(a,rmse_train,rmse_10cv))
plt.plot(alpha, t_rmse, label='RMSE-Train')
plt.plot(alpha, cv_rmse, label='RMSE_XVal')
plt.legend( ('RMSE-Train', 'RMSE_XVal') )
plt.ylabel('RMSE')
plt.xlabel('Alpha')
plt.show()

a = 0.3
for name,met in [
        ('linear regression', LinearRegression()),
        ('lasso', Lasso(fit_intercept=True, alpha=a)),
        ('ridge', Ridge(fit_intercept=True, alpha=a)),
        ]:
    met.fit(X_train,y_train)
    # p = np.array([met.predict(xi) for xi in x])
    p = met.predict(X_test)
    e = p-y_test
    total_error = np.dot(e,e)
    rmse_train = np.sqrt(total_error/len(p))

    kf = KFold(10)
    err = 0
    for train,test in kf.split(X):
        met.fit(X[train],y[train])
        p = met.predict(X[test])
        e = p-y[test]
        err += np.dot(e,e)

    rmse_10cv = np.sqrt(err/len(X))
    print('Method: %s' %name)
    print('RMSE on training: %.4f' %rmse_train)
    print('RMSE on 10-fold CV: %.4f' %rmse_10cv)
