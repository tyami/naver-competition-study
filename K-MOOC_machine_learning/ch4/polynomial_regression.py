import numpy as np
import matplotlib.pyplot as plt

def f(size):
    x = np.linspace(0, 5, size)
    y = x * np.sin(x ** 2) + 1
    return (x,y)


def sample(size):
    x = np.linspace(0, 5, size)
    y = x * np.sin(x ** 2) + 1 + np.random.randn(x.size) * 0.5
    return (x, y)

f_x, f_y = f(1000)
plt.plot(f_x, f_y)

X, y = sample(1000)
plt.scatter(X, y, s=3, c='black')
plt.show()

X.shape, y.shape

X = X.reshape(-1,1)
y = y.reshape(-1,1)
X.shape, y.shape


## Linear Regression은 불가능 !
from sklearn.linear_model import LinearRegression
lr = LinearRegression(fit_intercept=True)
lr.fit(X,y)
plt.plot(f_x, f_y)
plt.scatter(X.flatten(), y.flatten(), s=3, c='black')
plt.plot(X.flatten(), lr.predict(X).flatten())
plt.show()


## Polynomial을 써보자!
from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)
X_poly[:10]

lr = LinearRegression()
lr.fit(X_poly, y)

f_x, f_y = f(1000)
plt.plot(f_x, f_y)
plt.scatter(X.flatten(), y.flatten(), s=3, c="black")
plt.plot(X.flatten(), lr.predict(X_poly).flatten())
plt.show()

# 차수를 높여보자
poly_features = PolynomialFeatures(degree=16)
X_poly = poly_features.fit_transform(X)
X_poly[:10]

lr = LinearRegression()
lr.fit(X_poly, y)

f_x, f_y = f(1000)
plt.plot(f_x, f_y)
plt.scatter(X.flatten(), y.flatten(), s=3, c="black")
plt.plot(X.flatten(), lr.predict(X_poly).flatten())
plt.show()


## Lasso, ridge
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error

poly_range = list(range(10,50))
mse_lr = []
mse_lasso = []
mse_ridge = []
for poly_value in poly_range:
    poly_features = PolynomialFeatures(degree=poly_value)
    X_poly = poly_features.fit_transform(X)

    lr = LinearRegression()
    lr.fit(X_poly, y)

    mse_lr.append(mean_squared_error(lr.predict(X_poly), y))

    ridge = Ridge()
    ridge.fit(X_poly, y)

    mse_ridge.append(mean_squared_error(ridge.predict(X_poly), y))

    lasso = Lasso()
    lasso.fit(X_poly, y)

    mse_lasso.append(mean_squared_error(lasso.predict(X_poly), y))

import pandas as pd
from pandas import DataFrame

data = {"poly_range":poly_range, "mse_lr":mse_lr,
        "mse_ridge":mse_ridge, "mse_lasso":mse_lasso}
df = DataFrame(data).set_index("poly_range")
df

plt.plot(poly_range, df["mse_lr"], label="lr")
plt.plot(poly_range, df["mse_ridge"], label="ridge")
plt.plot(poly_range, df["mse_lasso"], label="lasso")
plt.legend()
plt.show()

df.min()
df["mse_ridge"].sort_values().head()


## Exercise
df = pd.read_csv("./K-MOOC_machine_learning/ch4/yield.csv",sep="\t")
df.head()

X = df["Temp"]
y = df["Yield"]
X = X.reshape(-1,1)

mse = []
poly_range = range(2,10)
for poly_value in poly_range:
    poly_features = PolynomialFeatures(degree=poly_value)

    X_poly = poly_features.fit_transform(X)

    lr = LinearRegression()
    lr.fit(X_poly, y)

    mse.append(mean_squared_error(lr.predict(X_poly), y))

data = {"poly_range":poly_range, "mse":mse}
df2 = DataFrame(data).set_index("poly_range")
df2["mse"].sort_values()

# 5th order is the best
poly_features = PolynomialFeatures(degree=5)

X_poly = poly_features.fit_transform(X)

lr = LinearRegression()
lr.fit(X_poly, y)

plt.scatter(df["Temp"], df["Yield"])
plt.plot(X, lr.predict(X_poly))
plt.show()
