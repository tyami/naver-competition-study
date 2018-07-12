# IMPORT MODULES
import pandas as pd
import numpy as np
df = pd.read_csv("./test.csv")
df.head()

## LOAD DATASET - simple variable
X = df["x"].values.reshape(-1,1)
y = df["y"].values

print(X, y)

## BUILD MODEL
import solution_linear_model
import imp
imp.reload(solution_linear_model)

lr = solution_linear_model.LinearRegression(fit_intercept=True)
lr.fit(X, y)
lr.intercept
lr.coef
lr.predict(X)[:10]


## Validation
from sklearn import linear_model
sk_lr = linear_model.LinearRegression(normalize=False)
sk_lr.fit(X, y)

sk_lr.intercept_
import numpy.testing as npt
npt.assert_almost_equal(sk_lr.intercept_, lr.intercept)

sk_lr.coef_
np.isclose(lr.coef, sk_lr.coef_)
lr.predict(X)[:10]

## Train -> Test validation
df_test = pd.read_csv("./train.csv")
df_test.head()
X_test = df["x"].values.reshape(-1,1)
lr.predict(X_test)[:5]
sk_lr.predict(X_test)[:5]


## multiple variables validation
df = pd.read_csv("./mlr09.csv")
df.head()
y = df["average_points_scored"].values
df.iloc[:,:-1].head()
X = df.iloc[:,:-1].values
X[:5]

# rescale
mu_X = np.mean(X, axis=0)
std_X = np.std(X, axis=0)

rescaled_X = (X - mu_X) / std_X
rescaled_X[:5]


lr.fit(rescaled_X , y)
lr.coef
lr.intercept
sk_lr.fit(rescaled_X, y)
sk_lr.coef_
sk_lr.intercept_
