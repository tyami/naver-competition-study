y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

# mean abolute error: 0에 가까울 수록 좋은 값.
from sklearn.metrics import median_absolute_error
median_absolute_error(y_true, y_pred)

# mean squared error: 0에 가까울 수록 좋은 값.
from sklearn.metrics import mean_squared_error
mean_squared_error(y_true, y_pred)

# R square: 1에 가까울 수록 좋은 값.
from sklearn.metrics import r2_score
r2_score(y_true, y_pred)

# Hold out
import numpy as np
from sklearn.model_selection import train_test_split

X, y = np.arange(10).reshape((5,2)), range(5)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=1/3, random_state=42)
