import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score, f1_score

# load dataset
df = pd.read_csv("./project/feature_data_usa.csv")
df.head()
# index number 제거
df.drop(["Unnamed: 0"], axis=1, inplace=True)

X = df.drop(["ET"], axis=1)
y = df["ET"]
n_1 = np.sum(y==1)
n_0 = np.sum(y==0)
r_state = 42

# resampler info
rsp_name = ["SMOTE"]
rsp = SMOTE(ratio={0: n_0, 1: n_1*3}, random_state=r_state)

# classifier info
clf_name = ["Decision Tree"]
clf = DecisionTreeClassifier(class_weight='balanced')

# Resampling
X_res, y_res = rsp.fit_sample(X, y)

# Split
X_train, trs1, y_train, trs2 = train_test_split(X_res, y_res, test_size=0, random_state=42)

# Model training
clf.fit(X_train, y_train)


# load seoul datast
df2 = pd.read_csv("./project/feature_data_seoul.csv")
# index number 제거
df2.drop(["Unnamed: 0"], axis=1, inplace=True)
df2.head()

X_test = df2.drop(["ET"], axis=1)

y_hat = clf.predict(X_test)

plt.plot(y_hat)
X_test.loc[np.where(y_hat==1)]
