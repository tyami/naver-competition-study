import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load dataset
df = pd.read_csv("./project/feature_data.csv")
df.head()
# index number 제거
df.drop(["Unnamed: 0"], axis=1, inplace=True)

X = df.drop(["ET"], axis=1)
y = df["ET"]
n_1 = np.sum(y==1)
n_0 = np.sum(y==0)
print(n_0, n_1)

r_state = 42

# Under sampling: Random 1:1 ratio
from imblearn.under_sampling import RandomUnderSampler

ratio =  'auto'
X_res, y_res = RandomUnderSampler(ratio=ratio, random_state=r_state).fit_sample(X,y)
y_res.shape

# Under sampling: Random 3:1 ratio
ratio =  {0: n_1*3, 1: n_1}
X_res, y_res = RandomUnderSampler(ratio=ratio, random_state=r_state).fit_sample(X,y)
y_res.shape#


# Over sampling: Random 1:1 ratio
ratio = 'auto'
from imblearn.over_sampling import RandomOverSampler
X_res, y_res = RandomOverSampler(ratio=ratio, random_state=r_state).fit_sample(X,y)
y_res.shape

# Over sampling: Random 3 times
ratio = {0: n_0, 1: n_1*3}
from imblearn.over_sampling import RandomOverSampler
X_res, y_res = RandomOverSampler(ratio=ratio, random_state=r_state).fit_sample(X,y)
y_res.shape

# Over sampling: SMOTE 1:1 ratio
ratio = 'auto'
from imblearn.over_sampling import SMOTE
X_res, y_res = SMOTE(ratio=ratio, random_state=r_state).fit_sample(X,y)
y_res.shape

# Over sampling: Random 3 times
ratio = {0: n_0, 1: n_1*3}
from imblearn.over_sampling import SMOTE
X_res, y_res = SMOTE(ratio=ratio, random_state=r_state).fit_sample(X,y)
y_res.shape




from sklearn import datasets
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

from imblearn import over_sampling as os
from imblearn import pipeline as pl
from imblearn.metrics import classification_report_imbalanced

print(__doc__)

RANDOM_STATE = 42

# Generate a dataset
X, y = datasets.make_classification(n_classes=2, class_sep=2,
                                    weights=[0.1, 0.9], n_informative=10,
                                    n_redundant=1, flip_y=0, n_features=20,
                                    n_clusters_per_class=4, n_samples=5000,
                                    random_state=RANDOM_STATE)

pipeline = pl.make_pipeline(os.SMOTE(random_state=RANDOM_STATE),
                            LinearSVC(random_state=RANDOM_STATE))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=RANDOM_STATE)

# Train the classifier with balancing
pipeline.fit(X_train, y_train)

# Test the classifier and get the prediction
y_pred_bal = pipeline.predict(X_test)

# Show the classification report
print(classification_report_imbalanced(y_test, y_pred_bal))
