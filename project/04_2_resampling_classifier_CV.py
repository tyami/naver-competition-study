import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score, f1_score
from sklearn.cross_validation import cross_val_score

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
rsp_names = ["Random Under Sampler", "Random Over Sampler", "SMOTE"]
resamplers = [RandomUnderSampler(ratio={0: n_1*3, 1: n_1}, random_state=r_state),
             RandomOverSampler(ratio={0: n_0, 1: n_1*3}, random_state=r_state),
             SMOTE(ratio={0: n_0, 1: n_1*3}, random_state=r_state) ]

# classifier info
clf_names = ["Linear SVM", "Decision Tree", "Random Forest"]
classifiers = [ SVC(kernel="linear", class_weight='balanced', C=0.025, probability=True),
                DecisionTreeClassifier(class_weight='balanced'),
                RandomForestClassifier(class_weight='balanced')]

result_clf_names = []
result_rsp_names = []
result_scores = []
result_aucs = []
result_f1s = []

# iterate over resamplers
for rsp_name, rsp in zip(rsp_names, resamplers):
    # Resampling
    X_res, y_res = rsp.fit_sample(X, y)

    # iterate over classifiers
    for clf_name, clf in zip(clf_names, classifiers):
        # Model training
        score = cross_val_score(clf, X_res, y_res, cv=10).mean() * 100
        result_scores.append(score)

        # AUC
        auc = cross_val_score(clf, X_res, y_res, cv=10, scoring='roc_auc').mean()
        result_aucs.append(score)

        # F1 score
        f1 = cross_val_score(clf, X_res, y_res, cv=10, scoring='f1_macro').mean()
        result_f1s.append(score)

        result_clf_names.append(clf_name)
        result_rsp_names.append(rsp_name)

        print("Accuracy: {:.3f}, AUC: {:.3f}, F1 score: {:.3f} ".format(score, auc, f1))
