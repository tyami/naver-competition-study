import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


# load dataset
df = pd.read_csv("./project/feature_data.csv")

# index number 제거
df.drop(["Unnamed: 0"], axis=1, inplace=True)

df.head()

# Separate input features (X) and target variable (y)
y = df.ET
X = df.drop(["ET"], axis=1)
n_1 = np.sum(y==1)
n_0 = np.sum(y==0)

r_state = 42

# Resampling
ratio = {0: n_0, 1: n_1*3}
from imblearn.over_sampling import SMOTE
resampler = SMOTE(ratio=ratio, random_state=r_state)
X, y = resampler.fit_sample(X,y)

# Train, Test 분리 시키기: hold out 0.3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
# Support Vector Machine
from sklearn.svm import SVC
clf = SVC(kernel='linear',
            class_weight='balanced', # penalize
            probability=True)

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(class_weight='balanced')

# Random Forest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(class_weight='balanced')

# class weight
clf.fit(X_train, y_train)

# Predict on training set
y_hat = clf.predict(X_test)

# Is our model still predicting just one class?
print( np.unique( y_hat ) )
# Confusion matrix
cnf_mat = confusion_matrix(y_test, y_hat)

import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

plot_confusion_matrix(cnf_mat, classes=["UFO X", "UFO"])
# 하나로만 응답하는 건 아니군.

# How's our accuracy?
print( accuracy_score(y_test, y_hat) )

# What about AUROC?
y_hat_prob = clf.predict_proba(X_test)
y_hat_prob = [p[1] for p in y_hat_prob]
print( roc_auc_score(y_test, y_hat_prob) )
