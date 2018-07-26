import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

data_path = './project/ufo-air-quality/UFOPOLLUTANTS.csv'
df = pd.read_csv(data_path)

df.head()
df.columns

# 계절 변수화
df["season"] = np.where((df["month"] < 3) | (df["month"] > 11), "winter",
                   np.where(df["month"] < 6, "spring",
                               np.where(df["month"] < 9, "summer", "fall")))

# 시간 변수화
df["time"] = np.where(df["hour"] < 6, "dawn",
                   np.where(df["hour"] < 12, "morning",
                               np.where(df["hour"] < 18, "afternoon", "night")))

df.head()

# 데이터 처리
# "Not in a city" row 제거하기
df = df.loc[(df['location'] == "inland") | (df['location'] == "coast")]

# 필요없는 col 제거하기
df.drop(["State.Code",
            "NO2.1st.Max.Value", "NO2.1st.Max.Hour", "NO2.AQI",
            "O3.1st.Max.Value", "O3.1st.Max.Hour", "O3.AQI",
            "SO2.1st.Max.Value", "SO2.1st.Max.Hour", "SO2.AQI",
            "CO.1st.Max.Value", "CO.1st.Max.Hour", "CO.AQI"], axis=1, inplace=True)
# 이름 바꾸기
df.rename(columns = {"NO2.Mean":"NO2", "O3.Mean":"O3", "SO2.Mean":"SO2", "CO.Mean":"CO"}, inplace=True);


## 각 조건별 Histgoram
# 지역
n_bins = len(df[df["ET"]==1]["state"].unique())
plt.hist(df[df["ET"]==1]["state"], n_bins); plt.xticks(rotation=80)

plt.hist(df[df["ET"]==1]["location"], 2)

# 시간 1: 계절
plt.subplot(121); plt.hist(df[df["ET"]==1]["month"], 12); plt.title("month")
plt.subplot(122); plt.hist(df[df["ET"]==1]["season"], 4); plt.title("season")
plt.show()

# 시간 2: 시간
plt.subplot(121); plt.hist(df[df["ET"]==1]["hour"], 24); plt.title("Hour")
plt.subplot(122); plt.hist(df[df["ET"]==1]["time"], 4); plt.title("Time")
plt.show()


# 대기조건 별
plt.subplot(221); plt.hist(df[df["ET"]==1]["NO2"]); plt.title("NO2")
plt.subplot(222); plt.hist(df[df["ET"]==1]["O3"]); plt.title("O3")
plt.subplot(223); plt.hist(df[df["ET"]==1]["SO2"]); plt.title("SO2")
plt.subplot(224); plt.hist(df[df["ET"]==1]["CO"]); plt.title("CO")
plt.show()


# drop variables
df.drop(["city", "state", "day", "month", "year", "hour"], axis=1, inplace=True)
# one hot encoding
df = pd.get_dummies(data=df, columns=["season"], prefix="season")
df = pd.get_dummies(data=df, columns=["time"], prefix="time")
df = pd.get_dummies(data=df, columns=["location"], prefix="location")

# data reformat
df = df[["location_inland", "location_coast",
        "season_spring", "season_summer", "season_fall", "season_winter",
        "time_dawn", "time_morning", "time_afternoon", "time_night",
        "NO2", "O3", "SO2", "CO",
        "ET"]]

df.head()

# save
df.to_csv("./project/feature_data.csv")


## classification

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
# Separate input features (X) and target variable (y)
y = df.ET
X = df.drop(["ET"], axis=1)

# Train, Test 분리 시키기: hold out 0.3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train model
clf_3 = SVC(kernel='linear',
            class_weight='balanced', # penalize
            probability=True)

clf_3.fit(X_train, y_train)

# Predict on training set
y_hat = clf_3.predict(X_test)

# Is our model still predicting just one class?
print( np.unique( y_hat ) )

# How's our accuracy?
print( accuracy_score(y_test, y_hat) )

# What about AUROC?
y_hat_prob = clf_3.predict_proba(X)
y_hat_prob = [p[1] for p in y_hat_prob]
print( roc_auc_score(y, y_hat_prob) )

y_test == 1
