import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

test_df = pd.read_csv("./K-MOOC_machine_learning/kaggle_bike_sharing/data/test.csv", parse_dates=["datetime"])
train_df = pd.read_csv("./K-MOOC_machine_learning/kaggle_bike_sharing/data/train.csv", parse_dates=["datetime"])

all_df = pd.concat((train_df, test_df), axis=0).reset_index()
all_df.head()
all_df.tail()

# index
test_index = list(range(len(train_df)))
train_index = list(range(len(train_df), len(all_df)))

all_df.isnull().sum()


def rmsle(y, y_):
    log1 = np.nan_to_num(np.log(y + 1)) # inf는 큰 값, nan은 0에 가까운 "숫자" 리턴
    log2 = np.nan_to_num(np.log(y_ + 1))
    calc = (log1 - log2) ** 2
    return np.sqrt(np.mean(calc))

submission_df = pd.read_csv("./K-MOOC_machine_learning/kaggle_bike_sharing/data/sampleSubmission.csv")
submission_df.head()    
rmsle(submission_df["count"].values, np.random.randint(0, 100, size=len(submission_df)))



pd.get_dummies(all_df["season"], prefix="season")
pre_df = all_df.merge(pd.get_dummies(all_df["season"], prefix="season"), left_index=True, right_index=True)
pre_df.head()
pre_df = pre_df.merge(pd.get_dummies(pre_df["weather"], prefix="weather"), left_index=True, right_index=True)

pre_df["year"] = pre_df["datetime"].dt.year
pre_df["month"] = pre_df["datetime"].dt.month
pre_df["day"] = pre_df["datetime"].dt.day
pre_df["hour"] = pre_df["datetime"].dt.hour
pre_df["weekday"] = pre_df["datetime"].dt.weekday

pre_df = pre_df.merge(pd.get_dummies(pre_df["weekday"], prefix="weekday"), left_index=True, right_index=True)
pre_df.head()

pre_df.dtypes

category_variable_list = ["season","weather","workingday","season_1","season_2","season_3","season_4","weather_1","weather_2","weather_3","weather_4","year","month","day","hour","weekday","weekday_0","weekday_1","weekday_2","weekday_3","weekday_4","weekday_5","weekday_6"]
for var_name in category_variable_list:
    pre_df[var_name] = pre_df[var_name].astype("category")
pre_df.dtypes

fig, axes = plt.subplots(nrows=3,ncols=3)
fig.set_size_inches(12, 5)
axes[0][0].bar(train_df["year"], train_df["count"])
axes[0][1].bar(train_df["weather"], train_df["count"])
axes[0][2].bar(train_df["workingday"], train_df["count"])
axes[1][0].bar(train_df["holiday"], train_df["count"])
axes[1][1].bar(train_df["weekday"], train_df["count"])
axes[1][2].bar(train_df["month"], train_df["count"])
axes[2][0].bar(train_df["day"], train_df["count"])
axes[2][1].bar(train_df["hour"], train_df["count"])
plt.show()

series_data = train_df.groupby(["month"])["count"].mean()
series_data.index.tolist()[:5]

fig, ax = plt.subplots()
ax.bar(range(len(serires_data)), serires_data)
fig.set_size_inches(12,5)
plt.show()
