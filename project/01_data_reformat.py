import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

data_path = './project/ufo-air-quality/UFOPOLLUTANTS.csv'
df = pd.read_csv(data_path)

df.head()
df.columns

# 계절
df["weather"] = np.where((df["month"] < 3) | (df["month"] > 11), "winter",
                   np.where(df["month"] < 6, "spring",
                               np.where(df["month"] < 9, "summer", "fall")))

df_ = pd.get_dummies(df["weather"], prefix="weather")

df_.head()

# 시간
df["time"] = np.where(df["hour"] < 6, "dawn",
                   np.where(df["hour"] < 12, "morning",
                               np.where(df["hour"] < 18, "afternoon", "night")))

df["time_dawn"] = df["hour"] < 6
df["time_morning"] = (df["hour"] >= 6) & (df["hour"] < 12)
df["time_afternoon"] = (df["hour"] >= 12) & (df["hour"] < 18)
df["time_night"] = (df["hour"] >= 18) & (df["hour"] < 24)
df.head()

# 지역: "Not in a city" 제거하기


# 대기조건
df.drop(["State.Code", "city",
                "NO2.1st.Max.Value", "NO2.1st.Max.Hour", "NO2.AQI",
                "O3.1st.Max.Value", "O3.1st.Max.Hour", "O3.AQI",
                "SO2.1st.Max.Value", "SO2.1st.Max.Hour", "SO2.AQI",
                "CO.1st.Max.Value", "CO.1st.Max.Hour", "CO.AQI"], axis=1, inplace=True)

df.rename(columns = {"NO2.Mean":"NO2", "O3.Mean":"O3", "SO2.Mean":"SO2", "CO.Mean":"CO"}, inplace=True);
df.head()
df.tail()




## 각 조건별
# 지역
n_bins = len(df[df["ET"]==1]["state"].unique())
plt.hist(df[df["ET"]==1]["state"], n_bins); plt.xticks(rotation=80)
plt.hist()

plt.hist(df[df["ET"]==1]["location"], 2)

# 시간 1: 계절
plt.subplot(121); plt.hist(df[df["ET"]==1]["month"], 12); plt.title("month")
plt.subplot(122); plt.hist(df[df["ET"]==1]["weather"], 4); plt.title("weather")
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

plt.hist(df[df["ET"]==1]["time.dawn"])
plt.hist(df[df["ET"]==1]["time.morning"])
