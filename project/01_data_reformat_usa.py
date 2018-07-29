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

# 단위 맞추기 - ppm으로
df["NO2"] = df["NO2"] / 1000
df["SO2"] = df["SO2"] / 1000


## 각 조건별 Histgoram
# 장소 1: state
n_bins = len(df[df["ET"]==1]["state"].unique())
plt.figure(); plt.hist(df[df["ET"]==1]["state"], n_bins, color="darkorchid"); plt.xticks(rotation=80); plt.title("state")
plt.savefig('./fig_hist_state.png',bbox_inches='tight',format = 'png',dpi = 300)

# 장소 2: location
plt.figure(); plt.hist(df[df["ET"]==1]["location"], 2, rwidth=0.5, color="darkorchid"); plt.title("location"); plt.xticks([0.25, 0.75])
plt.savefig('./fig_hist_location.png',bbox_inches='tight',format = 'png',dpi = 300)

# 시간 1: month
plt.figure(); plt.hist(df[df["ET"]==1]["month"], 12, color="darkorchid"); plt.title("month"); plt.xticks(np.arange(1,13,1));
plt.savefig('./fig_hIst_month.png',bbox_inches='tight',format = 'png',dpi = 300)

# 시간 2: season
lbl_bar = ["spring","summer","fall","winter"]
dat_bar = [sum(df[df["ET"]==1]["season"]=="spring"), sum(df[df["ET"]==1]["season"]=="summer"),
           sum(df[df["ET"]==1]["season"]=="fall"), sum(df[df["ET"]==1]["season"]=="winter")]
plt.figure(); plt.bar(range(len(lbl_bar)), dat_bar, color="darkorchid");  plt.title("season"); plt.xticks(range(len(lbl_bar)), lbl_bar)
plt.savefig('./fig_hist_season.png',bbox_inches='tight',format = 'png',dpi = 300)

# 시간 3: hour
plt.figure(); plt.hist(df[df["ET"]==1]["hour"], np.arange(25), color="darkorchid"); plt.title("Hour"); plt.xticks(np.arange(25))
plt.savefig('./fig_hour.png',bbox_inches='tight',format = 'png',dpi = 300)

# 시간 4: time
lbl_bar = ["dawn","morning","afternoon","night"]
dat_bar = [sum(df[df["ET"]==1]["time"]=="dawn"), sum(df[df["ET"]==1]["time"]=="morning"),
           sum(df[df["ET"]==1]["time"]=="afternoon"), sum(df[df["ET"]==1]["time"]=="night")]
plt.figure(); plt.bar(range(len(lbl_bar)), dat_bar, color="darkorchid");  plt.title("Time"); plt.xticks(range(len(lbl_bar)), lbl_bar)
plt.savefig('./fig_hist_time.png',bbox_inches='tight',format = 'png',dpi = 300)


def plot_two_hist(data1, data2, n_bins, lbl_data1, lbl_data2, str_title):

    n, bins, patches = plt.hist(data1, bins=n_bins, alpha=0.7, normed=True, label=lbl_data1, color="darkorchid")
    n, bins, patches = plt.hist(data2, bins=bins, alpha=0.7, normed=True, label=lbl_data2, color="dimgray")

    plt.title(str_title)
    plt.legend()


# 대기조건 1-4: NO2, O3, SO2, CO
plt.figure(); plot_two_hist(df[df["ET"]==1]["NO2"], df[df["ET"]==0]["NO2"], 25, "UFO", "UFO X", "NO2")
plt.savefig('./fig_NO2.png',bbox_inches='tight',format = 'png',dpi = 300)

plt.figure(); plot_two_hist(df[df["ET"]==1]["O3"], df[df["ET"]==0]["O3"], 25, "UFO", "UFO X",  "O3")
plt.savefig('./fig_O3.png',bbox_inches='tight',format = 'png',dpi = 300)

plt.figure(); plot_two_hist(df[df["ET"]==1]["SO2"], df[df["ET"]==0]["SO2"], 25, "UFO", "UFO X",  "SO2")
plt.savefig('./fig_SO2.png',bbox_inches='tight',format = 'png',dpi = 300)

plt.figure(); plot_two_hist(df[df["ET"]==1]["CO"], df[df["ET"]==0]["CO"], 25, "UFO", "UFO X", "CO")
plt.savefig('./fig_CO.png',bbox_inches='tight',format = 'png',dpi = 300)


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
df.to_csv("./project/feature_data_usa.csv")
