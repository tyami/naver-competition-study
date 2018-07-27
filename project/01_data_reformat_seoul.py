import pandas as pd
import numpy as np

df = pd.read_csv("./project/seoulairreport/SeoulHourlyAvgAirPollution.csv")

df.columns

# 시간 변수화
# 계절
df["month"] = (df["측정일시"] // 1000000) % 100
df["season"] = np.where((df["month"] < 3) | (df["month"] > 11), "winter",
                   np.where(df["month"] < 6, "spring",
                               np.where(df["month"] < 9, "summer", "fall")))
# 시간
df["hour"] = (df["측정일시"] // 100) % 100
df["time"] = np.where(df["hour"] < 6, "dawn",
                   np.where(df["hour"] < 12, "morning",
                               np.where(df["hour"] < 18, "afternoon", "night")))

#지역 변수화
df["location"] = "inland"

# 이름 바꾸기
df.rename(columns = {"이산화질소농도(ppm)":"NO2", "오존농도(ppm)":"O3",
                     "아황산가스(ppm)":"SO2", "일산화탄소농도(ppm)":"CO"},
          inplace=True)

# drop variables
df.drop(["month", "hour",
         "측정일시", "측정소명", "미세먼지(㎍/㎥)", "초미세먼지(㎍/㎥)"],
        axis=1, inplace=True)

# one hot encoding
df = pd.get_dummies(data=df, columns=["season"], prefix="season")
df = pd.get_dummies(data=df, columns=["time"], prefix="time")
df = pd.get_dummies(data=df, columns=["location"], prefix="location")

# vacant variable
df["location_coast"] = 0
df["season_spring"] = 0
df["season_summer"] = 0
df["season_winter"] = 0
df["ET"] = np.nan

# data reformat
df = df[["location_inland", "location_coast",
        "season_spring", "season_summer", "season_fall", "season_winter",
        "time_dawn", "time_morning", "time_afternoon", "time_night",
        "NO2", "O3", "SO2", "CO",
        "ET"]]

df.head()

# save
df.to_csv("./project/feature_data_seoul.csv")
