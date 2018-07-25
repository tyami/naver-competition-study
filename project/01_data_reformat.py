import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

data_path = './project/ufo-air-quality/UFOPOLLUTANTS.csv'
df = pd.read_csv(data_path)

df.head()
df.columns

df.drop(["State.Code", "city", "state",
                "NO2.1st.Max.Value", "NO2.1st.Max.Hour", "NO2.AQI",
                "O3.1st.Max.Value", "O3.1st.Max.Hour", "O3.AQI",
                "SO2.1st.Max.Value", "SO2.1st.Max.Hour", "SO2.AQI",
                "CO.1st.Max.Value", "CO.1st.Max.Hour", "CO.AQI"], axis=1, inplace=True)

df.rename(columns = {"NO2.Mean":"NO2", "O3.Mean":"O3", "SO2.Mean":"SO2", "CO.Mean":"CO"}, inplace=True);
df.head()
df.tail()

df["time.dawn"] = df["hour"] < 6
df["time.morning"] = (df["hour"] >= 6) & (df["hour"] < 12)
df["time.afternoon"] = (df["hour"] >= 12) & (df["hour"] < 18)
df["time.night"] = (df["hour"] >= 18) & (df["hour"] < 24)

df.head()


## 각 조건별
# ㅅ간
plt.hist(df[df["ET"]==1]["hour"])

plt.hist(df[df["ET"]==1]["month"])
