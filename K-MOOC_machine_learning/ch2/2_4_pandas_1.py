## Pandas: 구조화된 데이터 처리를 위한 package ! Python의 Excel
# numpy의 wraper (numpy의 기능들을 다 사용할 수 있다)

import pandas as pd

# Boston housing 문제
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
df_data = pd.read_csv(data_url, sep='\s+', header = None)

df_data.head() # 처음 다섯 줄 출력

df_data.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM','AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

df_data.head()

df_data.values

type(df_data.values) # pandas 데이터 타입: numpy

## Series: data column
from pandas import Series, DataFrame
list_data = [1,2,3,4,5]
example_obj = Series(data = list_data)
example_obj # index, value
list_name = ['a','b','c','d','e']
example_obj = Series(data = list_data, index = list_name)
example_obj

dict_data = {'a':1,'b':2,'c':3,'d':4,'e':5}
example_obj = Series(data = dict_data)
example_obj

## DataFrame: data 전체: matrix
raw_data = {'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'],
        'last_name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze'],
        'age': [42, 52, 36, 24, 73],
        'city': ['San Francisco', 'Baltimore', 'Miami', 'Douglas', 'Boston']}
df = pd.DataFrame(raw_data, columns = ['first_name', 'last_name', 'age', 'city'])
DataFrame(raw_data, columns = ["age", "city"])

df = DataFrame(raw_data, columns = ["first_data", "last_name", "age", "city", "debt"])

# 두 가지 방식의 Series 데이터 추출
df.first_name
df["first_name"]

# 데이터 접근
df.loc[1] # loc: 특정 row 접근. index 의 이름을 기준
df.loc[1:2]
df.iloc[1:] # index 순서를 기준

import numpy as np
s = pd.Series(np.nan, index = [49,48,47,46,45,1,2,3,4,5])
s.loc[:3]
s.iloc[:3]

df

# 데이터 새로 할당
df.debt = df.age > 40
df

df.T # transpose

df.values
df.to_csv()

del df["debt"]
df


## Selection  & Drop
df.head(3)
df[["first_data", "age"]].head(3)

df[:3] # column 없이 쓰는 경우는 row 기준
df["age"][:3]

ag = df["age"]
ag[ag <40]
ag[[1,3]]

df.index = df["last_name"]
df
del df["last_name"]
df
