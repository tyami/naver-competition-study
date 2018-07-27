import pandas as pd
seoul = pd.read_csv('SeoulHourlyAvgAirPollution.csv')
print(seoul)
sLength = len(seoul['day'])
print(sLength)
seoul['time'] = 'none'
seoul.loc[1,'time'] = 'what'
print(seoul)
tmp = 600
for i in range(0,sLength):
    if seoul.loc[i,'hour'] >= tmp * 0 and seoul.loc[i,'hour'] < tmp * 1 :
        seoul.loc[i,'time'] = 'dawn'
    if seoul.loc[i,'hour'] >= tmp * 1 and seoul.loc[i,'hour'] < tmp * 2:
        seoul.loc[i,'time'] = 'morning'
    if seoul.loc[i,'hour'] >= tmp * 2 and seoul.loc[i,'hour'] < tmp * 3:
        seoul.loc[i,'time'] = 'afternoon'
    if seoul.loc[i,'hour'] >= tmp * 3 and seoul.loc[i,'hour'] < tmp * 4:
        seoul.loc[i,'time'] = 'night'
print(seoul)
seoul.to_csv('SeoulHourlyAvgAirPollution_ver2')

import sklearn
