# IMPORT MODULES
import pandas as pd
import numpy as np
df = pd.read_csv("./K-MOOC_machine_learning/ch4/test.csv")
df.head()

## LOAD DATASET - simple variable
X = df["x"].values.reshape(-1,1)
y = df["y"].values

print(X, y)

# 하다 말았따
