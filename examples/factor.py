import dolphindb.orca as orca
import pandas as pd
import numpy as np

orca.connect("localhost", 8848)

# Orca

df = orca.read_csv("D:/DolphinDB/Orca/database/data.csv", partitioned=False)

ask = df["av1"]
bid = df["bv1"]
p = df["mp"].iloc[0]
for i in range(2, 11):
    ask += df["av"+str(i)] * np.exp(-10*(i-1)/p)
    bid += df["bv"+str(i)] * np.exp(-10*(i-1)/p)

vol_diff = (0.5 * (bid / ask).log()).compute()

vot = df["wp"].ewm(span=10, adjust=False).std()
vot = 1/(1+(-100*(vot)).exp()).compute()
