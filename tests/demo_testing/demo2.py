from setup.settings import *
import orca as pd

pd.connect(HOST, PORT, "admin", "123456")

import os.path as path

# configure data directory
DATA_DIR = path.abspath(path.join(__file__, "../setup/data"))
fileName = 'testcsv.csv'
data = path.join(DATA_DIR, fileName)
data = data.replace('\\', '/')


def testVariables():
    odf = pd.read_csv(data)
    a = odf.rolling(4).sum()
    a.to_csv(data)
    b = a.groupby("distance").sum()
    c = b[b["recession_velocity"] > 1000].mean()
    print(a.to_pandas())
    print(b.to_pandas())
    print(c.to_pandas())


if __name__ == "__main__":
    s = pd.default_session()
    print(s.run("objs()"))
    print(len(s.run("objs()")))
    testVariables()
    print(s.run("objs()"))
    print(len(s.run("objs()")))
