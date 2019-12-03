import orca
from setup.settings import *
import matplotlib.pyplot as plt
import os.path as path

# configure data directory
DATA_DIR = path.abspath(path.join(__file__, "../setup/data"))
fileName = 'wages_hours.csv'
data = path.join(DATA_DIR, fileName)
data = data.replace('\\', '/')

orca.connect(HOST, PORT, "admin", "123456")

odf = orca.read_csv(data)
odf.head()

odf = orca.read_csv(data, sep="\t")
odf.head()

odf2 = odf[["AGE", "RATE"]]
odf2.head()

data_sorted = odf2.sort_values(["AGE"])
data_sorted.head()

data_sorted.set_index("AGE", inplace=True)
data_sorted.head()

data_sorted.plot()
plt.show()

odf2.set_index("AGE", inplace=True)
odf2.head()

odf2.plot()
plt.show()
