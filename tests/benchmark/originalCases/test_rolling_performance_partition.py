import unittest
import time
import os
import orca
import dolphindb as ddb
import os.path as path
from setup.settings import *
import csv

PRECISION = 6
# configure data directory
DATA_DIR = path.abspath(path.join(__file__, "../setup/data"))
fileName = 'onlyNumerical_performance.csv'
data = os.path.join(DATA_DIR, fileName)
data = data.replace('\\', '/')
dfsDatabase = "dfs://onlyNumerical_performanceDB"

# Orca connect to a DolphinDB server
orca.connect(HOST, PORT, "admin", "123456")

# Python API connect to DolphinDB server with login info
s = ddb.session()
success = s.connect(HOST, PORT, "admin", "123456")

if success:
    dolphindb_script = """
    login("admin", "123456")
    dbPath="dfs://onlyNumerical_performanceDB"
    if(existsDatabase(dbPath))
       dropDatabase(dbPath)
    schema = extractTextSchema(\"""" + data + """\")
    cols = exec name from schema
    types = ["INT", "BOOL", "CHAR", "SHORT", "INT", "LONG", "FLOAT", "DOUBLE"]
    schema = table(10000000:0, cols, types)
    tt=schema(schema).colDefs
    tt.drop!(`typeInt)
    tt.rename!(`name`type)
    range_schema=0..10*2000000+1
    db = database(dbPath, RANGE, range_schema )
    tb = db.createPartitionedTable(schema, `tb, `id)
    db.loadTextEx(`tb,`id, \"""" + data + """\" ,, tt)"""
    s.run(dolphindb_script)

# import
odfs = orca.read_table(dfsDatabase, 'tb')

# generate report.csv
csvfile = open(WORK_DIR + 'report.csv', 'a')
writer = csv.writer(csvfile)
writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))])
writer.writerow(['rolling', 'orca partition'])


def reportToCsv(operation, time):
    data = [operation, time]
    writer.writerow(data)


class RollingTest(unittest.TestCase):
    def test_rolling_param_window_sum(self):
        t1 = time.time()
        odfs.rolling(window=5).sum().to_pandas()
        reportToCsv("rolling sum int", str(time.time() - t1))

    def test_rolling_param_window_count(self):
        t1 = time.time()
        odfs.rolling(window=5).count().to_pandas()
        reportToCsv("rolling count int", str(time.time() - t1))

    def test_rolling_param_window_mean(self):
        t1 = time.time()
        odfs.rolling(window=5).mean().to_pandas()
        reportToCsv("rolling mean int", str(time.time() - t1))

    def test_rolling_param_window_max(self):
        t1 = time.time()
        odfs.rolling(window=5).max().to_pandas()
        reportToCsv("rolling max int", str(time.time() - t1))

    def test_rolling_param_window_min(self):
        t1 = time.time()
        odfs.rolling(window=5).min().to_pandas()
        reportToCsv("rolling min int", str(time.time() - t1))

    def test_rolling_param_window_std(self):
        t1 = time.time()
        odfs.rolling(window=5).std().to_pandas()
        reportToCsv("rolling std int", str(time.time() - t1))

    def test_rolling_param_window_var(self):
        t1 = time.time()
        print(odfs.rolling(window=5).var().to_pandas())
        reportToCsv("rolling var int", str(time.time() - t1))
        csvfile.close()


if __name__ == '__main__':
    unittest.main()
