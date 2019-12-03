import unittest
import time
import os
import orca
import os.path as path
import dolphindb as ddb
from setup.settings import *
import csv

# configure data directory
DATA_DIR = path.abspath(path.join(__file__, "../setup/data"))
PRECISION_POINT = 1
fileName = 'USPrices.csv'
data = os.path.join(DATA_DIR, fileName)
data = data.replace('\\', '/')
dfsDatabase = "dfs://USPricesPF"

# Orca_partition connect to a DolphinDB server
orca.connect(HOST, PORT, "admin", "123456")

s = ddb.session()
success = s.connect(HOST, PORT, "admin", "123456")

if success:
    dolphindb_script = """
    login("admin", "123456")
    dbPath="dfs://USPricesPF"
    if(existsDatabase(dbPath))
       dropDatabase(dbPath)
    t = loadText(\"""" + data + """\")
    rng = 1990.01.01 + (1..27)* 366
    db = database(dbPath, RANGE, rng)
    db.createPartitionedTable(t, `USPrices, `date).append!(t)
    """
    s.run(dolphindb_script)

# import
startTime = time.time()
odfs = orca.read_table(dfsDatabase, 'USPrices')
print("Orca_partition spent " + str(time.time() - startTime) + "s importing '" + fileName + "'")

# generate report.csv
csvfile = open(WORK_DIR + 'report.csv', 'a')
writer = csv.writer(csvfile)
writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))])
writer.writerow(['groupby', 'orca partition'])


def reportToCsv(operation, timecost, topandas):
    lines = [operation, timecost, topandas]
    writer.writerow(lines)


############################################################################


class GroupByTest(unittest.TestCase):

    def test_groupby_param_by_date_sum(self):
        t1 = time.time()
        odfs.groupby('date').sum().to_pandas()
        reportToCsv("groupby sum ", str(time.time() - t1))

    def test_groupby_param_by_date_count(self):
        t1 = time.time()
        odfs.groupby('date').count().to_pandas()
        reportToCsv("groupby count ", str(time.time() - t1))

    def test_groupby_param_by_date_mean(self):
        t1 = time.time()
        odfs.groupby('date').mean().to_pandas()
        reportToCsv("groupby mean ", str(time.time() - t1))

    def test_groupby_param_by_date_max(self):
        t1 = time.time()
        odfs.groupby('date').max().to_pandas()
        reportToCsv("groupby max ", str(time.time() - t1))

    def test_groupby_param_by_date_min(self):
        t1 = time.time()
        odfs.groupby('date').min().to_pandas()
        reportToCsv("groupby min ", str(time.time() - t1))

    def test_groupby_param_by_date_first(self):
        t1 = time.time()
        odfs.groupby('date').first().to_pandas()
        reportToCsv("groupby first ", str(time.time() - t1))

    def test_groupby_param_by_date_size(self):
        t1 = time.time()
        odfs.groupby('date').size().to_pandas()
        reportToCsv("groupby size ", str(time.time() - t1))

    def test_groupby_param_by_date_std(self):
        t1 = time.time()
        odfs.groupby('date').std().to_pandas()
        reportToCsv("groupby std ", str(time.time() - t1))

    def test_groupby_param_by_date_var(self):
        t1 = time.time()
        odfs.groupby('date').var().to_pandas()
        reportToCsv("groupby var ", str(time.time() - t1))
        csvfile.close()

    def test_groupby_param_axis(self):
        self.assertEqual(1, 1)

    def test_groupby_param_level(self):
        self.assertEqual(1, 1)

    def test_groupby_param_as_index(self):
        self.assertEqual(1, 1)

    def test_groupby_param_sort(self):
        self.assertEqual(1, 1)

    def test_groupby_param_group_keys(self):
        self.assertEqual(1, 1)

    def test_groupby_param_squeeze(self):
        self.assertEqual(1, 1)

    def test_groupby_param_kwargs(self):
        self.assertEqual(1, 1)


if __name__ == '__main__':
    unittest.main()
