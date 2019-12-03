import unittest
import time
import os
import orca
import os.path as path
from setup.settings import *
import csv

# configure data directory
DATA_DIR = path.abspath(path.join(__file__, "../setup/data"))
fileName = 'onlyNumerical_performance.csv'
data = os.path.join(DATA_DIR, fileName)
data = data.replace('\\', '/')

# Orca connect to a DolphinDB server
orca.connect(HOST, PORT, "admin", "123456")

# import
startTime = time.time()
odf = orca.read_csv(data)
print("Orca spent " + str(time.time() - startTime) + "s importing '" + fileName + "'")

# generate report.csv
csvfile = open(WORK_DIR + 'report.csv', 'a')
writer = csv.writer(csvfile)
writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))])
writer.writerow(['rolling', 'orca'])


def reportToCsv(operation, timecost):
    lines = [operation, timecost]
    writer.writerow(lines)


class RollingTest(unittest.TestCase):
    def test_rolling_param_window_sum(self):
        t1 = time.time()
        odf.rolling(window=5).sum().to_pandas()
        reportToCsv("rolling sum int", str(time.time() - t1))

    def test_rolling_param_window_count(self):
        t1 = time.time()
        odf.rolling(window=5).count().to_pandas()
        reportToCsv("rolling count int", str(time.time() - t1))

    def test_rolling_param_window_mean(self):
        t1 = time.time()
        odf.rolling(window=5).mean().to_pandas()
        reportToCsv("rolling mean int", str(time.time() - t1))

    def test_rolling_param_window_max(self):
        t1 = time.time()
        odf.rolling(window=5).max().to_pandas()
        reportToCsv("rolling max int", str(time.time() - t1))

    def test_rolling_param_window_min(self):
        t1 = time.time()
        odf.rolling(window=5).min().to_pandas()
        reportToCsv("rolling min int", str(time.time() - t1))

    def test_rolling_param_window_std(self):
        t1 = time.time()
        odf.rolling(window=5).std().to_pandas()
        reportToCsv("rolling std int", str(time.time() - t1))

    def test_rolling_param_window_var(self):
        t1 = time.time()
        print(odf.rolling(window=5).var().to_pandas())
        reportToCsv("rolling var int", str(time.time() - t1))
        csvfile.close()


if __name__ == '__main__':
    unittest.main()
