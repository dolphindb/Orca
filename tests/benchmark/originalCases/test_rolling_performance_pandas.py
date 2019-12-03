import unittest
import time
import os
import orca
import pandas as pd
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
pdf = pd.read_csv(data)
pr("Pandas spent " + str(time.time() - startTime) + "s importing '" + fileName + "'")

# generate report.csv
csvfile = open(WORK_DIR + 'report.csv', 'a')
writer = csv.writer(csvfile)
writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))])
writer.writerow(['rolling', 'pandas'])


def reportToCsv(operation, timecost):
    lines = [operation, timecost]
    writer.writerow(lines)


class RollingTest(unittest.TestCase):

    def test_rolling_param_window_sum(self):
        t1 = time.time()
        pdf.rolling(window=5).sum()
        reportToCsv("rolling sum ", str(time.time() - t1))

    def test_rolling_param_window_count(self):
        t1 = time.time()
        pdf.rolling(window=5).count()
        reportToCsv("rolling count ", str(time.time() - t1))

    def test_rolling_param_window_mean(self):
        t1 = time.time()
        pdf.rolling(window=5).mean()
        reportToCsv("rolling mean ", str(time.time() - t1))

    def test_rolling_param_window_max(self):
        t1 = time.time()
        pdf.rolling(window=5).max()
        reportToCsv("rolling max ", str(time.time() - t1))

    def test_rolling_param_window_min(self):
        t1 = time.time()
        pdf.rolling(window=5).min()
        reportToCsv("rolling min ", str(time.time() - t1))

    def test_rolling_param_window_std(self):
        t1 = time.time()
        pdf.rolling(window=5).std()
        reportToCsv("rolling std ", str(time.time() - t1))

    def test_rolling_param_window_var(self):
        t1 = time.time()
        pdf.rolling(window=5).var()
        reportToCsv("rolling var", str(time.time() - t1))
        csvfile.close()


if __name__ == '__main__':
    unittest.main()
