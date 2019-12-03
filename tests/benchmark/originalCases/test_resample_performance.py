import unittest
import time
import os
import orca
import os.path as path
from setup.settings import *
import csv

PRECISION = 4
# configure data directory
DATA_DIR = path.abspath(path.join(__file__, "../setup/data"))
fileName = 'allTypes_performance.csv'
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
writer.writerow(['resample', 'orca'])


def reportToCsv(operation, timecost):
    lines = [operation, timecost]
    writer.writerow(lines)


class ResampleTest(unittest.TestCase):

    def test_resample_param_rule_year_param_on_date_sum(self):
        t1 = time.time()
        odf.resample("y", on="date").sum()
        reportToCsv("resample year sum ", str(time.time() - t1))

    def test_resample_param_rule_year_param_on_date_count(self):
        t1 = time.time()
        odf.resample("y", on="date").count()
        reportToCsv("resample year count ", str(time.time() - t1))

    def test_resample_param_rule_year_param_on_date_mean(self):
        t1 = time.time()
        odf.resample("y", on="date").mean()
        reportToCsv("resample year mean ", str(time.time() - t1))

    def test_resample_param_rule_year_param_on_date_max(self):
        t1 = time.time()
        odf.resample("y", on="date").max()
        reportToCsv("resample year max ", str(time.time() - t1))

    def test_resample_param_rule_year_param_on_date_min(self):
        t1 = time.time()
        odf.resample("y", on="date").min()
        reportToCsv("resample year min ", str(time.time() - t1))

    def test_resample_param_rule_year_param_on_date_std(self):
        t1 = time.time()
        odf.resample("y", on="date").max()
        reportToCsv("resample year std ", str(time.time() - t1))

    def test_resample_param_rule_year_param_on_date_var(self):
        t1 = time.time()
        odf.resample("y", on="date").max()
        reportToCsv("resample year var ", str(time.time() - t1))

    def test_resample_param_rule_month_param_on_date_sum(self):
        t1 = time.time()
        odf.resample("m", on="date").sum()
        reportToCsv("resample month sum ", str(time.time() - t1))

    def test_resample_param_rule_month_param_on_date_count(self):
        t1 = time.time()
        odf.resample("m", on="date").count()
        reportToCsv("resample month count ", str(time.time() - t1))

    def test_resample_param_rule_month_param_on_date_mean(self):
        t1 = time.time()
        odf.resample("m", on="date").mean()
        reportToCsv("resample month mean ", str(time.time() - t1))

    def test_resample_param_rule_month_param_on_date_max(self):
        t1 = time.time()
        odf.resample("m", on="date").max()
        reportToCsv("resample month max ", str(time.time() - t1))

    def test_resample_param_rule_month_param_on_date_min(self):
        t1 = time.time()
        odf.resample("m", on="date").min()
        reportToCsv("resample month min ", str(time.time() - t1))

    def test_resample_param_rule_month_param_on_date_std(self):
        t1 = time.time()
        odf.resample("m", on="date").max()
        reportToCsv("resample month std ", str(time.time() - t1))

    def test_resample_param_rule_month_param_on_date_var(self):
        t1 = time.time()
        odf.resample("m", on="date").max()
        reportToCsv("resample month var ", str(time.time() - t1))

    def test_resample_param_rule_day_param_on_date_sum(self):
        t1 = time.time()
        odf.resample("d", on="date").sum()
        reportToCsv("resample day sum ", str(time.time() - t1))

    def test_resample_param_rule_day_param_on_date_count(self):
        t1 = time.time()
        odf.resample("d", on="date").count()
        reportToCsv("resample day count ", str(time.time() - t1))

    def test_resample_param_rule_day_param_on_date_mean(self):
        t1 = time.time()
        odf.resample("d", on="date").mean()
        reportToCsv("resample day mean ", str(time.time() - t1))

    def test_resample_param_rule_day_param_on_date_max(self):
        t1 = time.time()
        odf.resample("d", on="date").max()
        reportToCsv("resample day max ", str(time.time() - t1))

    def test_resample_param_rule_day_param_on_date_min(self):
        t1 = time.time()
        odf.resample("d", on="date").min()
        reportToCsv("resample day min ", str(time.time() - t1))

    def test_resample_param_rule_day_param_on_date_std(self):
        t1 = time.time()
        odf.resample("d", on="date").max()
        reportToCsv("resample day std ", str(time.time() - t1))

    def test_resample_param_rule_day_param_on_date_var(self):
        t1 = time.time()
        odf.resample("d", on="date").max()
        reportToCsv("resample day var ", str(time.time() - t1))

    def test_resample_param_rule_week_param_on_date_sum(self):
        t1 = time.time()
        odf.resample("W", on="date").sum()
        reportToCsv("resample week sum ", str(time.time() - t1))

    def test_resample_param_rule_week_param_on_date_count(self):
        t1 = time.time()
        odf.resample("W", on="date").count()
        reportToCsv("resample week count ", str(time.time() - t1))

    def test_resample_param_rule_week_param_on_date_mean(self):
        t1 = time.time()
        odf.resample("W", on="date").mean()
        reportToCsv("resample week mean ", str(time.time() - t1))

    def test_resample_param_rule_week_param_on_date_max(self):
        t1 = time.time()
        odf.resample("W", on="date").max()
        reportToCsv("resample week max ", str(time.time() - t1))

    def test_resample_param_rule_week_param_on_date_min(self):
        t1 = time.time()
        odf.resample("W", on="date").min()
        reportToCsv("resample week min ", str(time.time() - t1))

    def test_resample_param_rule_week_param_on_date_std(self):
        t1 = time.time()
        odf.resample("W", on="date").max()
        reportToCsv("resample week std ", str(time.time() - t1))

    def test_resample_param_rule_week_param_on_date_var(self):
        t1 = time.time()
        odf.resample("W", on="date").max()
        reportToCsv("resample week var ", str(time.time() - t1))

    def test_resample_param_rule_hour_param_on_timestamp_sum(self):
        t1 = time.time()
        odf.resample("H", on="timestamp").sum()
        reportToCsv("resample hour sum ", str(time.time() - t1))

    def test_resample_param_rule_hour_param_on_timestamp_count(self):
        t1 = time.time()
        odf.resample("H", on="timestamp").count()
        reportToCsv("resample hour count ", str(time.time() - t1))

    def test_resample_param_rule_hour_param_on_timestamp_mean(self):
        t1 = time.time()
        odf.resample("H", on="timestamp").mean()
        reportToCsv("resample hour mean ", str(time.time() - t1))

    def test_resample_param_rule_hour_param_on_timestamp_max(self):
        t1 = time.time()
        odf.resample("H", on="timestamp").max()
        reportToCsv("resample hour max ", str(time.time() - t1))

    def test_resample_param_rule_hour_param_on_timestamp_min(self):
        t1 = time.time()
        odf.resample("H", on="timestamp").min()
        reportToCsv("resample hour min ", str(time.time() - t1))

    def test_resample_param_rule_hour_param_on_timestamp_std(self):
        t1 = time.time()
        odf.resample("H", on="timestamp").max()
        reportToCsv("resample hour std ", str(time.time() - t1))

    def test_resample_param_rule_hour_param_on_timestamp_var(self):
        t1 = time.time()
        odf.resample("H", on="timestamp").max()
        reportToCsv("resample hour var ", str(time.time() - t1))

    def test_resample_param_rule_minute_param_on_timestamp_sum(self):
        t1 = time.time()
        odf.resample("T", on="timestamp").sum()
        reportToCsv("resample minute sum ", str(time.time() - t1))

    def test_resample_param_rule_minute_param_on_timestamp_count(self):
        t1 = time.time()
        odf.resample("T", on="timestamp").count()
        reportToCsv("resample minute count ", str(time.time() - t1))

    def test_resample_param_rule_minute_param_on_timestamp_mean(self):
        t1 = time.time()
        odf.resample("T", on="timestamp").mean()
        reportToCsv("resample minute mean ", str(time.time() - t1))

    def test_resample_param_rule_minute_param_on_timestamp_max(self):
        t1 = time.time()
        odf.resample("T", on="timestamp").max()
        reportToCsv("resample minute max ", str(time.time() - t1))

    def test_resample_param_rule_minute_param_on_timestamp_min(self):
        t1 = time.time()
        odf.resample("T", on="timestamp").min()
        reportToCsv("resample minute min ", str(time.time() - t1))

    def test_resample_param_rule_minute_param_on_timestamp_std(self):
        t1 = time.time()
        odf.resample("T", on="timestamp").max()
        reportToCsv("resample minute std ", str(time.time() - t1))

    def test_resample_param_rule_minute_param_on_timestamp_var(self):
        t1 = time.time()
        odf.resample("T", on="timestamp").max()
        reportToCsv("resample minute var ", str(time.time() - t1))

    def test_resample_param_rule_second_param_on_timestamp_sum(self):
        t1 = time.time()
        odf.resample("S", on="timestamp").sum()
        reportToCsv("resample second sum ", str(time.time() - t1))

    def test_resample_param_rule_second_param_on_timestamp_count(self):
        t1 = time.time()
        odf.resample("S", on="timestamp").count()
        reportToCsv("resample second count ", str(time.time() - t1))

    def test_resample_param_rule_second_param_on_timestamp_mean(self):
        t1 = time.time()
        odf.resample("S", on="timestamp").mean()
        reportToCsv("resample second mean ", str(time.time() - t1))

    def test_resample_param_rule_second_param_on_timestamp_max(self):
        t1 = time.time()
        odf.resample("S", on="timestamp").max()
        reportToCsv("resample second max ", str(time.time() - t1))

    def test_resample_param_rule_second_param_on_timestamp_min(self):
        t1 = time.time()
        odf.resample("S", on="timestamp").min()
        reportToCsv("resample second min ", str(time.time() - t1))

    def test_resample_param_rule_second_param_on_timestamp_std(self):
        t1 = time.time()
        odf.resample("S", on="timestamp").max()
        reportToCsv("resample second std ", str(time.time() - t1))

    def test_resample_param_rule_second_param_on_timestamp_var(self):
        t1 = time.time()
        odf.resample("S", on="timestamp").max()
        reportToCsv("resample second var ", str(time.time() - t1))
        csvfile.close()


if __name__ == '__main__':
    unittest.main()
