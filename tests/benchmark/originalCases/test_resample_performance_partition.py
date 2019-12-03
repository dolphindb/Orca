import unittest
import time
import os
import orca
import dolphindb as ddb
import os.path as path
from setup.settings import *
import csv

# configure data directory
DATA_DIR = path.abspath(path.join(__file__, "../setup/data"))
fileName = 'allTypes_performance.csv'
data = os.path.join(DATA_DIR, fileName)
data = data.replace('\\', '/')
dfsDatabase = "dfs://allTypes_performanceDB"

# Orca connect to a DolphinDB server
orca.connect(HOST, PORT, "admin", "123456")

# Python API connect to DolphinDB server with login info
s = ddb.session()
success = s.connect(HOST, PORT, "admin", "123456")

if success:
    dolphindb_script = """
    login("admin", "123456")
    dbPath="dfs://allTypes_performanceDB"
    if(existsDatabase(dbPath))
       dropDatabase(dbPath)
    cols = exec name from extractTextSchema(\"""" + data + """\")
    types = exec type from extractTextSchema(\"""" + data + """\")
    schema = table(50000:0, cols, types)
    tt=schema(schema).colDefs
    tt.drop!(`typeInt)
    tt.rename!(`name`type)
    db = database(dbPath, RANGE, 1 5000001 10000001 15000001 20000001 25000001 30000001)
    tb1 = db.createPartitionedTable(schema, `tb1, `id)
    db.loadTextEx(`tb1,`id, \"""" + data + """\" ,, tt)"""
    s.run(dolphindb_script)

# import
startTime = time.time()
odfs = orca.read_table(dfsDatabase, 'tb1')
print("Orca spent " + str(time.time() - startTime) + "s importing '" + fileName + "'")

# generate report.csv
csvfile = open(WORK_DIR + 'report.csv', 'a')
writer = csv.writer(csvfile)
writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))])
writer.writerow(['resample', 'orca partition'])


def reportToCsv(operation, timecost):
    lines = [operation, timecost]
    writer.writerow(lines)


class ResampleTest(unittest.TestCase):
    def test_resample_param_rule_day_param_on_date_count(self):
        t1 = time.time()
        odfs.resample("d", on="date").count()
        reportToCsv("resample day count ", str(time.time() - t1))

        t1 = time.time()
        odfs.resample("3d", on="date").count()
        reportToCsv("resample 3day count ", str(time.time() - t1))

    def test_resample_param_rule_day_param_on_date_mean(self):
        t1 = time.time()
        odfs.resample("d", on="date").mean()
        reportToCsv("resample day mean ", str(time.time() - t1))

        t1 = time.time()
        odfs.resample("3d", on="date").mean()
        reportToCsv("resample 3day mean ", str(time.time() - t1))

    def test_resample_param_rule_day_param_on_date_max(self):
        t1 = time.time()
        odfs.resample("d", on="date").max()
        reportToCsv("resample day max ", str(time.time() - t1))

        t1 = time.time()
        odfs.resample("3d", on="date").max()
        reportToCsv("resample 3day max ", str(time.time() - t1))

    def test_resample_param_rule_day_param_on_date_min(self):
        t1 = time.time()
        odfs.resample("d", on="date").min()
        reportToCsv("resample day min ", str(time.time() - t1))

        t1 = time.time()
        odfs.resample("3d", on="date").min()
        reportToCsv("resample 3day min ", str(time.time() - t1))

    def test_resample_param_rule_day_param_on_date_std(self):
        t1 = time.time()
        odfs.resample("d", on="date").max()
        reportToCsv("resample day std ", str(time.time() - t1))

        t1 = time.time()
        odfs.resample("3d", on="date").max()
        reportToCsv("resample 3day std ", str(time.time() - t1))

    def test_resample_param_rule_day_param_on_date_sum(self):
        t1 = time.time()
        odfs.resample("d", on="date").sum()
        reportToCsv("resample day sum ", str(time.time() - t1))

        t1 = time.time()
        odfs.resample("3d", on="date").sum()
        reportToCsv("resample 3day sum ", str(time.time() - t1))

    def test_resample_param_rule_day_param_on_date_var(self):
        t1 = time.time()
        odfs.resample("d", on="date").max()
        reportToCsv("resample day var ", str(time.time() - t1))

        t1 = time.time()
        odfs.resample("3d", on="date").max()
        reportToCsv("resample 3day var ", str(time.time() - t1))

    def test_resample_param_rule_hour_param_on_timestamp_count(self):
        t1 = time.time()
        odfs.resample("H", on="timestamp").count()
        reportToCsv("resample hour count ", str(time.time() - t1))

        t1 = time.time()
        odfs.resample("3H", on="timestamp").count()
        reportToCsv("resample 3hour count ", str(time.time() - t1))

    def test_resample_param_rule_hour_param_on_timestamp_mean(self):
        t1 = time.time()
        odfs.resample("H", on="timestamp").mean()
        reportToCsv("resample hour mean ", str(time.time() - t1))

        t1 = time.time()
        odfs.resample("3H", on="timestamp").mean()
        reportToCsv("resample 3hour mean ", str(time.time() - t1))

    def test_resample_param_rule_hour_param_on_timestamp_max(self):
        t1 = time.time()
        odfs.resample("H", on="timestamp").max()
        reportToCsv("resample hour max ", str(time.time() - t1))

        t1 = time.time()
        odfs.resample("3H", on="timestamp").max()
        reportToCsv("resample 3hour max ", str(time.time() - t1))

    def test_resample_param_rule_hour_param_on_timestamp_min(self):
        t1 = time.time()
        odfs.resample("H", on="timestamp").min()
        reportToCsv("resample hour min ", str(time.time() - t1))

        t1 = time.time()
        odfs.resample("3H", on="timestamp").min()
        reportToCsv("resample 3hour min ", str(time.time() - t1))

    def test_resample_param_rule_hour_param_on_timestamp_std(self):
        t1 = time.time()
        odfs.resample("H", on="timestamp").max()
        reportToCsv("resample hour std ", str(time.time() - t1))

        t1 = time.time()
        odfs.resample("3H", on="timestamp").max()
        reportToCsv("resample 3hour std ", str(time.time() - t1))

    def test_resample_param_rule_hour_param_on_timestamp_sum(self):
        t1 = time.time()
        odfs.resample("H", on="timestamp").sum()
        reportToCsv("resample hour sum ", str(time.time() - t1))

        t1 = time.time()
        odfs.resample("3H", on="timestamp").sum()
        reportToCsv("resample 3hour sum ", str(time.time() - t1))

    def test_resample_param_rule_hour_param_on_timestamp_var(self):
        t1 = time.time()
        odfs.resample("H", on="timestamp").max()
        reportToCsv("resample hour var ", str(time.time() - t1))

        t1 = time.time()
        odfs.resample("3H", on="timestamp").max()
        reportToCsv("resample 3hour var ", str(time.time() - t1))

    def test_resample_param_rule_minute_param_on_timestamp_count(self):
        t1 = time.time()
        odfs.resample("T", on="timestamp").count()
        reportToCsv("resample minute count ", str(time.time() - t1))
        t1 = time.time()
        odfs.resample("3T", on="timestamp").count()
        reportToCsv("resample 3minute count ", str(time.time() - t1))

    def test_resample_param_rule_minute_param_on_timestamp_mean(self):
        t1 = time.time()
        odfs.resample("T", on="timestamp").mean()
        reportToCsv("resample minute mean ", str(time.time() - t1))

        t1 = time.time()
        odfs.resample("3T", on="timestamp").mean()
        reportToCsv("resample 3minute mean ", str(time.time() - t1))

    def test_resample_param_rule_minute_param_on_timestamp_max(self):
        t1 = time.time()
        odfs.resample("T", on="timestamp").max()
        reportToCsv("resample minute max ", str(time.time() - t1))

        t1 = time.time()
        odfs.resample("3T", on="timestamp").max()
        reportToCsv("resample 3minute max ", str(time.time() - t1))

    def test_resample_param_rule_minute_param_on_timestamp_min(self):
        t1 = time.time()
        odfs.resample("T", on="timestamp").min()
        reportToCsv("resample minute min ", str(time.time() - t1))

        t1 = time.time()
        odfs.resample("3T", on="timestamp").min()
        reportToCsv("resample 3minute min ", str(time.time() - t1))

    def test_resample_param_rule_minute_param_on_timestamp_std(self):
        t1 = time.time()
        odfs.resample("T", on="timestamp").max()
        reportToCsv("resample minute std ", str(time.time() - t1))

        t1 = time.time()
        odfs.resample("3T", on="timestamp").max()
        reportToCsv("resample 3minute std ", str(time.time() - t1))

    def test_resample_param_rule_minute_param_on_timestamp_sum(self):
        t1 = time.time()
        odfs.resample("T", on="timestamp").sum()
        reportToCsv("resample minute sum ", str(time.time() - t1))

        t1 = time.time()
        odfs.resample("3T", on="timestamp").sum()
        reportToCsv("resample 3minute sum ", str(time.time() - t1))

    def test_resample_param_rule_minute_param_on_timestamp_var(self):
        t1 = time.time()
        odfs.resample("T", on="timestamp").max()
        reportToCsv("resample minute var ", str(time.time() - t1))

        t1 = time.time()
        odfs.resample("3T", on="timestamp").max()
        reportToCsv("resample 3minute var ", str(time.time() - t1))

    def test_resample_param_rule_month_param_on_date_count(self):
        t1 = time.time()
        odfs.resample("m", on="date").count()
        reportToCsv("resample month count ", str(time.time() - t1))

        t1 = time.time()
        odfs.resample("3m", on="date").count()
        reportToCsv("resample 3month count ", str(time.time() - t1))

    def test_resample_param_rule_month_param_on_date_mean(self):
        t1 = time.time()
        odfs.resample("m", on="date").mean()
        reportToCsv("resample month mean ", str(time.time() - t1))

        t1 = time.time()
        odfs.resample("3m", on="date").mean()
        reportToCsv("resample 3month mean ", str(time.time() - t1))

    def test_resample_param_rule_month_param_on_date_max(self):
        t1 = time.time()
        odfs.resample("m", on="date").max()
        reportToCsv("resample month max ", str(time.time() - t1))

        t1 = time.time()
        odfs.resample("3m", on="date").max()
        reportToCsv("resample 3month max ", str(time.time() - t1))

    def test_resample_param_rule_month_param_on_date_min(self):
        t1 = time.time()
        odfs.resample("m", on="date").min()
        reportToCsv("resample month min ", str(time.time() - t1))

        t1 = time.time()
        odfs.resample("3m", on="date").min()
        reportToCsv("resample 3month min ", str(time.time() - t1))

    def test_resample_param_rule_month_param_on_date_std(self):
        t1 = time.time()
        odfs.resample("m", on="date").max()
        reportToCsv("resample month std ", str(time.time() - t1))

        t1 = time.time()
        odfs.resample("3m", on="date").max()
        reportToCsv("resample 3month std ", str(time.time() - t1))

    def test_resample_param_rule_month_param_on_date_sum(self):
        t1 = time.time()
        odfs.resample("m", on="date").sum()
        reportToCsv("resample month sum ", str(time.time() - t1))

        t1 = time.time()
        odfs.resample("3m", on="date").sum()
        reportToCsv("resample 3month sum ", str(time.time() - t1))

    def test_resample_param_rule_month_param_on_date_var(self):
        t1 = time.time()
        odfs.resample("m", on="date").max()
        reportToCsv("resample month var ", str(time.time() - t1))

        t1 = time.time()
        odfs.resample("3m", on="date").max()
        reportToCsv("resample 3month var ", str(time.time() - t1))

    def test_resample_param_rule_second_param_on_timestamp_count(self):
        t1 = time.time()
        odfs.resample("S", on="timestamp").count()
        reportToCsv("resample second count ", str(time.time() - t1))

        t1 = time.time()
        odfs.resample("3S", on="timestamp").count()
        reportToCsv("resample 3second count ", str(time.time() - t1))

    def test_resample_param_rule_second_param_on_timestamp_mean(self):
        t1 = time.time()
        odfs.resample("S", on="timestamp").mean()
        reportToCsv("resample second mean ", str(time.time() - t1))

        t1 = time.time()
        odfs.resample("3S", on="timestamp").mean()
        reportToCsv("resample 3second mean ", str(time.time() - t1))

    def test_resample_param_rule_second_param_on_timestamp_max(self):
        t1 = time.time()
        odfs.resample("S", on="timestamp").max()
        reportToCsv("resample second max ", str(time.time() - t1))

        t1 = time.time()
        odfs.resample("3S", on="timestamp").max()
        reportToCsv("resample 3second max ", str(time.time() - t1))

    def test_resample_param_rule_second_param_on_timestamp_min(self):
        t1 = time.time()
        odfs.resample("S", on="timestamp").min()
        reportToCsv("resample second min ", str(time.time() - t1))

        t1 = time.time()
        odfs.resample("3S", on="timestamp").min()
        reportToCsv("resample 3second min ", str(time.time() - t1))

    def test_resample_param_rule_second_param_on_timestamp_std(self):
        t1 = time.time()
        odfs.resample("S", on="timestamp").max()
        reportToCsv("resample second std ", str(time.time() - t1))

        t1 = time.time()
        odfs.resample("3S", on="timestamp").max()
        reportToCsv("resample 3second std ", str(time.time() - t1))

    def test_resample_param_rule_second_param_on_timestamp_sum(self):
        t1 = time.time()
        odfs.resample("S", on="timestamp").sum()
        reportToCsv("resample second sum ", str(time.time() - t1))

        t1 = time.time()
        odfs.resample("3S", on="timestamp").sum()
        reportToCsv("resample 3second sum ", str(time.time() - t1))

    def test_resample_param_rule_second_param_on_timestamp_var(self):
        t1 = time.time()
        odfs.resample("S", on="timestamp").max()
        reportToCsv("resample second var ", str(time.time() - t1))

        t1 = time.time()
        odfs.resample("3S", on="timestamp").max()
        reportToCsv("resample 3second var ", str(time.time() - t1))

    def test_resample_param_rule_week_param_on_date_count(self):
        t1 = time.time()
        odfs.resample("W", on="date").count()
        reportToCsv("resample week count ", str(time.time() - t1))

        t1 = time.time()
        odfs.resample("3W", on="date").count()
        reportToCsv("resample 3week count ", str(time.time() - t1))

    def test_resample_param_rule_week_param_on_date_mean(self):
        t1 = time.time()
        odfs.resample("W", on="date").mean()
        reportToCsv("resample week mean ", str(time.time() - t1))

        t1 = time.time()
        odfs.resample("3W", on="date").mean()
        reportToCsv("resample 3week mean ", str(time.time() - t1))

    def test_resample_param_rule_week_param_on_date_max(self):
        t1 = time.time()
        odfs.resample("W", on="date").max()
        reportToCsv("resample week max ", str(time.time() - t1))

        t1 = time.time()
        odfs.resample("3W", on="date").max()
        reportToCsv("resample 3week max ", str(time.time() - t1))

    def test_resample_param_rule_week_param_on_date_min(self):
        t1 = time.time()
        odfs.resample("W", on="date").min()
        reportToCsv("resample week min ", str(time.time() - t1))

        t1 = time.time()
        odfs.resample("3W", on="date").min()
        reportToCsv("resample 3week min ", str(time.time() - t1))

    def test_resample_param_rule_week_param_on_date_std(self):
        t1 = time.time()
        odfs.resample("W", on="date").max()
        reportToCsv("resample week std ", str(time.time() - t1))

        t1 = time.time()
        odfs.resample("3W", on="date").max()
        reportToCsv("resample 3week std ", str(time.time() - t1))

    def test_resample_param_rule_week_param_on_date_sum(self):
        t1 = time.time()
        odfs.resample("W", on="date").sum()
        reportToCsv("resample week sum ", str(time.time() - t1))

        t1 = time.time()
        odfs.resample("3W", on="date").sum()
        reportToCsv("resample 3week sum ", str(time.time() - t1))

    def test_resample_param_rule_week_param_on_date_var(self):
        t1 = time.time()
        odfs.resample("W", on="date").max()
        reportToCsv("resample week var ", str(time.time() - t1))

        t1 = time.time()
        odfs.resample("3W", on="date").max()
        reportToCsv("resample 3week var ", str(time.time() - t1))

    def test_resample_param_rule_year_param_on_date_count(self):
        t1 = time.time()
        odfs.resample("y", on="date").count()
        reportToCsv("resample year count ", str(time.time() - t1))

        t1 = time.time()
        odfs.resample("3y", on="date").count()
        reportToCsv("resample 3year count ", str(time.time() - t1))

    def test_resample_param_rule_year_param_on_date_mean(self):
        t1 = time.time()
        odfs.resample("y", on="date").mean()
        reportToCsv("resample year mean ", str(time.time() - t1))

        t1 = time.time()
        odfs.resample("3y", on="date").mean()
        reportToCsv("resample 3year mean ", str(time.time() - t1))

    def test_resample_param_rule_year_param_on_date_max(self):
        t1 = time.time()
        odfs.resample("y", on="date").max()
        reportToCsv("resample year max ", str(time.time() - t1))

        t1 = time.time()
        odfs.resample("3y", on="date").max()
        reportToCsv("resample 3year max ", str(time.time() - t1))

    def test_resample_param_rule_year_param_on_date_min(self):
        t1 = time.time()
        odfs.resample("y", on="date").min()
        reportToCsv("resample year min ", str(time.time() - t1))

        t1 = time.time()
        odfs.resample("3y", on="date").min()
        reportToCsv("resample 3year min ", str(time.time() - t1))

    def test_resample_param_rule_year_param_on_date_std(self):
        t1 = time.time()
        odfs.resample("y", on="date").max()
        reportToCsv("resample year std ", str(time.time() - t1))

        t1 = time.time()
        odfs.resample("3y", on="date").max()
        reportToCsv("resample 3year std ", str(time.time() - t1))

    def test_resample_param_rule_year_param_on_date_sum(self):
        t1 = time.time()
        odfs.resample("y", on="date").sum()
        reportToCsv("resample year sum ", str(time.time() - t1))

        t1 = time.time()
        odfs.resample("3y", on="date").sum()
        reportToCsv("resample 3year sum ", str(time.time() - t1))

    def test_resample_param_rule_year_param_on_date_var(self):
        t1 = time.time()
        odfs.resample("y", on="date").max()
        reportToCsv("resample year var ", str(time.time() - t1))

        t1 = time.time()
        odfs.resample("3y", on="date").max()
        reportToCsv("resample 3year var ", str(time.time() - t1))

        csvfile.close()
if __name__ == '__main__':
    unittest.main()
