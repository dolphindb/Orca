import argparse
import json
import sys
import csv
import time
import os.path as path
from contexttimer import Timer
from setup.settings import *
from pandas_driver import PandasDriver
from orca_driver import OrcaDriver
from orca_partition_driver import OrcaPartitionDriver

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pandas operations')
    parser.add_argument('n', help='rows of records in a file')
    parser.add_argument('program', help='one of pandas, orca or orcapartition')
    args = parser.parse_args()

    csvfile = open(path.abspath(path.join(__file__, "../../../reports/benchmark_report_" +
                                          time.strftime('%Y-%m-%d', time.localtime(time.time())))) + ".csv", 'a')
    writer = csv.writer(csvfile)
    writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))])
    writer.writerow(["'program':" + args.program, "'n':" + args.n])
    results = {'program': args.program, 'n': args.n}


    def reportToCsv(operation, timecost):
        lines = [operation, timecost]
        writer.writerow(lines)


    dataDir = path.abspath(path.join(__file__, "../setup/data"))
    employee_file = dataDir + "/ticker_" + args.n + ".csv"
    bonus_file = dataDir + "/value_" + args.n + ".csv"
    # functions = [s for s in dir(OrcaDriver) if not s.startswith("__") and s!='join']
    functions = ('a_load', 'filter', 'groupby', 'select', 'sort', 'resample_M', 'resample_3M',
    'resample_A', 'resample_3A', 'resample_Q', 'resample_3Q')
    # functions = ['a_load', 'resample_D']

    if args.program == "pandas":
        driver = PandasDriver(employee_file, bonus_file)
    elif args.program == "orca":
        driver = OrcaDriver(employee_file, bonus_file)
    elif args.program == "orcapartition":
        driver = OrcaPartitionDriver(employee_file, bonus_file)
        functions = ('a_load', 'filter', 'groupby', 'select', 'sort', 'resample_M', 'resample_3M',
                     'resample_A', 'resample_3A', 'resample_Q', 'resample_3Q')
        # functions = [s for s in dir(OrcaDriver) if not s.startswith("__") and s != 'join']

    else:
        raise ValueError("bad value for program")

    for task in functions:
        with Timer() as timer:
            getattr(driver, task)()
        results[task] = timer.elapsed
        reportToCsv(task, results[task])
    csvfile.close()

    # json.dump(results, sys.stdout)
    # print  # newline to file
