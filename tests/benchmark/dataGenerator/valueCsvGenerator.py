import argparse
import random
import csv
import os.path as path
import pandas as pd
import numpy as np

# python valueCsvGenerator.py 15

parser = argparse.ArgumentParser(description='Generate bonus table')
parser.add_argument('n', help='rows of records in a file')
args = parser.parse_args()

dataDir = path.abspath(path.join(__file__, "../../testPerformance/setup/data"))
filea = dataDir + "/ticker_" + args.n + ".csv"
fileb = dataDir + "/value_" + args.n + ".csv"

if not path.exists(filea):
    print("please generate employee file first using tickerCsvGenerator.py")
    exit()

if __name__ == "__main__":

    types = list(pd.read_csv(filea).loc[:, 'type'])
    random.shuffle(types)
    with open(fileb, "w") as fh:
        writer = csv.writer(fh)
        i = 0
        writer.writerow(('id', 'type', 'value'))
        for type in types:
            i += 1
            # ("type", "value")
            writer.writerow((i, type, random.uniform(1e3, 1e4)))
