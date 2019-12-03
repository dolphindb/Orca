import argparse
import random
import string
import os.path as path
import pandas as pd

# python tickerCsvGenerator.py 15

parser = argparse.ArgumentParser(description='Generate employee table')
parser.add_argument('n', help='rows of records in a file')
args = parser.parse_args()

dataDir = path.abspath(path.join(__file__, "../../testPerformance/setup/data"))
filea = dataDir + "/ticker_" + args.n + ".csv"


def random_string(length, chars=string.ascii_letters):
    return ''.join(random.choice(chars) for _ in range(length))


def gen_csv(N):
    # ("Name", "Dept", "Birth", "Salary")
    a = pd.date_range('20100101130000', periods=3650, freq="1D", name='date')
    tmp = pd.DataFrame(index=a)
    df = tmp.sample(N, replace=True)
    df.sort_index(inplace=True)
    ticker = list()
    type = list()
    bid = list()
    svalue = list()
    price = list()
    for _ in range(N):
        # yield (
        #     random_string(8), random_string(1, chars='abcdefg'), random.randint(1900, 2000), random.uniform(1e4, 1e5))
        ticker.append(random_string(8))
        type.append(random_string(1, chars='abcdefg'))
        bid.append(random.randint(1965, 2001))
        svalue.append(random.uniform(1e4, 1e5))
        price.append(random.uniform(1e2, 1e3))

    df["time"] = pd.date_range('20100101130000', periods=N, freq="ms", name='date')
    df["ticker"] = ticker
    df["type"] = type
    df["bid"] = bid
    df["svalue"] = svalue
    df["price"] = price
    df.to_csv(filea)


if __name__ == "__main__":
    gen_csv(int(args.n))
    # with open(filea, "w") as fh:
    #     writer = csv.writer(fh)
    #
    #     for row in gen_csv(length):
    #         writer.writerow(row)
