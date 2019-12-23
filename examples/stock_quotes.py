import dolphindb.orca as orca
import matplotlib.pyplot as plt


orca.connect('localhost', 8848, 'admin', '123456')


def main():
    quotes = orca.read_table('dfs://TAQ', 'quotes')

    x = quotes[
        quotes.symbol == 'LEH',
        quotes.date == '2007.08.21',
        quotes.time == '15:59:59'
    ][['symbol', 'time', 'bid', 'ofr']].compute()

    # calculate the average bid-ask spread of LEH on 2007.08.31
    tmp = quotes[
        quotes.date == '2007.08.31',
        quotes.symbol == 'LEH',
        quotes.time.between('09:30:00', '15:59:59'),
        quotes.ofr > quotes.bid,
        quotes.bid > 0,
        quotes.ofr / quotes.bid < 1.2
    ]
    tmp.set_index('time', inplace=True)
    avg_spread = ((tmp.ofr-tmp.bid) / (tmp.ofr+tmp.bid) * 2).groupby(tmp.index.hourminute, lazy=True).mean()
    avg_spread.plot()
    plt.show()

    # calculate the average bid-ask spread of all stocks on 2007.09.25
    tmp = quotes[
        quotes.date == '2007.09.25',
        quotes.time.between('09:30:00', '15:59:59'),
        quotes.ofr > quotes.bid,
        quotes.bid > 0,
        quotes.ofr / quotes.bid < 1.2
    ]
    tmp.set_index('time', inplace=True)
    avg_spread = ((tmp.ofr-tmp.bid) / (tmp.ofr+tmp.bid) * 2).groupby(tmp.index.hourminute, lazy=True).mean()
    avg_spread.plot()
    plt.show()

    tmp = quotes[
        quotes.date.between('2007.09.25', '2007.09.27'),
        quotes.time.between('09:30:00', '15:59:59'),
        quotes.ofr > quotes.bid,
        quotes.bid > 0,
        quotes.ofr / quotes.bid < 1.2
    ]
    tmp.set_index('time', inplace=True)
    avg_spread = ((tmp.ofr-tmp.bid) / (tmp.ofr+tmp.bid) * 2).groupby(tmp.index.hourminute, lazy=True).mean()
    avg_spread.plot()
    plt.show()


if __name__ == '__main__':
    main()
