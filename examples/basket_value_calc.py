import dolphindb.orca as orca


orca.connect('localhost', 8848, 'admin', '123456')


def main():
    quotes = orca.read_table('dfs://TAQ', 'quotes')

    date_value = '2007.09.04'
    num = 500
    syms = quotes[
        quotes.date == date_value,
        quotes.time.between('09:30:00', '15:59:59'),
        quotes.bid > 0,
        quotes.bid < quotes.ofr,
        quotes.ofr < quotes.bid * 1.2
    ].groupby('symbol', sort=False, lazt=True).size().sort_values(ascending=False).head(num)

    data = quotes[
        quotes.date == date_value,
        quotes.symbol.isin(syms.index),
        quotes.bid > 0,
        quotes.bid < quotes.ofr,
        quotes.ofr < quotes.bid * 1.2,
        quotes.time.between('09:30:00', '15:59:59')
    ]
    data.set_index(['time', 'symbol'], inplace=True)
    data = ((data.bid+data.ofr) / 2).groupby(level=[0,1], lazy=True).mean()
    data = data.unstack(level=-1).ffill().sum(axis=1)


if __name__ == '__main__':
    main()
