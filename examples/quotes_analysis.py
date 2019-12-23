import dolphindb.orca as orca
import matplotlib.pyplot as plt


orca.connect('localhost', 8848, 'admin', '123456')


def main():
    quotes = orca.read_table('dfs://TAQ', 'quotes')

    len(quotes)

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

    tmp = quotes[
        quotes.date.between('2007.09.25', '2007.09.27'),
        quotes.time.between('09:30:00', '15:59:59'),
        quotes.ofr > quotes.bid,
        quotes.bid > 0,
        quotes.ofr / quotes.bid < 1.2
    ]
    tmp.set_index('time', inplace=True)
    quote_size = (tmp.bidsiz + tmp.ofrsiz).groupby(tmp.index.hourminute, lazy=True).mean()
    quote_size.plot()
    plt.show()


    # Trades
    trades = orca.read_table('dfs://TAQ', 'trades')
    tmp = trades[
        trades.date.between('2007.09.25', '2007.09.27'),
        trades.time.between('09:30:00', '15:59:59')
    ]
    volumn = tmp.groupby(tmp.time.hourminute, lazy=True)['size'].sum()
    volumn.plot()
    plt.show()


    # EQY
    trade = orca.read_table('dfs://EQY', 'trade')
    nbbo = orca.read_table('dfs://EQY', 'nbbo')

    tmp = nbbo[
        nbbo.time.dt.time.between('09:30:00', '15:59:59'),
        nbbo.offer_price > nbbo.bid_price,
        nbbo.bid_price > 0,
        nbbo.offer_price / nbbo.bid_price < 1.2,
    ]
    tmp.set_index('time', inplace=True)
    avg_spread = ((tmp.offer_price-tmp.bid_price) / (tmp.offer_price+tmp.bid_price) * 2).groupby(tmp.index.hourminute, lazy=True).mean()
    avg_spread.plot()
    plt.show()

    quote_size = (tmp.offer_size+tmp.bid_size).groupby(tmp.index.hourminute, lazy=True).mean()
    quote_size.plot()
    plt.show()

    tmp = trade[trade.time.dt.time.between('09:30:00', '15:59:59')]
    tmp.set_index('time', inplace=True)
    volume = tmp.groupby(tmp.index.hourminute, lazy=True)['trade_volume'].sum()
    volume.plot()
    plt.show()

    # Market open
    tmp = quotes[
        quotes.date.between('2007.09.25', '2007.09.27'),
        quotes.bid > 0,
        quotes.bid < quotes.ofr,
        quotes.ofr < quotes.bid * 1.2,
        quotes.time.between('09:30:00', '15:59:59'),
        ~quotes.symbol.isin(['ZVZZT', 'ZXZZT', 'ZWZZT'])
    ]
    tmp.set_index('time', inplace=True)
    price = orca.DataFrame()
    price['price'] = ((tmp.bid+tmp.ofr) / 2).groupby(['symbol', 'date', tmp.index.hourminute], lazy=True).mean()
    price['ret'] = price.price.pct_change().groupby(level='symbol', lazy=True).transform()
    disp = price[price.ret < 1].groupby(level='time', lazy=True)['ret'].std()
    disp.plot(x='time')
    plt.show()


if __name__ == '__main__':
    main()
