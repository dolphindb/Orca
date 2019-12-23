import dolphindb.orca as orca


orca.connect("localhost", 8848, "admin", "123456")


def main():
    quotes = orca.read_table('dfs://TAQ', 'quotes')

    # select 500 most liquid stocks, get minute level returns,
    # and calculate the pair wise correlation
    date_value = '2007.08.01'
    num = 500
    syms = quotes[
        quotes.date == date_value,
        quotes.time.between('09:30:00', '15:59:59'),
        quotes.bid > 0,
        quotes.bid < quotes.ofr,
        quotes.ofr < quotes.bid * 1.2,
    ].groupby('symbol', sort=False, lazy=True).size().sort_values(ascending=False).head(num)

    tmp = quotes[
        quotes.date == date_value,
        quotes.time.between('09:30:00', '15:59:59'),
        quotes.symbol.isin(syms.index),
        quotes.bid > 0,
        quotes.bid < quotes.ofr,
        quotes.ofr < quotes.bid * 1.2
    ]
    price_matrix = tmp.pivot_table(None, tmp.time.dt.hourminute, 'symbol', 'mean(bid+ofr)/2')
    ret_matrix = price_matrix.pct_change()

    # get 10 most correlated stocks for each stock
    pairwise_corr = ret_matrix.corr().stack()
    pairwise_corr.rename('corr', inplace=True)
    most_correlated = pairwise_corr.groupby(level=0).filter('rank(corr, false) between 1:10')
    print(most_correlated)


if __name__ == '__main__':
    main()
