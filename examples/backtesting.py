import dolphindb.orca as orca
from datetime import datetime

orca.connect("localhost", 8848)

data_path = "D:/data/Western/"


def prepare():
    dailybar_stock = orca.read_csv(data_path + "dailybar_stock.csv")
    dailybar_stock_adjfactor = orca.read_csv(data_path + "dailybar_stock_adjfactor.csv", partitioned=False)
    dailybar_stock_extension = orca.read_csv(data_path + "dailybar_stock_extension.csv", partitioned=False)
    financials = orca.read_csv(data_path + "finacial_balancesheet.csv")
    financials = financials.sort_values(by=["symbol", "ann_date"])

    dailybar_stock = dailybar_stock.merge(dailybar_stock_adjfactor, on=["symbol", "odate"], how="left")
    dailybar_stock = dailybar_stock.merge(dailybar_stock_extension, on=["symbol", "odate"], how="left")
    dailybar_stock = dailybar_stock[
        ["symbol", "exchangeid_x", "odate", "open", "high", "low", "close_x",
         "volume", "amount", "change", "changep", "yclose", "adj_factor",
         "mcap", "tcap"]
    ]
    dailybar_stock.rename(columns={"exchangeid_x": "exchangeid", "close_x": "close"}, inplace=True)
    dailybar_stock["closeAdj"] = dailybar_stock.close * dailybar_stock.adj_factor

    dailybar_index = orca.read_csv(data_path + "dailybar_index.csv")
    index_ret = dailybar_index.loc[dailybar_index.symbol == "000001", ["odate", "close", "yclose"]]
    index_ret = index_ret.compute()
    index_ret["indexRet"] = index_ret.close / index_ret.yclose - 1
    index_ret = index_ret[["odate", "indexRet"]]

    dailybar_stock = dailybar_stock.merge(index_ret, on="odate", how="left")
    dailybar_stock.rename(columns={"odate": "date"}, inplace=True)
    index_ret.rename(columns={"odate": "date"}, inplace=True)

    financials["assetGrowth"] = financials.total_assets / financials.total_assets.shift(4) - 1
    dailybar_stock = dailybar_stock.merge_window(financials, -182, -1, left_on=["symbol", "date"], right_on=["symbol", "ann_date"])
    dailybar_stock = dailybar_stock.agg(
        {"ann_date": "last(ann_date)",
         "end_date": "last(end_date)",
         "total_cur_assets": "last(total_cur_assets)",
         "total_nca": "last(total_nca)",
         "total_assets": "last(total_assets)",
         "total_cur_liab": "last(total_cur_liab)",
         "total_ncl": "last(total_ncl)",
         "total_liab": "last(total_liab)",
         "assetGrowth": "last(assetGrowth)"
        })
    
    dailybar_stock = dailybar_stock[
            ~((dailybar_stock.symbol == "600000")
            & dailybar_stock.date.between("2019.08.05", "2019.08.23")
            & (dailybar_stock.volume == 0))
    ].sort_values(by=["symbol", "date"])
    return dailybar_stock


def load_price_data(df):
    stocks = df[
        ["symbol", "date", "close", "closeAdj", "yclose", "volume", "mcap",
         "indexRet", "total_assets", "total_liab", "total_cur_liab", "assetGrowth"]
    ].sort_values(by=["symbol", "date"])
    stocks["upLimit"] = stocks.close.notnull() & (stocks.close == (stocks.yclose * 1.1).round(2))
    stocks["downLimit"] = stocks.close.notnull() & (stocks.close == (stocks.yclose * 0.9).round(2))
    stocks["RET"] = (stocks / stocks.shift(1) - 1)["closeAdj"].groupby("symbol", lazy=True).transform()
    stocks["fRET"] = (stocks.shift(-1) / stocks - 1)["closeAdj"].groupby("symbol", lazy=True).transform()
    stocks["cumretIndex"] = (stocks + 1)["RET"].groupby("symbol", lazy=True).cumprod()
    stocks["turnover"] = (stocks.volume / (stocks.mcap / stocks.close)).groupby("symbol", lazy=True).transform()
    stocks["reversal"] = (-(stocks.shift(1) / stocks.shift(21) - 1))["cumretIndex"].groupby("symbol", lazy=True).transform()
    stocks["size"] = -stocks.mcap.log()
    stocks["DASTD"] = stocks.groupby("symbol", lazy=True)["RET"].transform("mstd{,60}")
    stocks["momentum"] = (-stocks.shift(21) / stocks.shift(252) - 1)["cumretIndex"].groupby("symbol", lazy=True).transform()
    stocks["CMRA"] = stocks.groupby("symbol", lazy=True)["RET"].transform("(RET->log(mmax(cumprod(1+RET), 252))-log(mmin(cumprod(1+RET), 252)))")
    stocks["BTOP"] = (stocks.total_assets - stocks.total_liab) / stocks.mcap
    stocks["DTOA"] = stocks.total_liab / stocks.total_assets
    stocks["MLEV"] = (stocks.total_liab - stocks.total_cur_liab) / stocks.mcap
    stocks["BLEV"] = (stocks.total_liab - stocks.total_cur_liab) / (stocks.total_assets - stocks.total_liab)
    stocks["turnover_m"] = stocks.groupby("symbol", lazy=True)["turnover"].transform("mavg{,21}")
    stocks["turnover_q"] = stocks.groupby("symbol", lazy=True)["turnover"].transform("mavg{,63}")
    stocks["turnover_y"] = stocks.groupby("symbol", lazy=True)["turnover"].transform("mavg{,252}")
    stocks["beta"] = stocks.groupby("symbol", lazy=True)["RET"].transform("mbeta{,indexRet,252}")
    return stocks


def get_factor_returns(df):
    stdize = r"(x->(x-avg(x))\std(x))"
    columns = ["assetGrowth", "beta", "reversal", "size", "DASTD", "momentum", "CMRA",
               "BTOP", "DTOA", "MLEV", "BLEV", "turnover_m", "turnover_q", "turnover_y"]
    tmp = df[
        df.assetGrowth.notnull(),
        df.reversal.notnull(),
        df["size"].notnull(),
        df.DASTD.notnull(),
        df.momentum.notnull(),
        df.fRET.notnull(),
        df.CMRA.notnull(),
        df.BTOP.notnull(),
        df.DTOA.notnull(),
        df.MLEV.notnull(),
        df.BLEV.notnull(),
        df.turnover_m.notnull(),
        df.turnover_q.notnull(),
        df.turnover_y.notnull(),
        df.date.between("2005.01.01", "2017.12.31")
    ][["symbol", "date", "fRET"] + columns].sort_values(by="date")
    tmp[columns] = tmp.groupby("date", lazy=True)[columns].transform(stdize)
    tmp["volatility"] = (tmp.DASTD + tmp.CMRA) / 2
    tmp["leverage"] = (tmp.MLEV + tmp.DTOA + tmp.BLEV) / 3
    tmp["liquidity"] = - (tmp.turnover_m + tmp.turnover_q + tmp.turnover_y) / 3
    
    x_columns = ["size", "volatility", "leverage", "momentum", "reversal", "liquidity"]
    tmp0 = tmp.groupby("date").ols(y="fRET", x=x_columns, column_names=["int"] + x_columns)
    tmp1 = tmp[["date", "symbol", "size", "volatility", "leverage",
                "momentum", "reversal", "liquidity"]].sort_values(by=["date", "symbol"])
    return tmp0, tmp1


def main():
    start_time = datetime.now()
    dailybar_stock = prepare()
    price_data = load_price_data(dailybar_stock)
    factor_returns, factors = get_factor_returns(price_data)
    t = factor_returns.copy()
    t.volatility = t.volatility.cumsum()
    t.leverage = t.leverage.cumsum()
    t.liquidity = t.liquidity.cumsum()
    t.momentum = t.momentum.cumsum()
    print('Time elapsed (hh:mm:ss.ms) {}'.format(datetime.now() - start_time))
    t.plt()


if __name__ == "__main__":
    main()
