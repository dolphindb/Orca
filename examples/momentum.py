import dolphindb.orca as orca
import matplotlib.pyplot as plt

US = 'C:/DolphinDB/Orca/databases/USstocks.csv'

orca.connect('localhost', 8848)


def load_price_data(df):
    USstocks = df[df.date.dt.weekday.between(0, 4), df.PRC.notnull(), df.VOL.notnull()][
                   ['PERMNO', 'date', 'PRC', 'VOL', 'RET', 'SHROUT']
               ].sort_values(by=['PERMNO', 'date'])
    USstocks['PRC'] = USstocks.PRC.abs()
    USstocks['MV'] = USstocks.SHROUT * USstocks.PRC
    USstocks['cumretIndex'] = (USstocks + 1)['RET'].groupby('PERMNO', lazy=True).cumprod()
    USstocks['signal'] = (USstocks.shift(21) / USstocks.shift(252) - 1).groupby(
                            'PERMNO', lazy=True)['cumretIndex'].transform()
    return USstocks


def gen_trade_tables(df):
    USstocks = df[(df.PRC > 5), (df.MV > 100000), (df.VOL > 0), (df.signal.notnull())]
    USstocks = USstocks[['date', 'PERMNO', 'MV', 'signal']].sort_values(by='date')
    return USstocks


def form_portfolio(start_date, end_date, tradables, holding_days, groups, wt_scheme):
    ports = tradables[tradables.date.between(start_date, end_date)].groupby('date').filter('count(PERMNO) >= 100')
    ports['rank'] = ports.groupby('date')['signal'].transform('rank{{,true,{groups}}}'.format(groups=groups))
    ports['wt'] = 0.0
    
    ports_rank_eq_0 = (ports['rank'] == 0)
    ports_rank_eq_groups_sub_1 = (ports['rank'] == groups-1)
    if wt_scheme == 1:
        ports.loc[ports_rank_eq_0, 'wt'] = \
            ports[ports_rank_eq_0].groupby(['date'])['PERMNO'].transform(
                r'(PERMNO->-1\count(PERMNO)\{holding_days})'.format(holding_days=holding_days)
            )
        ports.loc[ports_rank_eq_groups_sub_1, 'wt'] = \
            ports[ports_rank_eq_groups_sub_1].groupby(['date'])['PERMNO'].transform(
                r'(PERMNO->1\count(PERMNO)\\{holding_days})'.format(holding_days=holding_days)
            )
    elif wt_scheme == 2:
        ports.loc[ports_rank_eq_0, 'wt'] = \
            ports[ports_rank_eq_0].groupby(['date'])['MV'].transform(
                r'(MV->-MV\sum(MV)\{holding_days})'.format(holding_days=holding_days)
            )
        ports.loc[ports_rank_eq_groups_sub_1, 'wt'] = \
            ports[ports_rank_eq_groups_sub_1].groupby(['date'])['MV'].transform(
                r'(MV->MV\sum(MV)\{holding_days})'.format(holding_days=holding_days)
            )
    ports = ports.loc[ports.wt != 0, ['PERMNO', 'date', 'wt']].sort_values(by=['PERMNO', 'date'])
    ports.rename(columns={'date': 'tranche'}, inplace=True)
    return ports


def calc_stock_pnl(ports, daily_rtn, holding_days, end_date, last_days):
    dates = ports[['tranche']].drop_duplicates().sort_values(by='tranche')

    dates_after_ages = orca.DataFrame()
    for age in range(1, holding_days+1):
        dates_after_age_i = dates.copy()
        dates_after_age_i['age'] = age
        dates_after_age_i['date_after_age'] = dates_after_age_i['tranche'].shift(-age)
        dates_after_ages.append(dates_after_age_i, inplace=True)

    pos = ports.merge(dates_after_ages, on='tranche')
    pos = pos.join(last_days, on='PERMNO')
    pos = pos.loc[(pos.date_after_age.notnull() & (pos.date_after_age <= pos.last_day.clip(upper=end_date))),
                  ['date_after_age', 'PERMNO', 'tranche', 'age', 'wt']]
    pos = pos.compute()
    pos.rename(columns={'date_after_age': 'date', 'wt': 'expr'}, inplace=True)
    pos['ret'] = 0.0
    pos['pnl'] = 0.0

    # use set_index to make it easy to equal join two Frames
    daily_rtn.set_index(['date', 'PERMNO'], inplace=True)
    pos.set_index(['date', 'PERMNO'], inplace=True)
    pos['ret'] = daily_rtn['RET']
    pos.reset_index(inplace=True)
    pos['expr'] = (pos.expr * (1 + pos.ret).cumprod()).groupby(
                    ['PERMNO', 'tranche'], lazy=True).transform()
    pos['pnl'] = pos.expr * pos.ret / (1 + pos.ret)

    return pos


def main():
    df = orca.read_csv(US)
    
    price_data = load_price_data(df)
    tradables = gen_trade_tables(price_data)
    
    start_date, end_date = orca.Timestamp("1996.01.01"), orca.Timestamp("2017.01.01")
    holding_days = 5
    groups = 10
    ports = form_portfolio(start_date, end_date, tradables, holding_days, groups, 2)
    daily_rtn = price_data.loc[price_data.date.between(start_date, end_date), ['date', 'PERMNO', 'RET']]

    last_days = price_data.groupby('PERMNO')['date'].max()
    last_days.rename("last_day", inplace=True)
    stock_pnl = calc_stock_pnl(ports, daily_rtn, holding_days, end_date, last_days)

    port_pnl = stock_pnl.groupby('date')['pnl'].sum()
    cumulative_return = port_pnl.cumsum()
    cumulative_return.plot()
    plt.show()


if __name__ == '__main__':
    main()

