from columns import value_columns, ticker_columns
from setup.settings import *
import orca
orca.connect(HOST, PORT, "admin", "123456")


class OrcaPartitionDriver(object):
    def __init__(self, ticker_file, value_file):
        self.ticker_file = ticker_file
        self.value_file = value_file

        self.df_ticker = None
        self.df_value = None

        script = """
                login('admin', '123456')
                if(existsDatabase('dfs://testOrcaTicker'))
                   dropDatabase('dfs://testOrcaTicker')
                schema=extractTextSchema('{data1}')
                db=database('dfs://testOrcaTicker', HASH, [DATE,20])
                loadTextEx(db,`tickers,`date, '{data1}')
                
                if(existsDatabase('dfs://testOrcaValue'))
                   dropDatabase('dfs://testOrcaValue')
                schema=extractTextSchema('{data2}')
                db=database('dfs://testOrcaValue', HASH, [INT, 4])
                loadTextEx(db,`values,`id, '{data2}')
                """.format(data1=ticker_file, data2=value_file)

        s = orca.default_session()
        s.run(script)

    def a_load(self):
        self.df_ticker = orca.read_table("dfs://testOrcaTicker", 'tickers')
        self.df_ticker.columns = ticker_columns

        self.df_value = orca.read_table("dfs://testOrcaValue", 'values')
        self.df_value.columns = value_columns

    def groupby(self):
        self.df_ticker.groupby("type").agg({'svalue': 'mean', 'price': 'sum'})

    def filter(self):
        _ = self.df_ticker[self.df_ticker['type'] == 'a'].compute()

    def select(self):
        _ = self.df_ticker[["ticker", "type"]].compute()

    def sort(self):
        self.df_ticker.sort_values(by='ticker')

    def join(self):
        joined = self.df_ticker.merge(self.df_value, on='type')
        joined['total'] = joined['value'] + joined['svalue']

    def resample_D(self):
        self.df_ticker.resample('D', on='date')['svalue'].mean()

    def resample_3D(self):
        self.df_ticker.resample('3D', on='date')['svalue'].mean()

    def resample_Q(self):
        self.df_ticker.resample('Q', on='date')['svalue'].mean()

    def resample_3Q(self):
        self.df_ticker.resample('3Q', on='date')['svalue'].mean()

    def resample_A(self):
        self.df_ticker.resample('A', on='date')['svalue'].mean()

    def resample_3A(self):
        self.df_ticker.resample('3A', on='date')['svalue'].mean()
