import unittest
import os
import orca
import pandas as pd
import matplotlib.pyplot as plt
import os.path as path
from setup.settings import *


def _create_odf_csv(data, dfsDatabase):
    # call function default_session() to get session object
    s = orca.default_session()
    dolphindb_script = """
    login("admin", "123456")
    dbPath="dfs://USPricesDB"
    if(existsDatabase(dbPath))
       dropDatabase(dbPath)
    cols = exec name from extractTextSchema('{data}')
    types = exec type from extractTextSchema('{data}')
    schema = table(50000:0, cols, types)
    tt=schema(schema).colDefs
    tt.drop!(`typeInt)
    tt.rename!(`name`type)
    db = database(dbPath, RANGE, 2010.01.04 2011.01.04  2012.01.04 2013.01.04 2014.01.04 2015.01.04  2016.01.04)
    USPrice = db.createPartitionedTable(schema, `USPrices, `date)
    db.loadTextEx(`USPrices,`date, '{data}' ,, tt)""".format(data=data)
    s.run(dolphindb_script)
    return orca.read_table(dfsDatabase, 'USPrices')


class Csv:
    pdf_csv = None
    odfs_csv = None


class DfsPLottingTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # configure data directory
        DATA_DIR = path.abspath(path.join(__file__, "../setup/data"))
        fileName = 'USPricesSample.csv'
        data = os.path.join(DATA_DIR, fileName)
        data = data.replace('\\', '/')
        dfsDatabase = "dfs://USPricesDB"

        # Orca connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

        # import
        Csv.pdf_csv = pd.read_csv(data)
        Csv.odfs_csv = _create_odf_csv(data, dfsDatabase)

    @property
    def odfs_series_plot(self):
        return Csv.odfs_csv.groupby('date').sum()['BID']

    @property
    def odfs_frame_plot(self):
        return Csv.odfs_csv.groupby('date').sum()

    @property
    def pdf_series_plot(self):
        return Csv.pdf_csv.groupby('date').sum()['BID']

    @property
    def pdf_frame_plot(self):
        return Csv.pdf_csv.groupby('date').sum()

    # def test_dfs_plot_area(self):
    #     self.pdf_series_plot.plot.area(x='date', y='BID')
    #     # plt.show()
    #     self.odfs_series_plot.plot.area(x='date', y='BID')
    #     # plt.show()

    def test_dfs_plot_bar(self):
        self.pdf_series_plot.plot.bar(x='date', y='BID')
        # plt.show()
        self.odfs_series_plot.plot.bar(x='date', y='BID')
        # plt.show()

    def test_dfs_plot_barh(self):
        self.pdf_series_plot.plot.barh(x='date', y='BID')
        # plt.show()
        self.odfs_series_plot.plot.barh(x='date', y='BID')
        # plt.show()

    def test_dfs_plot_box(self):
        self.odfs_series_plot.plot.box(by='date')
        # plt.show()
        self.pdf_series_plot.plot.box(by='date')
        # plt.show()

    def test_dfs_plot_density(self):
        self.odfs_series_plot.plot.density(bw_method=0.3)
        # plt.show()
        self.pdf_series_plot.plot.density(bw_method=0.3)
        # plt.show()

    def test_dfs_plot_hexbin(self):
        self.odfs_frame_plot.plot.hexbin(x='SHRCD', y='BID', gridsize=20)
        # plt.show()
        self.pdf_frame_plot.plot.hexbin(x='SHRCD', y='BID', gridsize=20)
        # plt.show()

    def test_dfs_plot_hist(self):
        self.odfs_series_plot.plot.hist(by='date', bins=10)
        # plt.show()
        self.pdf_series_plot.plot.hist(by='date', bins=10)
        # plt.show()

    def test_dfs_plot_kde(self):
        self.odfs_series_plot.plot.kde(bw_method=0.3)
        # plt.show()
        self.pdf_series_plot.plot.kde(bw_method=0.3)
        # plt.show()

    def test_dfs_plot_line(self):
        self.pdf_series_plot.plot.line(x='date', y='BID')
        # plt.show()
        self.odfs_series_plot.plot.line(x='date', y='BID')
        # plt.show()

    def test_dfs_plot_pie(self):
        self.pdf_series_plot.plot.pie(y='date', figsize=(6, 3))
        # plt.show()
        self.odfs_series_plot.plot.pie(y='date', figsize=(6, 3))
        # plt.show()

    def test_dfs_plot_scatter(self):
        self.odfs_frame_plot.plot.scatter(x='SHRCD', y='BID')
        # plt.show()
        self.pdf_frame_plot.plot.scatter(x='SHRCD', y='BID')
        # plt.show()

    def test_dfs_boxplot(self):
        self.odfs_frame_plot.boxplot()
        # plt.show()
        self.pdf_frame_plot.boxplot()
        # plt.show()

    def test_dfs_hist(self):
        self.odfs_series_plot.hist()
        # plt.show()
        self.pdf_series_plot.hist()
        # plt.show()


if __name__ == '__main__':
    unittest.main()