import unittest
import os
import orca
import pandas as pd
import matplotlib.pyplot as plt
import os.path as path
import base64
from io import BytesIO
from setup.settings import *


class Csv(object):
    pdf_csv = None
    odf_csv = None


class PLottingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # configure data directory
        DATA_DIR = path.abspath(path.join(__file__, "../setup/data"))
        fileName = 'USPricesSample.csv'
        data = os.path.join(DATA_DIR, fileName)
        data = data.replace('\\', '/')

        # Orca connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

        # import
        Csv.odf_csv = orca.read_csv(data)
        Csv.odf_csv.set_index('date', inplace=True)
        Csv.pdf_csv = pd.read_csv(data)
        Csv.pdf_csv.set_index('date', inplace=True)

    @property
    def odf_series_plot(self):
        return Csv.odf_csv.groupby('date').sum()['BID']

    @property
    def odf_frame_plot(self):
        return Csv.odf_csv.groupby('date').sum()

    @property
    def pdf_series_plot(self):
        return Csv.pdf_csv.groupby('date').sum()['BID']

    @property
    def pdf_frame_plot(self):
        return Csv.pdf_csv.groupby('date').sum()

    # def test_plot_area(self):
    #     self.odf_series_plot.plot.area(x='date', y='BID')
    #     # plt.show()
    #     self.pdf_series_plot.plot.area(x='date', y='BID')
    #     # plt.show()

    def test_plot_bar(self):
        self.odf_series_plot.plot.bar(x='date', y='BID')
        # plt.show()
        self.pdf_series_plot.plot.bar(x='date', y='BID')
        # plt.show()

    def test_plot_barh(self):
        self.odf_series_plot.plot.barh(x='date', y='BID')
        # plt.show()
        self.pdf_series_plot.plot.barh(x='date', y='BID')
        # plt.show()

    def test_plot_box(self):
        self.odf_series_plot.plot.box(by='date')
        # plt.show()
        self.pdf_series_plot.plot.box(by='date')
        # plt.show()

    def test_plot_density(self):
        self.odf_series_plot.plot.density(bw_method=0.3)
        # plt.show()
        self.pdf_series_plot.plot.density(bw_method=0.3)
        # plt.show()

    def test_plot_hexbin(self):
        self.odf_frame_plot.plot.hexbin(x='SHRCD', y='BID', gridsize=20)
        # plt.show()
        self.pdf_frame_plot.plot.hexbin(x='SHRCD', y='BID', gridsize=20)
        # plt.show()

    def test_plot_hist(self):
        self.odf_series_plot.plot.hist(by='date', bins=10)
        # plt.show()
        self.pdf_series_plot.plot.hist(by='date', bins=10)
        # plt.show()

    def test_series_hist(self):
        # TODO: NOT IMPLEMENTED
        pass
        # pdf = pd.DataFrame({
        #     'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 50],
        # }, index=[0, 1, 3, 5, 6, 8, 9, 9, 9, 10, 10])
        #
        # odf = orca.DataFrame(pdf)
        #
        # def plot_to_base64(ax):
        #     bytes_data = BytesIO()
        #     ax.figure.savefig(bytes_data, format='png')
        #     bytes_data.seek(0)
        #     b64_data = base64.b64encode(bytes_data.read())
        #     plt.close(ax.figure)
        #     return b64_data
        #
        # _, ax1 = plt.subplots(1, 1)
        # # Using plot.hist() because pandas changes ticos props when called hist()
        # ax1 = pdf['a'].plot.hist()
        # _, ax2 = plt.subplots(1, 1)
        # ax2 = odf['a'].hist()
        # self.assertEqual(plot_to_base64(ax1), plot_to_base64(ax2))


    def test_plot_kde(self):
        self.odf_series_plot.plot.kde(bw_method=0.3)
        # plt.show()
        self.pdf_series_plot.plot.kde(bw_method=0.3)
        # plt.show()

    # def test_plot_line(self):
    #     self.odf_series_plot.plot.line(x='date', y='BID')
    #     # plt.show()
    #     self.pdf_series_plot.plot.line(x='date', y='BID')
    #     # plt.show()

    def test_plot_pie(self):
        self.odf_series_plot.plot.pie(y='date', figsize=(6, 3))
        # plt.show()
        self.pdf_series_plot.plot.pie(y='date', figsize=(6, 3))
        # plt.show()

    def test_plot_scatter(self):
        self.odf_frame_plot.plot.scatter(x='SHRCD', y='BID')
        # plt.show()
        self.pdf_frame_plot.plot.scatter(x='SHRCD', y='BID')
        # plt.show()

    def test_boxplot(self):
        self.odf_frame_plot.boxplot()
        # plt.show()
        self.pdf_frame_plot.boxplot()
        # plt.show()

    def test_hist(self):
        self.pdf_series_plot.hist()
        # plt.show()
        self.odf_series_plot.hist()
        # plt.show()


if __name__ == '__main__':
    unittest.main()