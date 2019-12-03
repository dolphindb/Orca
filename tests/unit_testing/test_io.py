import unittest
import orca
import os.path as path
from setup.settings import *
from pandas.util.testing import *


class InputOutputTest(unittest.TestCase):

    def setUp(self):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def loadData(self, filename):
        # configure data directory
        DATA_DIR = path.abspath(path.join(__file__, "../setup/data"))
        data = os.path.join(DATA_DIR, filename)
        data = data.replace('\\', '/')
        return data

    def test_read_csv_param_sep(self):
        data = self.loadData('USPricesSample.csv')
        pdf = pd.read_csv(data, parse_dates=[1])
        pdf.iloc[:, 9].fillna("", inplace=True)
        odf = orca.read_csv(data, dtype={"DLSTCD": "DOUBLE", "DLPRC": "DOUBLE"}).to_pandas()
        assert_frame_equal(pdf, odf, check_dtype=False)

        # test white space
        data = self.loadData('test_io.csv')
        pdf = pd.read_csv(data, parse_dates=[1], sep=" ")
        odf = orca.read_csv(data, sep=" ").to_pandas()
        assert_frame_equal(pdf, odf, check_dtype=False)

        # test delimiter
        odf = orca.read_csv(data, delimiter=" ").to_pandas()
        assert_frame_equal(pdf, odf, check_dtype=False)

    def test_read_csv_param_names(self):
        # whitout header
        data = self.loadData('test_io_names.csv')
        pdf = pd.read_csv(data, parse_dates=[1], names=['A', 'B', 'C'])
        # print(pdf)
        odf = orca.read_csv(data, names=['A', 'B', 'C']).to_pandas()
        # print(odf)
        assert_frame_equal(pdf, odf, check_dtype=False)

        # with header
        # pandas will parse the header to data, but orca will delete it.
        data = self.loadData('test_io.csv')
        pdf = pd.read_csv(data, parse_dates=[1], names=['A', 'B', 'C'], sep=' ')
        odf = orca.read_csv(data, names=['A', 'B', 'C'], sep=' ').to_pandas()
        # assert_frame_equal(pdf, odf, check_dtype=False)

    def test_read_csv_param_index_col(self):
        # test white space
        data = self.loadData('test_io.csv')
        pdf = pd.read_csv(data, parse_dates=[1], sep=" ", index_col=[1])
        # print(pdf)
        odf = orca.read_csv(data, sep=" ", index_col=1).to_pandas()
        # print(odf)
        assert_frame_equal(pdf, odf, check_dtype=False)

        data = self.loadData('test_io.csv')
        pdf = pd.read_csv(data, parse_dates=[1], sep=" ", index_col=['date'])
        odf = orca.read_csv(data, sep=" ", index_col=['date']).to_pandas()
        assert_frame_equal(pdf, odf, check_dtype=False)

        # whitout header
        data = self.loadData('test_io_names.csv')
        pdf = pd.read_csv(data, parse_dates=[1], names=['A', 'B', 'C'], index_col=1)
        # print(pdf)
        odf = orca.read_csv(data, names=['A', 'B', 'C'], index_col=1).to_pandas()
        # print(odf)
        assert_frame_equal(pdf, odf, check_dtype=False)

        # whitout header and use set names
        data = self.loadData('test_io_names.csv')
        pdf = pd.read_csv(data, parse_dates=[1], names=['A', 'B', 'C'], index_col=['A'])
        # print(pdf)
        odf = orca.read_csv(data, names=['A', 'B', 'C'], index_col=['A']).to_pandas()
        # print(odf)
        assert_frame_equal(pdf, odf, check_dtype=False)

    def test_read_csv_param_index_engine(self):
        # just to test where engine to use
        data = self.loadData('test_io_names.csv')
        pdf = pd.read_csv(data, parse_dates=[1], names=['A', 'B', 'C'], index_col=['A'])
        # print(pdf)
        odf = orca.read_csv(data, names=['A', 'B', 'C'], index_col=['A'], engine='c', parse_dates=[1]).to_pandas()
        # print(odf)
        assert_frame_equal(pdf, odf, check_dtype=False)

    def test_read_csv_param_usecols(self):
        # tip:parse_dates choose the data of order dertermine by  usecols
        data = self.loadData('test_io_names.csv')
        pdf = pd.read_csv(data, parse_dates=[0], names=['A', 'B', 'C'], usecols=[1, 2])
        # print(pdf)
        odf = orca.read_csv(data, names=['A', 'B', 'C'], usecols=[1, 2]).to_pandas()
        # print(odf)
        assert_frame_equal(pdf, odf, check_dtype=False)

        # without the header and use the names
        data = self.loadData('test_io_names.csv')
        pdf = pd.read_csv(data, parse_dates=[0], names=['A', 'B', 'C'], usecols=['B'])
        # print(pdf)
        odf = orca.read_csv(data, names=['A', 'B', 'C'], usecols=['B']).to_pandas()
        # print(odf)
        assert_frame_equal(pdf, odf, check_dtype=False)

        data = self.loadData('test_io.csv')
        pdf = pd.read_csv(data, parse_dates=[0], sep=" ", usecols=['date'])
        odf = orca.read_csv(data, sep=" ", usecols=['date']).to_pandas()
        assert_frame_equal(pdf, odf, check_dtype=False)

    def test_read_csv_param_squeeze(self):
        # TODO pandas的Seires是有name的,而orca的没有name
        data = self.loadData('test_io_squeeze.csv')
        pdf = pd.read_csv(data, squeeze=True)
        # print(pdf)
        odf = orca.read_csv(data, squeeze=True).to_pandas()
        # print(odf)
        # assert_series_equal(pdf, odf, check_dtype=False)

        # without the header and use the names
        data = self.loadData('test_io_names.csv')
        pdf = pd.read_csv(data, parse_dates=[0], names=['A', 'B', 'C'], usecols=['B'], squeeze=True)
        # print(pdf)
        odf = orca.read_csv(data, names=['A', 'B', 'C'], usecols=['B'], squeeze=True).to_pandas()
        # print(odf)
        # assert_series_equal(pdf, odf, check_dtype=False)

    def test_read_csv_param_prefix(self):
        # whitout header and use set names, the prefix is not used
        data = self.loadData('test_io_names.csv')
        pdf = pd.read_csv(data, parse_dates=[1], names=['A', 'B', 'C'], prefix='X')
        # print(pdf)
        odf = orca.read_csv(data, names=['A', 'B', 'C'], prefix='X').to_pandas()
        # print(odf)
        assert_frame_equal(pdf, odf, check_dtype=False)

        # without header and names
        data = self.loadData('test_io_names.csv')
        pdf = pd.read_csv(data, parse_dates=[1], header=None, prefix='X')
        # print(pdf)
        odf = orca.read_csv(data, prefix='X').to_pandas()
        # print(odf)
        assert_frame_equal(pdf, odf, check_dtype=False)

    def test_read(self):
        data = self.loadData('test_io_names.csv')
        pdf = pd.read_csv(data, parse_dates=[1])
        # feather to_parquet need dependency pyarraw
        file = pdf.to_html()
        # assert_equal(pd.read_html(file)[0], orca.read_html(file)[0])

        # pdf.to_pickle("test.pickle")
        data = self.loadData('test.pickle')
        # assert_frame_equal(pd.read_pickle(data), orca.read_pickle(data))

        file = pdf.to_msgpack()
        # assert_frame_equal(pd.read_msgpack(file), orca.read_msgpack(file))
        '''
        pdf.to_parquet("test.parquet")
        data = self.loadData('test.parquet')
        assert_frame_equal(pd.read_parquet(data), orca.read_parquet(data))
       

        pdf.to_feather("test.feather")
        data=self.loadData('test.feather')
        assert_frame_equal(pd.read_hdf(file), orca.read_hdf(file))
         '''


if __name__ == '__main__':
    unittest.main()
