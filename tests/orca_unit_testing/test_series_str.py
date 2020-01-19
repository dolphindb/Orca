import unittest
import orca
from setup.settings import *
from pandas.util.testing import *


class SeriesStrTest(unittest.TestCase):
    def setUp(self):
       self.PRECISION = 5

    @classmethod
    def setUpClass(cls):
       # connect to a DolphinDB server
       orca.connect(HOST, PORT, "admin", "123456")

    @property
    def ps(self):
        return pd.Series(['Foo', 'ss ', 'sW', 'qa'], name='x')

    @property
    def os(self):
        return orca.Series(self.ps)

    @property
    def psa(self):
        return pd.Series([10, 1, 19, np.nan], index=['a', 'b', 'c', 'd'])

    @property
    def psb(self):
        return pd.Series([-1, np.nan, 1, np.nan], index=['a', 'b', 'd', 'e'])

    def test_series_str_count(self):
        assert_series_equal(self.ps.str.count('a'), self.os.str.count("a").to_pandas(),check_dtype=False)

    def test_series_str_startsWith(self):
        assert_series_equal(self.ps.str.startswith('Fo'), self.os.str.startswith('Fo').to_pandas(), check_dtype=False)

    def test_series_str_endswith(self):
        assert_series_equal(self.ps.str.endswith('W'), self.os.str.endswith('W').to_pandas(), check_dtype=False)

    def test_series_str_find(self):
        assert_series_equal(self.ps.str.find('Fo'), self.os.str.find('Fo').to_pandas(), check_dtype=False)

    def test_series_str_get(self):
        assert_series_equal(self.ps.str.get(1), self.os.str.get(1).to_pandas(), check_dtype=False)

    def test_series_str_just(self):
        # TODO: pandas not cut the str when length is not enough
        # assert_series_equal(self.ps.str.ljust(1), self.os.str.ljust(1).to_pandas(), check_dtype=False)
        assert_series_equal(self.ps.str.ljust(10), self.os.str.ljust(10).to_pandas(), check_dtype=False)
        assert_series_equal(self.ps.str.ljust(10,'A'), self.os.str.ljust(10,'A').to_pandas(), check_dtype=False)

        assert_series_equal(self.ps.str.rjust(10), self.os.str.rjust(10).to_pandas(), check_dtype=False)
        assert_series_equal(self.ps.str.rjust(10, 'A'), self.os.str.rjust(10, 'A').to_pandas(), check_dtype=False)

    def test_series_str_is(self):
        assert_series_equal(self.ps.str.isalnum(),self.os.str.isalnum().to_pandas())
        assert_series_equal(self.ps.str.isalpha(), self.os.str.isalpha().to_pandas())
        assert_series_equal(self.ps.str.isdigit(), self.os.str.isdigit().to_pandas())
        assert_series_equal(self.ps.str.isspace(), self.os.str.isspace().to_pandas())
        assert_series_equal(self.ps.str.islower(), self.os.str.islower().to_pandas())
        assert_series_equal(self.ps.str.isupper(), self.os.str.isupper().to_pandas())
        assert_series_equal(self.ps.str.istitle(), self.os.str.istitle().to_pandas())
        assert_series_equal(self.ps.str.isnumeric(), self.os.str.isnumeric().to_pandas())
        assert_series_equal(self.ps.str.isdecimal(), self.os.str.isdecimal().to_pandas())
