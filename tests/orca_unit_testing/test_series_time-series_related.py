import unittest
import orca
from setup.settings import *
from pandas.util.testing import *


class SeriesTSRelatedTest(unittest.TestCase):
    def setUp(self):
       self.PRECISION = 5

    @classmethod
    def setUpClass(cls):
       # connect to a DolphinDB server
       orca.connect(HOST, PORT, "admin", "123456")

    def test_series_time_series_related_between_time(self):
        idx = pd.date_range('2018-04-09', periods=4, freq='1D20min')
        ps = pd.Series([1, 2, 3, 4], index=idx)
        os = orca.Series(ps)
        assert_series_equal(os.between_time('0:15', '0:45').to_pandas(), ps.between_time('0:15', '0:45'))
        assert_series_equal(os.between_time('0:45', '0:15').to_pandas(), ps.between_time('0:45', '0:15'))
