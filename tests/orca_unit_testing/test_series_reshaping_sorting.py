from setup.settings import *
import unittest
import orca
import os.path as path
from pandas.util.testing import *


class SeriesReshapingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # configure data directory
        DATA_DIR = path.abspath(path.join(__file__, "../setup/data"))
        fileName = 'USPricesSample.csv'
        data = os.path.join(DATA_DIR, fileName)
        data = data.replace('\\', '/')

        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_series_reshaping_sorting_reorder_levels(self):
        arrays = [np.array(['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux']),
                  np.array(['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two'])]
        ps = pd.Series(np.random.randn(8), index=arrays)
        os = orca.Series(ps)
        assert_series_equal(os.reorder_levels([1, 0]).to_pandas(), ps.reorder_levels([1, 0]))