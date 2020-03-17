import unittest
import orca
from setup.settings import *
from pandas.util.testing import *


class FunctionReorderLevelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_function_reshaping_sorting_transposing_reorder_levels_dataframe(self):
        arrays = [np.array(['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux']),
                  np.array(['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two'])]
        pdf = pd.DataFrame(np.random.randn(8, 4), index=arrays)
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.reorder_levels([1, 0], axis=0).to_pandas(), pdf.reorder_levels([1, 0], axis=0))


if __name__ == '__main__':
    unittest.main()