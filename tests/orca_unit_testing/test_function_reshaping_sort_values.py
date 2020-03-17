import unittest
import orca
from setup.settings import *
from pandas.util.testing import *


class FunctionSortValuesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_function_reshaping_sorting_transposing_sort_values_dataframe(self):
        pdf = pd.DataFrame({'col1': ['A', 'A', 'B', 'E', 'D', 'C'], 'col2': [2, 1, 9, 8, 7, 4], 'col3': [0, 1, 9, 4, 2, 3]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.sort_values(by=['col1']).to_pandas(), pdf.sort_values(by=['col1']))

        # TODO: DIFFERENCES
        pdf = pd.DataFrame({'col1': ['A', 'A', 'B', np.nan, 'D', 'C'], 'col2': [2, 1, 9, 8, 7, 4], 'col3': [0, 1, 9, 4, 2, 3]})
        odf = orca.DataFrame(pdf)
        # assert_frame_equal(odf.sort_values(by=['col1']).to_pandas(), pdf.sort_values(by=['col1']))


if __name__ == '__main__':
    unittest.main()