import unittest
import orca
from setup.settings import *
from pandas.util.testing import *


class FunctionDroplevelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_function_reshaping_sorting_transposing_droplevel_dataframe(self):
        pdf = pd.DataFrame([[1, 2, 3, 4],
                            [5, 6, 7, 8],
                            [9, 10, 11, 12]
                            ]).set_index([0, 1]).rename_axis(['a', 'b'])

        pdf.columns = pd.MultiIndex.from_tuples([
            ('c', 'e'), ('d', 'f')
        ], names=['level_1', 'level_2'])
        odf = orca.DataFrame(pdf)
        assert_frame_equal(pdf.droplevel('a'), odf.droplevel('a').to_pandas())
        # TODO：NotImplementedError: Orca does not support axis == 1
        # assert_frame_equal(pdf.droplevel('level_2',axis=1), odf.droplevel('level_2',axis=1).to_pandas())


if __name__ == '__main__':
    unittest.main()