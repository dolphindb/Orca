import unittest
import orca
from setup.settings import *
from pandas.util.testing import *


class FunctionCompareTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    @property
    def pdf(self):
        return pd.DataFrame({'angles': [0, 3, 4], 'degrees': [360, 180, 360]},
                            index=['circle', 'triangle', 'rectangle'])

    @property
    def odf(self):
        return orca.DataFrame({'angles': [0, 3, 4], 'degrees': [360, 180, 360]},
                              index=['circle', 'triangle', 'rectangle'])

    def test_function_binary_operator_compare_dataframe(self):
        df = pd.DataFrame({'cost': [250, 150, 100],
                           'revenue': [100, 250, 300]},
                          index=['A', 'B', 'C'])
        assert_frame_equal((orca.DataFrame(df) == 100).to_pandas(), df == 100)
        assert_frame_equal((orca.DataFrame(df).eq(100)).to_pandas(), df.eq(100))
        assert_frame_equal((orca.DataFrame(df).ne(100)).to_pandas(), df.ne(100))

        other = pd.DataFrame({'revenue': [300, 250, 100, 150]},
                             index=['A', 'B', 'C', 'D'])

        # TODO BUG pandas result is False while orca result is nan
        # assert_frame_equal((orca.DataFrame(df).gt(orca.DataFrame(other))).to_pandas(), df.gt(other),
        #                    check_dtype=False)
        # assert_frame_equal((orca.DataFrame(df).eq(orca.DataFrame(other))).to_pandas(), df.eq(other),
        #                    check_dtype=False)
        # assert_frame_equal((orca.DataFrame(df).ne(orca.DataFrame(other))).to_pandas(), df.ne(other),
        #                    check_dtype=False)


if __name__ == '__main__':
    unittest.main()
