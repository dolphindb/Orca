import unittest
import orca
from setup.settings import *
from pandas.util.testing import *


class FunctionMulTest(unittest.TestCase):
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

    def test_function_binary_operator_mul_dataframe(self):
        p_other = pd.DataFrame({'angles': [0, 3, 4]}, index=['circle', 'triangle', 'rectangle'])
        o_other = orca.DataFrame({'angles': [0, 3, 4]}, index=['circle', 'triangle', 'rectangle'])
        assert_frame_equal(o_other.to_pandas(), p_other)

        p_index = pd.DataFrame({'angles': [3, 5, 8], 'degrees': [2, 5, 7]}, index=['circle', 'triangle', 'rectangle'])
        o_index = orca.DataFrame({'angles': [3, 5, 8], 'degrees': [2, 5, 7]}, index=['circle', 'triangle', 'rectangle'])
        assert_frame_equal(o_index.to_pandas(), p_index)

        pre = self.pdf * p_other
        ore = (self.odf * o_other).to_pandas()
        assert_frame_equal(ore, pre)

        pre = self.pdf.mul(p_other)
        ore = self.odf.mul(o_other).to_pandas()
        assert_frame_equal(ore, pre)

        pre = self.pdf.mul(p_index)
        ore = self.odf.mul(o_index).to_pandas()
        assert_frame_equal(ore, pre)


if __name__ == '__main__':
    unittest.main()
