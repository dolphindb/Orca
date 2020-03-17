import unittest
import orca
from setup.settings import *
from pandas.util.testing import *


class FunctionPowTest(unittest.TestCase):
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

    def test_function_binary_operator_pow_dataframe(self):
        odf = orca.DataFrame(self.pdf)

        assert_frame_equal(odf.pow(2).to_pandas(), self.pdf.pow(2), check_dtype=False)

        # TODO NOT IMPLEMENTED ERROR
        # pre = pdf.pow(pd.Series([1, 2], index=["angles", "degrees"]))
        # ore = (odf.pow(orca.Series([1, 2], index=["angles", "degrees"]))).to_pandas()
        # assert_frame_equal(ore, pre)


if __name__ == '__main__':
    unittest.main()
