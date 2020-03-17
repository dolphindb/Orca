import unittest
import orca
from setup.settings import *
from pandas.util.testing import *


class FunctionDivTest(unittest.TestCase):
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

    def test_function_binary_operator_div_dataframe_div_scalar(self):
        pre = self.pdf / 2
        ore = (self.odf / 2).to_pandas()
        assert_frame_equal(ore, pre)

        pre = self.pdf.div(2)
        ore = self.odf.div(2).to_pandas()
        assert_frame_equal(ore, pre)

    def test_function_binary_operator_div_dataframe_div_multiIndex_param_level_param_fill_value(self):
        pdf_multi = pd.DataFrame({'angles': [0, 3, 4, 4, 5, 6], 'degrees': [360, 180, 360, 360, 540, 720]},
                                 index=[['A', 'A', 'A', 'B', 'B', 'B'],
                                        ['circle', 'triangle', 'rectangle', 'square', 'pentagon', 'hexagon']])
        odf_multi = orca.DataFrame({'angles': [0, 3, 4, 4, 5, 6], 'degrees': [360, 180, 360, 360, 540, 720]},
                                   index=[['A', 'A', 'A', 'B', 'B', 'B'],
                                          ['circle', 'triangle', 'rectangle', 'square', 'pentagon', 'hexagon']])
        assert_frame_equal(odf_multi.to_pandas(), pdf_multi)

        # TODO: orca.DataFrame.rdiv(orca.DataFrame, level=1, fill_value=0)
        # pre = self.pdf_sh.rdiv(pdf_multi, level=1, fill_value=0)
        # ore = self.odf_sh.rdiv(odf_multi, level=1, fill_value=0).to_pandas()
        # assert_frame_equal(ore, pre)


if __name__ == '__main__':
    unittest.main()
