import unittest
import orca
from setup.settings import *
from pandas.util.testing import *


class FunctionTruedivTest(unittest.TestCase):
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

    def test_function_binary_operator_truediv_dataframe(self):
        pre = self.pdf.truediv(2)
        ore = (self.odf.truediv(2)).to_pandas()
        assert_frame_equal(ore, pre)

        pre = self.pdf.truediv(2, axis='index')
        ore = (self.odf.truediv(2, axis='index')).to_pandas()
        assert_frame_equal(ore, pre)

        pre = self.pdf.truediv(2, fill_value=0)
        ore = (self.odf.truediv(2, fill_value=0)).to_pandas()
        assert_frame_equal(ore, pre)

        pdf_multi = pd.DataFrame({'angles': [0, 3, 4, 4, 5, 6], 'degrees': [360, 180, 360, 360, 540, 720]},
                                 index=[['A', 'A', 'A', 'B', 'B', 'B'],
                                        ['circle', 'triangle', 'rectangle', 'square', 'pentagon', 'hexagon']])
        odf_multi = orca.DataFrame({'angles': [0, 3, 4, 4, 5, 6], 'degrees': [360, 180, 360, 360, 540, 720]},
                                   index=[['A', 'A', 'A', 'B', 'B', 'B'],
                                          ['circle', 'triangle', 'rectangle', 'square', 'pentagon', 'hexagon']])
        assert_frame_equal(pdf_multi, odf_multi.to_pandas())
        # TODO: BUG cannot join with no overlapping index names
        # pre = self.pdf_sh.truediv(pdf_multi, level=1, fill_value=0)
        # ore = self.odf_sh.truediv(odf_multi, level=1, fill_value=0).to_pandas()
        # assert_frame_equal(ore, pre)


if __name__ == '__main__':
    unittest.main()
