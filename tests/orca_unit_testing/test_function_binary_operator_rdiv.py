import unittest
import orca
from setup.settings import *
from pandas.util.testing import *


class FunctionRdivTest(unittest.TestCase):
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

    def test_function_binary_operator_rdiv_dataframe(self):
        pre = self.pdf.rdiv(2)
        ore = self.odf.rdiv(2).to_pandas()
        # 2/0 in pandas equals to inf while in orca equals to nan,
        # thus we replace these values with zeros for correctness assertion
        pre[np.isinf(pre)] = 0
        ore[np.isnan(ore)] = 0
        assert_frame_equal(ore, pre)


if __name__ == '__main__':
    unittest.main()
