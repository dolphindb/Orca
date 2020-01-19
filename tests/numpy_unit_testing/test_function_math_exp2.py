import unittest
from setup.settings import *
from numpy.testing import *
from pandas.util.testing import *
import numpy as np
import dolphindb_numpy as dnp
import pandas as pd
import orca


class FunctionExp2Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_function_math_exp2_scalar(self):
        self.assertEqual(dnp.exp2(1.2 + 1j), np.exp2(1.2 + 1j))
        self.assertEqual(dnp.exp2(0.5), np.exp2(0.5))
        self.assertEqual(dnp.exp2(1), np.exp2(1))
        self.assertEqual(dnp.exp2(-1), np.exp2(-1))
        self.assertEqual(dnp.exp2(0), np.exp2(0))
        self.assertEqual(dnp.isnan(dnp.exp2(dnp.nan)), True)
        self.assertEqual(np.isnan(np.exp2(np.nan)), True)

    def test_function_math_exp2_list(self):
        npa = np.exp2([2, 3, 9, -5.5, 0, np.nan])
        dnpa = dnp.exp2([2, 3, 9, -5.5, 0, dnp.nan])
        assert_array_equal(dnpa, npa)

    def test_function_math_exp2_array(self):
        npa = np.exp2(np.array([2, 3, 9, -5.5, 0, np.nan]))
        dnpa = dnp.exp2(dnp.array([2, 3, 9, -5.5, 0, dnp.nan]))
        assert_array_equal(dnpa, npa)

    def test_function_math_exp2_series(self):
        ps = pd.Series([2, 3, 9, -5.5, 0, np.nan])
        os = orca.Series(ps)
        # TODO： BUG
        # assert_series_equal(dnp.exp2(os).to_pandas(), np.exp2(ps))

    def test_function_math_exp2_dataframe(self):
        pdf = pd.DataFrame({"cola": [-1.7, -1.5, -0.2, np.nan, 2, 3, 9, -5.5, 0, np.nan],
                            "colb": [-1.7, -1.5, -0.2, np.nan, 2, 3, 9, -5.5, np.nan, 0]})
        odf = orca.DataFrame(pdf)
        # TODO： BUG
        # assert_frame_equal(dnp.exp2(odf).to_pandas(), np.exp2(pdf))


if __name__ == '__main__':
    unittest.main()
