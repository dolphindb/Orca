import unittest
from setup.settings import *
from numpy.testing import *
from pandas.util.testing import *
import numpy as np
import dolphindb_numpy as dnp
import pandas as pd
import orca


class FunctionExpm1Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_function_math_expm1_scalar(self):
        self.assertEqual(dnp.expm1(1.2 + 1j), np.expm1(1.2 + 1j))
        self.assertEqual(dnp.expm1(1e-10), np.expm1(1e-10))
        self.assertEqual(dnp.expm1(0.5), np.expm1(0.5))
        self.assertEqual(dnp.expm1(1), np.expm1(1))
        self.assertEqual(dnp.expm1(-1), np.expm1(-1))
        self.assertEqual(dnp.expm1(0), np.expm1(0))
        self.assertEqual(dnp.isnan(dnp.expm1(dnp.nan)), True)
        self.assertEqual(np.isnan(np.expm1(np.nan)), True)

    def test_function_math_expm1_list(self):
        npa = np.expm1([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0, np.nan])
        dnpa = dnp.expm1([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0, dnp.nan])
        assert_array_equal(dnpa, npa)

    def test_function_math_expm1_array(self):
        npa = np.expm1(np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0, np.nan]))
        dnpa = dnp.expm1(dnp.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0, dnp.nan]))
        assert_array_equal(dnpa, npa)

    def test_function_math_expm1_series(self):
        ps = pd.Series([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0, np.nan])
        os = orca.Series(ps)
        # TODO：BUG
        # assert_series_equal(dnp.expm1(os).to_pandas(), np.expm1(ps))

    def test_function_math_expm1_dataframe(self):
        pdf = pd.DataFrame({"cola": [-1.7, -1.5, -0.2, np.nan, 0.2, 1.5, 1.7, 2.0, np.nan],
                            "colb": [-1.7, -1.5, -0.2, np.nan, 0.2, 1.5, 1.7, np.nan, 2.0]})
        odf = orca.DataFrame(pdf)
        # TODO：BUG
        # assert_frame_equal(dnp.expm1(odf).to_pandas(), np.expm1(pdf))


if __name__ == '__main__':
    unittest.main()
