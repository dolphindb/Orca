import unittest
from setup.settings import *
from numpy.testing import *
from pandas.util.testing import *
import numpy as np
import dolphindb_numpy as dnp
import pandas as pd
import orca


class FunctionSqrtTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_function_math_sqrt_scalar(self):
        self.assertEqual(dnp.sqrt(1.2 + 1j), np.sqrt(1.2 + 1j))
        self.assertEqual(dnp.sqrt(1e-10), np.sqrt(1e-10))
        self.assertEqual(dnp.sqrt(0.5), np.sqrt(0.5))
        self.assertEqual(dnp.sqrt(1), np.sqrt(1))
        self.assertEqual(dnp.sqrt(0), np.sqrt(0))
        self.assertEqual(dnp.isnan(dnp.sqrt(dnp.nan)), True)
        self.assertEqual(np.isnan(np.sqrt(np.nan)), True)

    def test_function_math_sqrt_list(self):
        npa = np.sqrt([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, np.inf, np.nan])
        dnpa = dnp.sqrt([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, dnp.inf, dnp.nan])
        assert_array_equal(dnpa, npa)

    def test_function_math_sqrt_array(self):
        npa = np.sqrt(np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, np.inf, np.nan]))
        dnpa = dnp.sqrt(dnp.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, dnp.inf, dnp.nan]))
        assert_array_equal(dnpa, npa)

    def test_function_math_sqrt_series(self):
        ps = pd.Series([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, np.inf, np.nan])
        os = orca.Series(ps)
        assert_series_equal(dnp.sqrt(os).to_pandas(), np.sqrt(ps))

    def test_function_math_sqrt_dataframe(self):
        pdf = pd.DataFrame({"cola": [-1.7, -1.5, -0.2, 0.2, np.nan, 1.5, 1.7, np.inf, 2.0, np.nan],
                            "colb": [-1.7, -1.5, -0.2, 0.2, np.nan, 1.5, 1.7, np.inf, np.nan, 2.0]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(dnp.sqrt(odf).to_pandas(), np.sqrt(pdf))


if __name__ == '__main__':
    unittest.main()
