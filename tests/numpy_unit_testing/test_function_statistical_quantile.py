import unittest
from setup.settings import *
from numpy.testing import *
import numpy as np
import dolphindb_numpy as dnp
import pandas as pd
import orca


class FunctionquantileTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_function_math_quantile_scalar(self):
        self.assertEqual(dnp.quantile(0.5, 0.30), np.quantile(0.5, 0.30))
        self.assertEqual(dnp.quantile(1, 0.30), np.quantile(1, 0.30))
        self.assertEqual(dnp.quantile(-1, 0.30), np.quantile(-1, 0.30))
        self.assertEqual(dnp.quantile(0, 0.30), np.quantile(0, 0.30))
        self.assertEqual(dnp.isnan(dnp.quantile(dnp.nan, 0.30)), True)
        self.assertEqual(np.isnan(np.quantile(np.nan, 0.30)), True)

    def test_function_math_quantile_list(self):
        npa = np.quantile([1, 8, 27, -27, 0, 5, np.nan], 0.60)
        dnpa = dnp.quantile([1, 8, 27, -27, 0, 5, dnp.nan], 0.60)
        assert_array_equal(dnpa, npa)

    def test_function_math_quantile_array(self):
        npa = np.quantile(np.array([1, 8, 27, -27, 0, 5, np.nan]), 0.60)
        dnpa = dnp.quantile(dnp.array([1, 8, 27, -27, 0, 5, dnp.nan]), 0.60)
        assert_array_equal(dnpa, npa)

    def test_function_math_quantile_series(self):
        ps = pd.Series([-1, 8, 27, -27, 0, 5, np.nan])
        os = orca.Series(ps)
        self.assertEqual(dnp.isnan(dnp.quantile(os, 0.30)), True)
        self.assertEqual(np.isnan(np.quantile(ps, 0.30)), True)

        ps = pd.Series([-1, 8, 27, -27, 0, 5])
        os = orca.Series(ps)
        self.assertEqual(dnp.quantile(os, 0.60), np.quantile(ps, 0.60))

    def test_function_math_quantile_dataframe(self):
        pdf = pd.DataFrame({"cola": [-1, 8, 27, -27, 0, 5, np.nan, 1.5, 1.7, 2.0, np.nan],
                            "colb": [-1, 8, 27, -27, 0, 5, np.nan, 1.5, 1.7, np.nan, 2.0]})
        odf = orca.DataFrame(pdf)
        self.assertEqual(dnp.isnan(dnp.quantile(odf, 0.30)), True)
        self.assertEqual(np.isnan(np.quantile(pdf, 0.30)), True)

        pdf = pd.DataFrame({"cola": [-1, 8, 27, -27, 0, 5, 1.5, 1.7, 2.0],
                            "colb": [-1, 8, 27, -27, 0, 5, 1.5, 1.7, 2.0]})
        odf = orca.DataFrame(pdf)
        self.assertEqual(dnp.quantile(odf, 0.60), np.quantile(pdf, 0.60))

    def test_function_math_quantile_param_q_list_X_scalar(self):
        assert_array_equal(dnp.quantile(0.5, [0.60, 0.45, 0.20]), np.quantile(0.5, [0.60, 0.45, 0.20]))
        assert_array_equal(dnp.quantile(1, [0.60, 0.45, 0.20]), np.quantile(1, [0.60, 0.45, 0.20]))
        assert_array_equal(dnp.quantile(-1, [0.60, 0.45, 0.20]), np.quantile(-1, [0.60, 0.45, 0.20]))
        assert_array_equal(dnp.quantile(0, [0.60, 0.45, 0.20]), np.quantile(0, [0.60, 0.45, 0.20]))
        assert_array_equal(dnp.isnan(dnp.quantile(dnp.nan, [0.60, 0.45, 0.20])), True)
        assert_array_equal(np.isnan(np.quantile(np.nan, [0.60, 0.45, 0.20])), True)

    def test_function_math_quantile_param_q_list_X_list(self):
        npa = np.quantile([1, 8, 27, -27, 0, 5, np.nan], [0.60, 0.45, 0.20])
        dnpa = dnp.quantile([1, 8, 27, -27, 0, 5, dnp.nan], [0.60, 0.45, 0.20])
        assert_array_equal(dnpa, npa)

    def test_function_math_quantile_param_q_list_X_array(self):
        npa = np.quantile(np.array([1, 8, 27, -27, 0, 5, np.nan]), [0.60, 0.45, 0.20])
        dnpa = dnp.quantile(dnp.array([1, 8, 27, -27, 0, 5, dnp.nan]), [0.60, 0.45, 0.20])
        assert_array_equal(dnpa, npa)

    def test_function_math_quantile_param_q_list_X_series(self):
        ps = pd.Series([-1, 8, 27, -27, 0, 5, np.nan])
        os = orca.Series(ps)
        assert_array_equal(dnp.isnan(dnp.quantile(os, [0.60, 0.45, 0.20])), True)
        assert_array_equal(np.isnan(np.quantile(ps, [0.60, 0.45, 0.20])), True)

        ps = pd.Series([-1, 8, 27, -27, 0, 5])
        os = orca.Series(ps)
        assert_array_equal(dnp.quantile(os, [0.60, 0.45, 0.20]), np.quantile(ps, [0.60, 0.45, 0.20]))

    def test_function_math_quantile_param_q_list_X_dataframe(self):
        pdf = pd.DataFrame({"cola": [-1, 8, 27, -27, 0, 5, np.nan, 1.5, 1.7, 2.0, np.nan],
                            "colb": [-1, 8, 27, -27, 0, 5, np.nan, 1.5, 1.7, np.nan, 2.0]})
        odf = orca.DataFrame(pdf)
        assert_array_equal(dnp.isnan(dnp.quantile(odf, [0.60, 0.45, 0.20])), True)
        assert_array_equal(np.isnan(np.quantile(pdf, [0.60, 0.45, 0.20])), True)

        pdf = pd.DataFrame({"cola": [-1, 8, 27, -27, 0, 5, 1.5, 1.7, 2.0],
                            "colb": [-1, 8, 27, -27, 0, 5, 1.5, 1.7, 2.0]})
        odf = orca.DataFrame(pdf)
        assert_array_equal(dnp.quantile(odf, [0.60, 0.45, 0.20]), np.quantile(pdf, [0.60, 0.45, 0.20]))


if __name__ == '__main__':
    unittest.main()
