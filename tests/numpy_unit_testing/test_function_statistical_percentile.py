import unittest
from setup.settings import *
from numpy.testing import *
import numpy as np
import dolphindb_numpy as dnp
import pandas as pd
import orca


class FunctionPercentileTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_function_math_percentile_scalar(self):
        self.assertEqual(dnp.percentile(0.5, 30), np.percentile(0.5, 30))
        self.assertEqual(dnp.percentile(1, 30), np.percentile(1, 30))
        self.assertEqual(dnp.percentile(-1, 30), np.percentile(-1, 30))
        self.assertEqual(dnp.percentile(0, 30), np.percentile(0, 30))
        self.assertEqual(dnp.isnan(dnp.percentile(dnp.nan, 30)), True)
        self.assertEqual(np.isnan(np.percentile(np.nan, 30)), True)

    def test_function_math_percentile_list(self):
        npa = np.percentile([1, 8, 27, -27, 0, 5, np.nan], 60)
        dnpa = dnp.percentile([1, 8, 27, -27, 0, 5, dnp.nan], 60)
        assert_array_equal(dnpa, npa)

    def test_function_math_percentile_array(self):
        npa = np.percentile(np.array([1, 8, 27, -27, 0, 5, np.nan]), 60)
        dnpa = dnp.percentile(dnp.array([1, 8, 27, -27, 0, 5, dnp.nan]), 60)
        assert_array_equal(dnpa, npa)

    def test_function_math_percentile_series(self):
        ps = pd.Series([-1, 8, 27, -27, 0, 5, np.nan])
        os = orca.Series(ps)
        self.assertEqual(dnp.isnan(dnp.percentile(os, 30)), True)
        self.assertEqual(np.isnan(np.percentile(ps, 30)), True)

        ps = pd.Series([-1, 8, 27, -27, 0, 5])
        os = orca.Series(ps)
        self.assertEqual(dnp.percentile(os, 60), np.percentile(ps, 60))

    def test_function_math_percentile_dataframe(self):
        pdf = pd.DataFrame({"cola": [-1, 8, 27, -27, 0, 5, np.nan, 1.5, 1.7, 2.0, np.nan],
                            "colb": [-1, 8, 27, -27, 0, 5, np.nan, 1.5, 1.7, np.nan, 2.0]})
        odf = orca.DataFrame(pdf)
        self.assertEqual(dnp.isnan(dnp.percentile(odf, 30)), True)
        self.assertEqual(np.isnan(np.percentile(pdf, 30)), True)

        pdf = pd.DataFrame({"cola": [-1, 8, 27, -27, 0, 5, 1.5, 1.7, 2.0],
                            "colb": [-1, 8, 27, -27, 0, 5, 1.5, 1.7, 2.0]})
        odf = orca.DataFrame(pdf)
        self.assertEqual(dnp.percentile(odf, 60), np.percentile(pdf, 60))

    def test_function_math_percentile_param_q_list_X_scalar(self):
        assert_array_equal(dnp.percentile(0.5, [60, 45, 20]), np.percentile(0.5, [60, 45, 20]))
        assert_array_equal(dnp.percentile(1, [60, 45, 20]), np.percentile(1, [60, 45, 20]))
        assert_array_equal(dnp.percentile(-1, [60, 45, 20]), np.percentile(-1, [60, 45, 20]))
        assert_array_equal(dnp.percentile(0, [60, 45, 20]), np.percentile(0, [60, 45, 20]))
        assert_array_equal(dnp.isnan(dnp.percentile(dnp.nan, [60, 45, 20])), True)
        assert_array_equal(np.isnan(np.percentile(np.nan, [60, 45, 20])), True)

    def test_function_math_percentile_param_q_list_X_list(self):
        npa = np.percentile([1, 8, 27, -27, 0, 5, np.nan], [60, 45, 20])
        dnpa = dnp.percentile([1, 8, 27, -27, 0, 5, dnp.nan], [60, 45, 20])
        assert_array_equal(dnpa, npa)

    def test_function_math_percentile_param_q_list_X_array(self):
        npa = np.percentile(np.array([1, 8, 27, -27, 0, 5, np.nan]), [60, 45, 20])
        dnpa = dnp.percentile(dnp.array([1, 8, 27, -27, 0, 5, dnp.nan]), [60, 45, 20])
        assert_array_equal(dnpa, npa)

    def test_function_math_percentile_param_q_list_X_series(self):
        ps = pd.Series([-1, 8, 27, -27, 0, 5, np.nan])
        os = orca.Series(ps)
        assert_array_equal(dnp.isnan(dnp.percentile(os, [60, 45, 20])), True)
        assert_array_equal(np.isnan(np.percentile(ps, [60, 45, 20])), True)

        ps = pd.Series([-1, 8, 27, -27, 0, 5])
        os = orca.Series(ps)
        assert_array_equal(dnp.percentile(os, [60, 45, 20]), np.percentile(ps, [60, 45, 20]))

    def test_function_math_percentile_param_q_list_X_dataframe(self):
        pdf = pd.DataFrame({"cola": [-1, 8, 27, -27, 0, 5, np.nan, 1.5, 1.7, 2.0, np.nan],
                            "colb": [-1, 8, 27, -27, 0, 5, np.nan, 1.5, 1.7, np.nan, 2.0]})
        odf = orca.DataFrame(pdf)
        assert_array_equal(dnp.isnan(dnp.percentile(odf, [60, 45, 20])), True)
        assert_array_equal(np.isnan(np.percentile(pdf, [60, 45, 20])), True)

        pdf = pd.DataFrame({"cola": [-1, 8, 27, -27, 0, 5, 1.5, 1.7, 2.0],
                            "colb": [-1, 8, 27, -27, 0, 5, 1.5, 1.7, 2.0]})
        odf = orca.DataFrame(pdf)
        assert_array_equal(dnp.percentile(odf, [60, 45, 20]), np.percentile(pdf, [60, 45, 20]))


if __name__ == '__main__':
    unittest.main()
