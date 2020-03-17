import unittest
from setup.settings import *
from numpy.testing import *
import numpy as np
import dolphindb_numpy as dnp
import pandas as pd
import orca


class FunctionMedianTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_function_math_median_scalar(self):
        self.assertEqual(dnp.median(0.5), np.median(0.5))
        self.assertEqual(dnp.median(1), np.median(1))
        self.assertEqual(dnp.median(-1), np.median(-1))
        self.assertEqual(dnp.median(0), np.median(0))
        self.assertEqual(dnp.isnan(dnp.median(dnp.nan)), True)
        self.assertEqual(np.isnan(np.median(np.nan)), True)

    def test_function_math_median_list(self):
        npa = np.median([1, 8, 27, -27, 0, 5, np.nan])
        dnpa = dnp.median([1, 8, 27, -27, 0, 5, dnp.nan])
        assert_array_equal(dnpa, npa)

    def test_function_math_median_array(self):
        npa = np.median(np.array([1, 8, 27, -27, 0, 5, np.nan]))
        dnpa = dnp.median(dnp.array([1, 8, 27, -27, 0, 5, dnp.nan]))
        assert_array_equal(dnpa, npa)

    def test_function_math_median_series(self):
        ps = pd.Series([-1, 8, 27, -27, 0, 5, np.nan])
        os = orca.Series(ps)
        self.assertEqual(dnp.isnan(dnp.median(os)), True)
        self.assertEqual(np.isnan(np.median(ps)), True)

        ps = pd.Series([-1, 8, 27, -27, 0, 5])
        os = orca.Series(ps)
        self.assertEqual(dnp.median(os), np.median(ps))

    def test_function_math_median_dataframe(self):
        pdf = pd.DataFrame({"cola": [-1, 8, 27, -27, 0, 5, np.nan, 1.5, 1.7, 2.0, np.nan],
                            "colb": [-1, 8, 27, -27, 0, 5, np.nan, 1.5, 1.7, np.nan, 2.0]})
        odf = orca.DataFrame(pdf)
        self.assertEqual(dnp.isnan(dnp.median(odf)), True)
        self.assertEqual(np.isnan(np.median(pdf)), True)

        pdf = pd.DataFrame({"cola": [-1, 8, 27, -27, 0, 5, 1.5, 1.7, 2.0],
                            "colb": [-1, 8, 27, -27, 0, 5, 1.5, 1.7, 2.0]})
        odf = orca.DataFrame(pdf)
        self.assertEqual(dnp.median(odf), np.median(pdf))


if __name__ == '__main__':
    unittest.main()
