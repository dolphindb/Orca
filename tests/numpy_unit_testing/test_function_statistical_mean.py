import unittest
from setup.settings import *
from numpy.testing import *
from pandas.util.testing import *
import numpy as np
import dolphindb_numpy as dnp
import pandas as pd
import orca


class FunctionMeanTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_function_math_mean_scalar(self):
        self.assertEqual(dnp.mean(0.5), np.mean(0.5))
        self.assertEqual(dnp.mean(1), np.mean(1))
        self.assertEqual(dnp.mean(-1), np.mean(-1))
        self.assertEqual(dnp.mean(0), np.mean(0))
        self.assertEqual(dnp.isnan(dnp.mean(dnp.nan)), True)
        self.assertEqual(np.isnan(np.mean(np.nan)), True)

    def test_function_math_mean_list(self):
        npa = np.mean([1, 8, 27, -27, 0, 5, np.nan])
        dnpa = dnp.mean([1, 8, 27, -27, 0, 5, dnp.nan])
        assert_array_equal(dnpa, npa)

    def test_function_math_mean_array(self):
        npa = np.mean(np.array([1, 8, 27, -27, 0, 5, np.nan]))
        dnpa = dnp.mean(dnp.array([1, 8, 27, -27, 0, 5, dnp.nan]))
        assert_array_equal(dnpa, npa)

    def test_function_math_mean_series(self):
        ps = pd.Series([-1, 8, 27, -27, 0, 5, np.nan])
        os = orca.Series(ps)
        self.assertEqual(dnp.mean(os), np.mean(ps))

        ps = pd.Series([-1, 8, 27, -27, 0, 5])
        os = orca.Series(ps)
        self.assertEqual(dnp.mean(os), np.mean(ps))

    def test_function_math_mean_dataframe(self):
        pdf = pd.DataFrame({"cola": [-1, 8, 27, -27, 0, 5, np.nan, 1.5, 1.7, 2.0, np.nan],
                            "colb": [-1, 8, 27, -27, 0, 5, np.nan, 1.5, 1.7, np.nan, 2.0]})
        odf = orca.DataFrame(pdf)
        assert_series_equal(dnp.mean(odf).to_pandas(), np.mean(pdf))

        pdf = pd.DataFrame({"cola": [-1, 8, 27, -27, 0, 5, 1.5, 1.7, 2.0],
                            "colb": [-1, 8, 27, -27, 0, 5, 1.5, 1.7, 2.0]})
        odf = orca.DataFrame(pdf)
        assert_series_equal(dnp.mean(odf).to_pandas(), np.mean(pdf))


if __name__ == '__main__':
    unittest.main()
