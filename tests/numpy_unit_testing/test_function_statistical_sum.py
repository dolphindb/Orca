import unittest
from setup.settings import *
from numpy.testing import *
from pandas.util.testing import *
import numpy as np
import dolphindb_numpy as dnp
import pandas as pd
import orca


class FunctionSumTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_function_math_sum_scalar(self):
        self.assertEqual(dnp.sum(0.5), np.sum(0.5))
        self.assertEqual(dnp.sum(1), np.sum(1))
        self.assertEqual(dnp.sum(-1), np.sum(-1))
        self.assertEqual(dnp.sum(0), np.sum(0))
        self.assertEqual(dnp.isnan(dnp.sum(dnp.nan)), True)
        self.assertEqual(np.isnan(np.sum(np.nan)), True)

    def test_function_math_sum_list(self):
        npa = np.sum([1, 8, 27, -27, 0, 5, np.nan])
        dnpa = dnp.sum([1, 8, 27, -27, 0, 5, dnp.nan])
        assert_array_equal(dnpa, npa)

    def test_function_math_sum_array(self):
        npa = np.sum(np.array([1, 8, 27, -27, 0, 5, np.nan]))
        dnpa = dnp.sum(dnp.array([1, 8, 27, -27, 0, 5, dnp.nan]))
        assert_array_equal(dnpa, npa)

    def test_function_math_sum_series(self):
        ps = pd.Series([-1, 8, 27, -27, 0, 5, np.nan])
        os = orca.Series(ps)
        self.assertEqual(dnp.sum(os), np.sum(ps))

        ps = pd.Series([-1, 8, 27, -27, 0, 5])
        os = orca.Series(ps)
        self.assertEqual(dnp.sum(os), np.sum(ps))

    def test_function_math_sum_dataframe(self):
        pdf = pd.DataFrame({"cola": [-1, 8, 27, -27, 0, 5, np.nan, 1.5, 1.7, 2.0, np.nan],
                            "colb": [-1, 8, 27, -27, 0, 5, np.nan, 1.5, 1.7, np.nan, 2.0]})
        odf = orca.DataFrame(pdf)
        assert_series_equal(dnp.sum(odf).to_pandas(), np.sum(pdf))

        pdf = pd.DataFrame({"cola": [-1, 8, 27, -27, 0, 5, 1.5, 1.7, 2.0],
                            "colb": [-1, 8, 27, -27, 0, 5, 1.5, 1.7, 2.0]})
        odf = orca.DataFrame(pdf)
        assert_series_equal(dnp.sum(odf).to_pandas(), np.sum(pdf))


if __name__ == '__main__':
    unittest.main()
