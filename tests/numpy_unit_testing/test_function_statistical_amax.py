import unittest
from setup.settings import *
from numpy.testing import *
from pandas.util.testing import *
import numpy as np
import dolphindb_numpy as dnp
import pandas as pd
import orca


class FunctionAmaxTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_function_math_amax_scalar(self):
        self.assertEqual(dnp.amax(0.5), np.amax(0.5))
        self.assertEqual(dnp.amax(1), np.amax(1))
        self.assertEqual(dnp.amax(-1), np.amax(-1))
        self.assertEqual(dnp.amax(0), np.amax(0))
        self.assertEqual(dnp.isnan(dnp.amax(dnp.nan)), True)
        self.assertEqual(np.isnan(np.amax(np.nan)), True)

    def test_function_math_amax_list(self):
        npa = np.amax([1, 8, 27, -27, 0, 5, np.nan])
        dnpa = dnp.amax([1, 8, 27, -27, 0, 5, dnp.nan])
        assert_array_equal(dnpa, npa)

    def test_function_math_amax_array(self):
        npa = np.amax(np.array([1, 8, 27, -27, 0, 5, np.nan]))
        dnpa = dnp.amax(dnp.array([1, 8, 27, -27, 0, 5, dnp.nan]))
        assert_array_equal(dnpa, npa)

    def test_function_math_amax_series(self):
        ps = pd.Series([-1, 8, 27, -27, 0, 5, np.nan])
        os = orca.Series(ps)
        self.assertEqual(dnp.amax(os), np.amax(ps))

        ps = pd.Series([-1, 8, 27, -27, 0, 5])
        os = orca.Series(ps)
        self.assertEqual(dnp.amax(os), np.amax(ps))

    def test_function_math_amax_dataframe(self):
        pdf = pd.DataFrame({"cola": [-1, 8, 27, -27, 0, 5, np.nan, 1.5, 1.7, 2.0, np.nan],
                            "colb": [-1, 8, 27, -27, 0, 5, np.nan, 1.5, 1.7, np.nan, 2.0]})
        odf = orca.DataFrame(pdf)
        assert_series_equal(dnp.amax(odf).to_pandas(), np.amax(pdf))

        pdf = pd.DataFrame({"cola": [-1, 8, 27, -27, 0, 5, 1.5, 1.7, 2.0],
                            "colb": [-1, 8, 27, -27, 0, 5, 1.5, 1.7, 2.0]})
        odf = orca.DataFrame(pdf)
        assert_series_equal(dnp.amax(odf).to_pandas(), np.amax(pdf))


if __name__ == '__main__':
    unittest.main()
