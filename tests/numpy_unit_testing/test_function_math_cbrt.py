import unittest
from setup.settings import *
from numpy.testing import *
from pandas.util.testing import *
import numpy as np
import dolphindb_numpy as dnp
import pandas as pd
import orca


class FunctionCbrtTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_function_math_cbrt_scalar(self):
        self.assertEqual(dnp.cbrt(0.5), np.cbrt(0.5))
        self.assertEqual(dnp.cbrt(1), np.cbrt(1))
        self.assertEqual(dnp.cbrt(-1), np.cbrt(-1))
        self.assertEqual(dnp.cbrt(0), np.cbrt(0))
        self.assertEqual(dnp.isnan(dnp.cbrt(dnp.nan)), True)
        self.assertEqual(np.isnan(np.cbrt(np.nan)), True)

    def test_function_math_cbrt_list(self):
        npa = np.cbrt([1, 8, 27, -27, 0, 5, np.nan])
        dnpa = dnp.cbrt([1, 8, 27, -27, 0, 5, dnp.nan])
        assert_array_equal(dnpa, npa)

    def test_function_math_cbrt_array(self):
        npa = np.cbrt(np.array([1, 8, 27, -27, 0, 5, np.nan]))
        dnpa = dnp.cbrt(dnp.array([1, 8, 27, -27, 0, 5, dnp.nan]))
        assert_array_equal(dnpa, npa)

    def test_function_math_cbrt_series(self):
        ps = pd.Series([-1, 8, 27, -27, 0, 5, np.nan])
        os = orca.Series(ps)
        # TODO： BUG
        # assert_series_equal(dnp.cbrt(os).to_pandas(), np.cbrt(ps))

    def test_function_math_cbrt_dataframe(self):
        pdf = pd.DataFrame({"cola": [-1, 8, 27, -27, 0, 5, np.nan, 1.5, 1.7, 2.0, np.nan],
                            "colb": [-1, 8, 27, -27, 0, 5, np.nan, 1.5, 1.7, np.nan, 2.0]})
        odf = orca.DataFrame(pdf)
        print(dnp.cbrt(odf))
        # TODO： BUG
        assert_frame_equal(dnp.cbrt(pdf), np.cbrt(pdf))


if __name__ == '__main__':
    unittest.main()
