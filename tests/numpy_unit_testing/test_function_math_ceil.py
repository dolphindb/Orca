import unittest
from setup.settings import *
from numpy.testing import *
from pandas.util.testing import *
import numpy as np
import dolphindb_numpy as dnp
import pandas as pd
import orca


class FunctionCeilTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_function_math_ceil_scalar(self):
        self.assertEqual(dnp.ceil(0.5), np.ceil(0.5))
        self.assertEqual(dnp.ceil(1), np.ceil(1))
        self.assertEqual(dnp.ceil(-1), np.ceil(-1))
        self.assertEqual(dnp.ceil(0), np.ceil(0))
        self.assertEqual(dnp.isnan(dnp.ceil(dnp.nan)), True)
        self.assertEqual(np.isnan(np.ceil(np.nan)), True)

    def test_function_math_ceil_list(self):
        npa = np.ceil([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0, np.nan])
        dnpa = dnp.ceil([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0, dnp.nan])
        assert_array_equal(dnpa, npa)

    def test_function_math_ceil_array(self):
        npa = np.ceil(np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0, np.nan]))
        dnpa = dnp.ceil(dnp.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0, dnp.nan]))
        assert_array_equal(dnpa, npa)

    def test_function_math_ceil_series(self):
        ps = pd.Series([-1, 8, 27, -27, 0, 5, np.nan])
        os = orca.Series(ps)
        assert_series_equal(dnp.ceil(os).to_pandas(), np.ceil(ps))

    def test_function_math_ceil_dataframe(self):
        pdf = pd.DataFrame({"cola": [-1, 8, 27, -27, 0, 5, np.nan, 1.5, 1.7, 2.0, np.nan],
                            "colb": [-1, 8, 27, -27, 0, 5, np.nan, 1.5, 1.7, np.nan, 2.0]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(dnp.ceil(odf).to_pandas(), np.ceil(pdf))


if __name__ == '__main__':
    unittest.main()
