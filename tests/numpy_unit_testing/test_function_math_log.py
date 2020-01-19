import unittest
from setup.settings import *
from numpy.testing import *
from pandas.util.testing import *
import numpy as np
import dolphindb_numpy as dnp
import pandas as pd
import orca


class FunctionLogTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_function_math_log_scalar(self):
        self.assertEqual(dnp.log(1.2 + 1j), np.log(1.2 + 1j))
        self.assertEqual(dnp.log(1e-10), np.log(1e-10))
        self.assertEqual(dnp.log(0.5), np.log(0.5))
        self.assertEqual(dnp.log(1), np.log(1))
        self.assertEqual(dnp.log(0), np.log(0))
        self.assertEqual(dnp.isnan(dnp.log(-1)), True)
        self.assertEqual(np.isnan(np.log(-1)), True)
        self.assertEqual(dnp.isnan(dnp.log(dnp.nan)), True)
        self.assertEqual(np.isnan(np.log(np.nan)), True)

    def test_function_math_log_list(self):
        npa = np.log([1, np.e, np.e**2, 0, 8, 2.677, np.nan])
        dnpa = dnp.log([1, np.e, np.e**2, 0, 8, 2.677, dnp.nan])
        assert_array_equal(dnpa, npa)

    def test_function_math_log_array(self):
        npa = np.log(np.array([1, np.e, np.e**2, 0, 8, 2.677, np.nan]))
        dnpa = dnp.log(dnp.array([1, np.e, np.e**2, 0, 8, 2.677, dnp.nan]))
        assert_array_equal(dnpa, npa)

    def test_function_math_log_series(self):
        ps = pd.Series([1, np.e, np.e**2, 0, 8, 2.677, np.nan])
        os = orca.Series(ps)
        # TODO: log bug
        # assert_series_equal(dnp.log(os).to_pandas(), np.log(ps))

    def test_function_math_log_dataframe(self):
        pdf = pd.DataFrame({"cola": [1, np.e, np.e**2, np.nan, 0, 8, 2.677, 2.0, np.nan],
                            "colb": [1, np.e, np.e**2, np.nan, 0, 8, 2.677, np.nan, 2.0]})
        odf = orca.DataFrame(pdf)
        # TODO: log bug
        # assert_frame_equal(dnp.log(odf).to_pandas(), np.log(pdf))


if __name__ == '__main__':
    unittest.main()
