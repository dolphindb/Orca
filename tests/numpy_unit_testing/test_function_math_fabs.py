import unittest
from setup.settings import *
from numpy.testing import *
from pandas.util.testing import *
import numpy as np
import dolphindb_numpy as dnp
import pandas as pd
import orca


class FunctionFabsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_function_math_fabs_scalar(self):
        self.assertEqual(dnp.fabs(1e-10), np.fabs(1e-10))
        self.assertEqual(dnp.fabs(0.5), np.fabs(0.5))
        self.assertEqual(dnp.fabs(1), np.fabs(1))
        self.assertEqual(dnp.fabs(-1), np.fabs(-1))
        self.assertEqual(dnp.fabs(0), np.fabs(0))
        self.assertEqual(dnp.isnan(dnp.fabs(dnp.nan)), True)
        self.assertEqual(np.isnan(np.fabs(np.nan)), True)

    def test_function_math_fabs_list(self):
        npa = np.fabs([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0, np.nan])
        dnpa = dnp.fabs([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0, dnp.nan])
        assert_array_equal(dnpa, npa)

    def test_function_math_fabs_array(self):
        npa = np.fabs(np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0, np.nan]))
        dnpa = dnp.fabs(dnp.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0, dnp.nan]))
        assert_array_equal(dnpa, npa)

    def test_function_math_fabs_series(self):
        ps = pd.Series([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0, np.nan])
        os = orca.Series(ps)
        # TODO：BUG
        # assert_series_equal(dnp.fabs(os).to_pandas(), np.fabs(ps))

    def test_function_math_fabs_dataframe(self):
        pdf = pd.DataFrame({"cola": [-1.7, -1.5, -0.2, np.nan, 0.2, 1.5, 1.7, 2.0, np.nan],
                            "colb": [-1.7, -1.5, -0.2, np.nan, 0.2, 1.5, 1.7, np.nan, 2.0]})
        odf = orca.DataFrame(pdf)
        # TODO：BUG
        # assert_frame_equal(dnp.fabs(odf).to_pandas(), np.fabs(pdf))


if __name__ == '__main__':
    unittest.main()
