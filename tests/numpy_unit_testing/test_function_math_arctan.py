import unittest
from setup.settings import *
from numpy.testing import *
from pandas.util.testing import *
import numpy as np
import dolphindb_numpy as dnp
import pandas as pd
import orca


class FunctionArctanTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_function_math_arctan_scalar(self):
        self.assertEqual(dnp.arctan(1.2 + 1j), np.arctan(1.2 + 1j))
        self.assertEqual(dnp.arctan(0.5), np.arctan(0.5))
        self.assertEqual(dnp.arctan(1), np.arctan(1))
        self.assertEqual(dnp.arctan(-1), np.arctan(-1))
        self.assertEqual(dnp.arctan(0), np.arctan(0))
        self.assertEqual(dnp.isnan(dnp.arctan(dnp.nan)), True)
        self.assertEqual(np.isnan(np.arctan(np.nan)), True)

    def test_function_math_arctan_list(self):
        npa = np.arctan([-1, 0.5, 1.2 + 1j, np.nan])
        dnpa = dnp.arctan([-1, 0.5, 1.2 + 1j, dnp.nan])
        assert_array_equal(dnpa, npa)

    def test_function_math_arctan_array(self):
        npa = np.arctan(np.array([-1, 0.5, 1.2 + 1j, np.nan]))
        dnpa = dnp.arctan(dnp.array([-1, 0.5, 1.2 + 1j, dnp.nan]))
        assert_array_equal(dnpa, npa)

    def test_function_math_arctan_series(self):
        ps = pd.Series([-1, 0.5, np.nan])
        os = orca.Series(ps)
        assert_series_equal(dnp.arctan(os).to_pandas(), np.arctan(ps))

    def test_function_math_arctan_dataframe(self):
        pdf = pd.DataFrame({"cola": [-1.7, -1.5, -0.2, np.nan, 1.5, 1.7, 2.0, np.nan],
                            "colb": [-1.7, -1.5, -0.2, np.nan, 1.5, 1.7, np.nan, 2.0]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(dnp.arctan(odf).to_pandas(), np.arctan(pdf))


if __name__ == '__main__':
    unittest.main()
