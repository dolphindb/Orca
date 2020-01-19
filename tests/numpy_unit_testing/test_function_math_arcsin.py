import unittest
from setup.settings import *
from numpy.testing import *
from pandas.util.testing import *
import numpy as np
import dolphindb_numpy as dnp
import pandas as pd
import orca


class FunctionArcsinTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_function_math_arcsin_scalar(self):
        self.assertEqual(dnp.arcsin(1.2 + 1j), np.arcsin(1.2 + 1j))
        self.assertEqual(dnp.arcsin(0.5), np.arcsin(0.5))
        self.assertEqual(dnp.arcsin(1), np.arcsin(1))
        self.assertEqual(dnp.arcsin(-1), np.arcsin(-1))
        self.assertEqual(dnp.arcsin(0), np.arcsin(0))
        self.assertEqual(dnp.isnan(dnp.arcsin(dnp.nan)), True)
        self.assertEqual(np.isnan(np.arcsin(np.nan)), True)

    def test_function_math_arcsin_list(self):
        npa = np.arcsin([-1, 0.5, 1.2 + 1j, np.nan])
        dnpa = dnp.arcsin([-1, 0.5, 1.2 + 1j, dnp.nan])
        assert_array_equal(dnpa, npa)

    def test_function_math_arcsin_array(self):
        npa = np.arcsin(np.array([-1, 0.5, 1.2 + 1j, np.nan]))
        dnpa = dnp.arcsin(dnp.array([-1, 0.5, 1.2 + 1j, dnp.nan]))
        assert_array_equal(dnpa, npa)

    def test_function_math_arcsin_series(self):
        ps = pd.Series([-1, 0.5, np.nan])
        os = orca.Series(ps)
        assert_series_equal(dnp.arcsin(os).to_pandas(), np.arcsin(ps))

    def test_function_math_arcsin_dataframe(self):
        pdf = pd.DataFrame({"cola": [-1.7, -1.5, -0.2, np.nan, 1.5, 1.7, 2.0, np.nan],
                            "colb": [-1.7, -1.5, -0.2, np.nan, 1.5, 1.7, np.nan, 2.0]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(dnp.arcsin(odf).to_pandas(), np.arcsin(pdf))


if __name__ == '__main__':
    unittest.main()
