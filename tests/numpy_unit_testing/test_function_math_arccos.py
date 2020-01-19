import unittest
from setup.settings import *
from numpy.testing import *
from pandas.util.testing import *
import numpy as np
import dolphindb_numpy as dnp
import pandas as pd
import orca


class FunctionArccosTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_function_math_arccos_scalar(self):
        self.assertEqual(dnp.arccos(1.2 + 1j), np.arccos(1.2 + 1j))
        self.assertEqual(dnp.arccos(0.5), np.arccos(0.5))
        self.assertEqual(dnp.arccos(1), np.arccos(1))
        self.assertEqual(dnp.arccos(-1), np.arccos(-1))
        self.assertEqual(dnp.arccos(0), np.arccos(0))
        self.assertEqual(dnp.isnan(dnp.arccos(dnp.nan)), True)
        self.assertEqual(np.isnan(np.arccos(np.nan)), True)

    def test_function_math_arccos_list(self):
        npa = np.arccos([-1, 0.5, 1.2 + 1j, np.nan])
        dnpa = dnp.arccos([-1, 0.5, 1.2 + 1j, dnp.nan])
        assert_array_equal(dnpa, npa)

    def test_function_math_arccos_array(self):
        npa = np.arccos(np.array([-1, 0.5, 1.2 + 1j, np.nan]))
        dnpa = dnp.arccos(dnp.array([-1, 0.5, 1.2 + 1j, dnp.nan]))
        assert_array_equal(dnpa, npa)

    def test_function_math_arccos_series(self):
        ps = pd.Series([-1, 0.5, np.nan])
        os = orca.Series(ps)
        assert_series_equal(dnp.arccos(os).to_pandas(), np.arccos(ps))

    def test_function_math_arccos_dataframe(self):
        pdf = pd.DataFrame({"cola": [-1.7, -1.5, -0.2, np.nan, 1.5, 1.7, 2.0, np.nan],
                            "colb": [-1.7, -1.5, -0.2, np.nan, 1.5, 1.7, np.nan, 2.0]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(dnp.arccos(odf).to_pandas(), np.arccos(pdf))


if __name__ == '__main__':
    unittest.main()
