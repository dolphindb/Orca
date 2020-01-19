import unittest
from setup.settings import *
from numpy.testing import *
from pandas.util.testing import *
from math import pi
import numpy as np
import dolphindb_numpy as dnp
import pandas as pd
import orca


class FunctionTanTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_function_math_tan_scalar(self):
        self.assertEqual(dnp.tan(1.2 + 1j), np.tan(1.2 + 1j))
        self.assertEqual(dnp.tan(dnp.pi/2.), np.tan(np.pi/2.))
        self.assertEqual(dnp.tan(0.5), np.tan(0.5))
        self.assertEqual(dnp.tan(1), np.tan(1))
        self.assertEqual(dnp.tan(-1), np.tan(-1))
        self.assertEqual(dnp.tan(0), np.tan(0))
        self.assertEqual(dnp.isnan(dnp.tan(dnp.nan)), True)
        self.assertEqual(np.isnan(np.tan(np.nan)), True)

    def test_function_math_tan_list(self):
        npa = np.tan(list(np.array([0., 30., 45., 60., 90., np.nan]) * np.pi / 180.))
        dnpa = dnp.tan(list(dnp.array([0., 30., 45., 60., 90., dnp.nan]) * dnp.pi / 180.))
        assert_array_equal(dnpa, npa)

    def test_function_math_tan_array(self):
        npa = np.tan([-pi, pi/2, pi, 0, np.inf, np.nan])
        dnpa = dnp.tan([-pi, pi/2, pi, 0, dnp.inf, dnp.nan])
        assert_array_equal(dnpa, npa)

    def test_function_math_tan_series(self):
        ps = pd.Series(np.array([0., 30., 45., 60., 90., np.nan]) * np.pi / 180.)
        os = orca.Series(ps)
        assert_series_equal(dnp.tan(os).to_pandas(), np.tan(ps))

    def test_function_math_tan_dataframe(self):
        pdf = pd.DataFrame({"cola": np.array([0., 30., np.nan, 45., 60., 90., np.nan]) * np.pi / 180.,
                            "colb": np.array([0., 30., np.nan, 45., 60., np.nan, 90.]) * np.pi / 180.})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(dnp.tan(odf).to_pandas(), np.tan(pdf))


if __name__ == '__main__':
    unittest.main()
