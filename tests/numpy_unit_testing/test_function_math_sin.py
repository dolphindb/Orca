import unittest
from setup.settings import *
from numpy.testing import *
from pandas.util.testing import *
import numpy as np
import dolphindb_numpy as dnp
import pandas as pd
import orca


class FunctionSinTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_function_math_sin_scalar(self):
        self.assertEqual(dnp.sin(1.2 + 1j), np.sin(1.2 + 1j))
        self.assertEqual(dnp.sin(dnp.pi/2.), np.sin(np.pi/2.))
        self.assertEqual(dnp.sin(0.5), np.sin(0.5))
        self.assertEqual(dnp.sin(1), np.sin(1))
        self.assertEqual(dnp.sin(-1), np.sin(-1))
        self.assertEqual(dnp.sin(0), np.sin(0))
        self.assertEqual(dnp.isnan(dnp.sin(dnp.nan)), True)
        self.assertEqual(np.isnan(np.sin(np.nan)), True)

    def test_function_math_sin_list(self):
        npa = np.sin(list(np.array([0., 30., 45., 60., 90., np.nan]) * np.pi / 180.))
        dnpa = dnp.sin(list(dnp.array([0., 30., 45., 60., 90., dnp.nan]) * dnp.pi / 180.))
        assert_array_equal(dnpa, npa)

    def test_function_math_sin_array(self):
        npa = np.sin(np.array([0., 30., 45., 60., 90., np.nan]) * np.pi / 180.)
        dnpa = dnp.sin(dnp.array([0., 30., 45., 60., 90., dnp.nan]) * dnp.pi / 180.)
        assert_array_equal(dnpa, npa)

    def test_function_math_sin_series(self):
        ps = pd.Series(np.array([0., 30., 45., 60., 90., np.nan]) * np.pi / 180.)
        os = orca.Series(ps)
        assert_series_equal(dnp.sin(os).to_pandas(), np.sin(ps))

    def test_function_math_sin_dataframe(self):
        pdf = pd.DataFrame({"cola": np.array([0., 30., np.nan, 45., 60., 90., np.nan]) * np.pi / 180.,
                            "colb": np.array([0., 30., np.nan, 45., 60., np.nan, 90.]) * np.pi / 180.})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(dnp.sin(odf).to_pandas(), np.sin(pdf))


if __name__ == '__main__':
    unittest.main()
