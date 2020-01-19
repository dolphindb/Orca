import unittest
from setup.settings import *
from numpy.testing import *
from pandas.util.testing import *
import numpy as np
import dolphindb_numpy as dnp
import pandas as pd
import orca


class FunctionCosTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_function_math_cos_scalar(self):
        self.assertEqual(dnp.cos(1.2 + 1j), np.cos(1.2 + 1j))
        self.assertEqual(dnp.cos(dnp.pi/2.), np.cos(np.pi/2.))
        self.assertEqual(dnp.cos(0.5), np.cos(0.5))
        self.assertEqual(dnp.cos(1), np.cos(1))
        self.assertEqual(dnp.cos(-1), np.cos(-1))
        self.assertEqual(dnp.cos(0), np.cos(0))
        self.assertEqual(dnp.isnan(dnp.cos(dnp.nan)), True)
        self.assertEqual(np.isnan(np.cos(np.nan)), True)

    def test_function_math_cos_list(self):
        npa = np.cos(list(np.array([0., 30., 45., 60., 90., np.nan]) * np.pi / 180.))
        dnpa = dnp.cos(list(dnp.array([0., 30., 45., 60., 90., dnp.nan]) * dnp.pi / 180.))
        assert_array_equal(dnpa, npa)

    def test_function_math_cos_array(self):
        npa = np.cos(np.array([0., 30., 45., 60., 90., np.nan]) * np.pi / 180.)
        dnpa = dnp.cos(dnp.array([0., 30., 45., 60., 90., dnp.nan]) * dnp.pi / 180.)
        assert_array_equal(dnpa, npa)

    def test_function_math_cos_series(self):
        ps = pd.Series(np.array([0., 30., 45., 60., 90., np.nan]) * np.pi / 180.)
        os = orca.Series(ps)
        assert_series_equal(dnp.cos(os).to_pandas(), np.cos(ps))

    def test_function_math_cos_dataframe(self):
        pdf = pd.DataFrame({"cola": np.array([0., 30., np.nan, 45., 60., 90., np.nan]) * np.pi / 180.,
                            "colb": np.array([0., 30., np.nan, 45., 60., np.nan, 90.]) * np.pi / 180.})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(dnp.cos(odf).to_pandas(), np.cos(pdf))


if __name__ == '__main__':
    unittest.main()
