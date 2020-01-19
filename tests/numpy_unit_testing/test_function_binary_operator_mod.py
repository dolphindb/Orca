import unittest
from setup.settings import *
from numpy.testing import *
from pandas.util.testing import *
import numpy as np
import dolphindb_numpy as dnp
import pandas as pd
import orca


class FunctionModTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_function_math_binary_mod_scalar(self):
        self.assertEqual(dnp.mod(0.5, 9), np.mod(0.5, 9))
        self.assertEqual(dnp.mod(-1, 8.5), np.mod(-1, 8.5))

        self.assertEqual(dnp.mod(1, 4), 1)
        self.assertEqual(np.mod(1, 4), 1)
        self.assertEqual(dnp.mod(1, 4), np.mod(1, 4))

        self.assertEqual(dnp.mod(1, -5), -4)
        self.assertEqual(np.mod(1, -5), -4)
        self.assertEqual(dnp.mod(1, -5), np.mod(1, -5))

        self.assertEqual(dnp.mod(0, 9), 0)
        self.assertEqual(np.mod(0, 9), 0)
        self.assertEqual(dnp.mod(0, 9), np.mod(0, 9))

        self.assertEqual(dnp.isnan(dnp.mod(dnp.nan, -5)), True)
        self.assertEqual(np.isnan(np.mod(dnp.nan, -5)), True)

    def test_function_math_binary_mod_list(self):
        lst1 = [1, 2, 3]
        lst2 = [4, 6, 9]
        assert_array_equal(dnp.mod(lst1, lst2), np.mod(lst1, lst2))

    def test_function_math_binary_mod_array_with_scalar(self):
        npa = np.array([1, 2, 3])
        dnpa = dnp.array([1, 2, 3])
        assert_array_equal(dnp.mod(dnpa, 1), np.mod(npa, 1))
        assert_array_equal(dnp.mod(dnpa, dnp.nan), np.mod(npa, np.nan))
        assert_array_equal(dnp.mod(1, dnpa), np.mod(1, npa))

    def test_function_math_binary_mod_array_with_array(self):
        npa1 = np.array([1, 2, 3])
        npa2 = np.array([4, 6, 9])

        dnpa1 = dnp.array([1, 2, 3])
        dnpa2 = dnp.array([4, 6, 9])
        assert_array_equal(dnp.mod(dnpa1, dnpa2), np.mod(npa1, npa2))

    def test_function_math_binary_mod_array_with_array_param_out(self):
        npa1 = np.array([1, 2, 3])
        npa2 = np.array([4, 6, 9])
        npa = np.zeros(shape=(1, 3))

        dnpa1 = dnp.array([1, 2, 3])
        dnpa2 = dnp.array([4, 6, 9])
        dnpa = dnp.zeros(shape=(1, 3))

        np.mod(npa1, npa2, out=npa)
        dnp.mod(dnpa1, dnpa2, out=dnpa)
        assert_array_equal(dnpa, npa)

    def test_function_math_binary_mod_array_with_series(self):
        npa = np.array([1, 2, 3])
        dnpa = dnp.array([1, 2, 3])
        ps = pd.Series([4, 6, 9])
        os = orca.Series([4, 6, 9])
        assert_series_equal(dnp.mod(dnpa, os).to_pandas(), np.mod(npa, ps))
        assert_series_equal(dnp.mod(os, dnpa).to_pandas(), np.mod(ps, npa))

    def test_function_math_binary_mod_array_with_dataframe(self):
        npa = np.array([1, 2, 3])
        dnpa = dnp.array([1, 2, 3])
        pdf = pd.DataFrame({'A': [4, 6, 9]})
        odf = orca.DataFrame({'A': [4, 6, 9]})
        assert_frame_equal(odf.mod(dnpa, axis=0).to_pandas(), pdf.mod(npa, axis=0))


if __name__ == '__main__':
    unittest.main()
