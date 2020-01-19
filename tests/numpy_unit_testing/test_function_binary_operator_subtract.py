import unittest
from setup.settings import *
from numpy.testing import *
from pandas.util.testing import *
import numpy as np
import dolphindb_numpy as dnp
import pandas as pd
import orca


class FunctionSubtractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_function_math_binary_subtract_scalar(self):
        self.assertEqual(dnp.subtract(1.2 + 1j, 1.2 - 1j), np.subtract(1.2 + 1j, 1.2 - 1j))
        self.assertEqual(dnp.subtract(0.5, 9), np.subtract(0.5, 9))
        self.assertEqual(dnp.subtract(-1, 8.5), np.subtract(-1, 8.5))

        self.assertEqual(dnp.subtract(1, 4), -3)
        self.assertEqual(np.subtract(1, 4), -3)
        self.assertEqual(dnp.subtract(1, 4), np.subtract(1, 4))

        self.assertEqual(dnp.subtract(1, -5), 6)
        self.assertEqual(np.subtract(1, -5), 6)
        self.assertEqual(dnp.subtract(1, -5), np.subtract(1, -5))

        self.assertEqual(dnp.subtract(0, 9), -9)
        self.assertEqual(np.subtract(0, 9), -9)
        self.assertEqual(dnp.subtract(0, 9), np.subtract(0, 9))

        self.assertEqual(dnp.isnan(dnp.subtract(dnp.nan, -5)), True)
        self.assertEqual(np.isnan(np.subtract(dnp.nan, -5)), True)

    def test_function_math_binary_subtract_list(self):
        lst1 = [1, 2, 3]
        lst2 = [4, 6, 9]
        assert_array_equal(dnp.subtract(lst1, lst2), np.subtract(lst1, lst2))

    def test_function_math_binary_subtract_array_with_scalar(self):
        npa = np.array([1, 2, 3])
        dnpa = dnp.array([1, 2, 3])
        assert_array_equal(dnp.subtract(dnpa, 1), np.subtract(npa, 1))
        assert_array_equal(dnp.subtract(dnpa, dnp.nan), np.subtract(npa, np.nan))
        assert_array_equal(dnp.subtract(1, dnpa), np.subtract(1, npa))

    def test_function_math_binary_subtract_array_with_array(self):
        npa1 = np.array([1, 2, 3])
        npa2 = np.array([4, 6, 9])

        dnpa1 = dnp.array([1, 2, 3])
        dnpa2 = dnp.array([4, 6, 9])
        assert_array_equal(dnp.subtract(dnpa1, dnpa2), np.subtract(npa1, npa2))

    def test_function_math_binary_subtract_array_with_array_param_out(self):
        npa1 = np.array([1, 2, 3])
        npa2 = np.array([4, 6, 9])
        npa = np.zeros(shape=(1, 3))

        dnpa1 = dnp.array([1, 2, 3])
        dnpa2 = dnp.array([4, 6, 9])
        dnpa = dnp.zeros(shape=(1, 3))

        np.subtract(npa1, npa2, out=npa)
        dnp.subtract(dnpa1, dnpa2, out=dnpa)
        assert_array_equal(dnpa, npa)

    def test_function_math_binary_subtract_array_with_series(self):
        npa = np.array([1, 2, 3])
        dnpa = dnp.array([1, 2, 3])
        ps = pd.Series([4, 6, 9])
        os = orca.Series([4, 6, 9])
        assert_series_equal(dnp.subtract(dnpa, os).to_pandas(), np.subtract(npa, ps))
        assert_series_equal(dnp.subtract(os, dnpa).to_pandas(), np.subtract(ps, npa))

    def test_function_math_binary_subtract_array_with_dataframe(self):
        npa = np.array([1, 2, 3])
        dnpa = dnp.array([1, 2, 3])
        pdf = pd.DataFrame({'A': [4, 6, 9]})
        odf = orca.DataFrame({'A': [4, 6, 9]})
        # TODO: orca subtract bug
        # assert_frame_equal(odf.subtract(dnpa, axis=0).to_pandas(), pdf.subtract(npa, axis=0))


if __name__ == '__main__':
    unittest.main()
