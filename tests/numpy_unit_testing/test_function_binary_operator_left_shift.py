import unittest
from setup.settings import *
from numpy.testing import *
from pandas.util.testing import *
import numpy as np
import dolphindb_numpy as dnp
import pandas as pd
import orca


class FunctionLeftshiftTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_function_math_binary_left_shift_scalar(self):
        self.assertEqual(dnp.left_shift(1, 4), np.left_shift(1, 4))
        self.assertEqual(dnp.left_shift(1, -5), np.left_shift(1, -5))
        self.assertEqual(dnp.left_shift(0, 9), np.left_shift(0, 9))

    def test_function_math_binary_left_shift_list(self):
        lst1 = [1, 2, 3]
        lst2 = [4, 6, 9]
        assert_array_equal(dnp.left_shift(lst1, lst2), np.left_shift(lst1, lst2))

    def test_function_math_binary_left_shift_array_with_scalar(self):
        npa = np.array([1, 2, 3])
        dnpa = dnp.array([1, 2, 3])
        assert_array_equal(dnp.left_shift(dnpa, 1), np.left_shift(npa, 1))
        assert_array_equal(dnp.left_shift(1, dnpa), np.left_shift(1, npa))

    def test_function_math_binary_left_shift_array_with_array(self):
        npa1 = np.array([1, 2, 3])
        npa2 = np.array([4, 6, 9])

        dnpa1 = dnp.array([1, 2, 3])
        dnpa2 = dnp.array([4, 6, 9])
        assert_array_equal(dnp.left_shift(dnpa1, dnpa2), np.left_shift(npa1, npa2))

    def test_function_math_binary_left_shift_array_with_array_param_out(self):
        npa1 = np.array([1, 2, 3])
        npa2 = np.array([4, 6, 9])
        npa = np.zeros(shape=(1, 3))

        dnpa1 = dnp.array([1, 2, 3])
        dnpa2 = dnp.array([4, 6, 9])
        dnpa = dnp.zeros(shape=(1, 3))

        np.left_shift(npa1, npa2, out=npa)
        dnp.left_shift(dnpa1, dnpa2, out=dnpa)
        assert_array_equal(dnpa, npa)

    def test_function_math_binary_left_shift_array_with_series(self):
        npa = np.array([1, 2, 3])
        dnpa = dnp.array([1, 2, 3])
        ps = pd.Series([4, 6, 9])
        os = orca.Series([4, 6, 9])
        assert_series_equal(dnp.left_shift(dnpa, os).to_pandas(), np.left_shift(npa, ps), check_dtype=False)
        assert_series_equal(dnp.left_shift(os, dnpa).to_pandas(), np.left_shift(ps, npa), check_dtype=False)

        pser = pd.Series([1, 2, 3])
        oser = orca.Series([1, 2, 3])
        assert_series_equal(dnp.left_shift(os, oser).to_pandas(), np.left_shift(ps, pser))


if __name__ == '__main__':
    unittest.main()
