import unittest
from setup.settings import *
from numpy.testing import *
from pandas.util.testing import *
import numpy as np
import dolphindb_numpy as dnp
import pandas as pd
import orca


class FunctionLessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_function_math_binary_less_scalar(self):
        self.assertEqual(dnp.less(1.2 + 1j, 1.2 - 1j), np.less(1.2 + 1j, 1.2 - 1j))
        self.assertEqual(dnp.less(0.5, 9), np.less(0.5, 9))
        self.assertEqual(dnp.less(-1, 8.5), np.less(-1, 8.5))
        self.assertEqual(dnp.less(1, 4), np.less(1, 4))
        self.assertEqual(dnp.less(1, -5), np.less(1, -5))
        self.assertEqual(dnp.less(0, 9), np.less(0, 9))
        self.assertEqual(dnp.less(dnp.nan, -5), np.less(dnp.nan, -5))

    def test_function_math_binary_less_list(self):
        lst1 = [1, 2, 3]
        lst2 = [4, 6, 9]
        assert_array_equal(dnp.less(lst1, lst2), np.less(lst1, lst2))

    def test_function_math_binary_less_array_with_scalar(self):
        npa = np.array([1, 2, 3])
        dnpa = dnp.array([1, 2, 3])
        assert_array_equal(dnp.less(dnpa, 1), np.less(npa, 1))
        assert_array_equal(dnp.less(dnpa, dnp.nan), np.less(npa, np.nan))
        assert_array_equal(dnp.less(1, dnpa), np.less(1, npa))

    def test_function_math_binary_less_array_with_array(self):
        npa1 = np.array([1, 2, 3])
        npa2 = np.array([4, 6, 9])

        dnpa1 = dnp.array([1, 2, 3])
        dnpa2 = dnp.array([4, 6, 9])
        assert_array_equal(dnp.less(dnpa1, dnpa2), np.less(npa1, npa2))

    def test_function_math_binary_less_array_with_array_param_out(self):
        npa1 = np.array([1, 2, 3])
        npa2 = np.array([4, 6, 9])
        npa = np.zeros(shape=(1, 3))

        dnpa1 = dnp.array([1, 2, 3])
        dnpa2 = dnp.array([4, 6, 9])
        dnpa = dnp.zeros(shape=(1, 3))

        np.less(npa1, npa2, out=npa)
        dnp.less(dnpa1, dnpa2, out=dnpa)
        assert_array_equal(dnpa, npa)

    def test_function_math_binary_less_array_with_series(self):
        npa = np.array([1, 2, 3])
        dnpa = dnp.array([1, 2, 3])
        ps = pd.Series([4, 6, 9])
        os = orca.Series([4, 6, 9])
        assert_series_equal(dnp.less(dnpa, os).to_pandas(), np.less(npa, ps))
        assert_series_equal(dnp.less(os, dnpa).to_pandas(), np.less(ps, npa))

        pser = pd.Series([1, 2, 4])
        oser = orca.Series([1, 2, 4])
        assert_series_equal(dnp.less(os, oser).to_pandas(), np.less(ps, pser))

    def test_function_math_binary_less_array_with_dataframe(self):
        npa = np.array([1])
        dnpa = dnp.array([1])
        pdf = pd.DataFrame({'A': [4, 6, 9]})
        odf = orca.DataFrame({'A': [4, 6, 9]})
        assert_frame_equal(dnp.less(odf, dnpa).to_pandas(), np.less(pdf, npa))
        assert_frame_equal(dnp.less(dnpa, odf).to_pandas(), np.less(npa, pdf))

        pdfrm = pd.DataFrame({'A': [0, 7, 1]})
        odfrm = orca.DataFrame({'A': [0, 7, 1]})
        assert_frame_equal(dnp.less(odf, odfrm).to_pandas(), np.less(pdf, pdfrm))


if __name__ == '__main__':
    unittest.main()
