import unittest
from setup.settings import *
from numpy.testing import *
from pandas.util.testing import *
import numpy as np
import dolphindb_numpy as dnp
import pandas as pd
import orca


class FunctionMaximumTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_function_math_binary_maximum_scalar(self):
        self.assertEqual(dnp.maximum(1.2 + 1j, 1.2 - 1j), np.maximum(1.2 + 1j, 1.2 - 1j))
        self.assertEqual(dnp.maximum(0.5, 9), np.maximum(0.5, 9))
        self.assertEqual(dnp.maximum(-1, 8.5), np.maximum(-1, 8.5))
        self.assertEqual(dnp.maximum(1, 4), np.maximum(1, 4))
        self.assertEqual(dnp.maximum(1, -5), np.maximum(1, -5))
        self.assertEqual(dnp.maximum(0, 9), np.maximum(0, 9))
        self.assertEqual(dnp.isnan(dnp.maximum(dnp.nan, -5)), True)
        self.assertEqual(np.isnan(np.maximum(dnp.nan, -5)), True)

    def test_function_math_binary_maximum_list(self):
        lst1 = [1, 2, 3]
        lst2 = [4, 6, 9]
        assert_array_equal(dnp.maximum(lst1, lst2), np.maximum(lst1, lst2))

    def test_function_math_binary_maximum_array_with_scalar(self):
        npa = np.array([1, 2, 3])
        dnpa = dnp.array([1, 2, 3])
        assert_array_equal(dnp.maximum(dnpa, 1), np.maximum(npa, 1))
        assert_array_equal(dnp.maximum(dnpa, dnp.nan), np.maximum(npa, np.nan))
        assert_array_equal(dnp.maximum(1, dnpa), np.maximum(1, npa))

    def test_function_math_binary_maximum_array_with_array(self):
        npa1 = np.array([1, 2, 3])
        npa2 = np.array([4, 6, 9])

        dnpa1 = dnp.array([1, 2, 3])
        dnpa2 = dnp.array([4, 6, 9])
        assert_array_equal(dnp.maximum(dnpa1, dnpa2), np.maximum(npa1, npa2))

    def test_function_math_binary_maximum_array_with_array_param_out(self):
        npa1 = np.array([1, 2, 3])
        npa2 = np.array([4, 6, 9])
        npa = np.zeros(shape=(1, 3))

        dnpa1 = dnp.array([1, 2, 3])
        dnpa2 = dnp.array([4, 6, 9])
        dnpa = dnp.zeros(shape=(1, 3))

        np.maximum(npa1, npa2, out=npa)
        dnp.maximum(dnpa1, dnpa2, out=dnpa)
        assert_array_equal(dnpa, npa)

    def test_function_math_binary_maximum_array_with_series(self):
        npa = np.array([1, 2, 3])
        dnpa = dnp.array([1, 2, 3])
        ps = pd.Series([4, 6, 9])
        os = orca.Series([4, 6, 9])

        assert_series_equal(dnp.maximum(dnpa, os).to_pandas(), np.maximum(npa, ps))
        # TODO: maximum bug
        # assert_series_equal(dnp.maximum(os, dnpa).to_pandas(), np.maximum(ps, npa))
        #
        # pser = pd.Series([1, 2, 4])
        # oser = orca.Series([1, 2, 4])
        # assert_series_equal(dnp.maximum(os, oser).to_pandas(), np.maximum(ps, pser))

    def test_function_math_binary_maximum_array_with_dataframe(self):
        npa = np.array([1])
        dnpa = dnp.array([1])
        pdf = pd.DataFrame({'A': [4, 6, 9]})
        odf = orca.DataFrame({'A': [4, 6, 9]})
        # TODO: maximum bug
        # assert_frame_equal(dnp.maximum(odf, dnpa).to_pandas(), np.maximum(pdf, npa))
        # assert_frame_equal(dnp.maximum(dnpa, odf).to_pandas(), np.maximum(npa, pdf))
        #
        # pdfrm = pd.DataFrame({'A': [0, 7, 1]})
        # odfrm = orca.DataFrame({'A': [0, 7, 1]})
        # assert_frame_equal(dnp.maximum(odf, odfrm).to_pandas(), np.maximum(pdf, pdfrm))


if __name__ == '__main__':
    unittest.main()
