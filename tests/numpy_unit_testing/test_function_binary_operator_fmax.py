import unittest
from setup.settings import *
from numpy.testing import *
from pandas.util.testing import *
import numpy as np
import dolphindb_numpy as dnp
import pandas as pd
import orca


class FunctionFmaxTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_function_math_binary_fmax_scalar(self):
        self.assertEqual(dnp.fmax(1.2 + 1j, 1.2 - 1j), np.fmax(1.2 + 1j, 1.2 - 1j))
        self.assertEqual(dnp.fmax(0.5, 9), np.fmax(0.5, 9))
        self.assertEqual(dnp.fmax(-1, 8.5), np.fmax(-1, 8.5))
        self.assertEqual(dnp.fmax(1, 4), np.fmax(1, 4))
        self.assertEqual(dnp.fmax(1, -5), np.fmax(1, -5))
        self.assertEqual(dnp.fmax(0, 9), np.fmax(0, 9))
        self.assertEqual(dnp.fmax(dnp.nan, -5), np.fmax(dnp.nan, -5))

    def test_function_math_binary_fmax_list(self):
        lst1 = [1, 2, 3]
        lst2 = [4, 6, 9]
        assert_array_equal(dnp.fmax(lst1, lst2), np.fmax(lst1, lst2))

    def test_function_math_binary_fmax_array_with_scalar(self):
        npa = np.array([1, 2, 3])
        dnpa = dnp.array([1, 2, 3])
        assert_array_equal(dnp.fmax(dnpa, 1), np.fmax(npa, 1))
        assert_array_equal(dnp.fmax(dnpa, dnp.nan), np.fmax(npa, np.nan))
        assert_array_equal(dnp.fmax(1, dnpa), np.fmax(1, npa))

    def test_function_math_binary_fmax_array_with_array(self):
        npa1 = np.array([1, 2, 3])
        npa2 = np.array([4, 6, 9])

        dnpa1 = dnp.array([1, 2, 3])
        dnpa2 = dnp.array([4, 6, 9])
        assert_array_equal(dnp.fmax(dnpa1, dnpa2), np.fmax(npa1, npa2))

        # 1-dim with 2-dim
        assert_array_equal(dnp.fmax(dnpa1, dnp.eye(3)), np.fmax(npa1, np.eye(3)))

    def test_function_math_binary_fmax_array_with_array_param_out(self):
        npa1 = np.array([1, 2, 3])
        npa2 = np.array([4, 6, 9])
        npa = np.zeros(shape=(1, 3))

        dnpa1 = dnp.array([1, 2, 3])
        dnpa2 = dnp.array([4, 6, 9])
        dnpa = dnp.zeros(shape=(1, 3))

        np.fmax(npa1, npa2, out=npa)
        dnp.fmax(dnpa1, dnpa2, out=dnpa)
        assert_array_equal(dnpa, npa)

    def test_function_math_binary_fmax_array_with_series(self):
        npa = np.array([1, 2, 3])
        dnpa = dnp.array([1, 2, 3])
        ps = pd.Series([4, 6, 9])
        os = orca.Series([4, 6, 9])
        # TODO: fmax bug
        # assert_series_equal(dnp.fmax(dnpa, os).to_pandas(), np.fmax(npa, ps))
        # assert_series_equal(dnp.fmax(os, dnpa).to_pandas(), np.fmax(ps, npa))
        #
        # pser = pd.Series([1, 2, 4])
        # oser = orca.Series([1, 2, 4])
        # assert_series_equal(dnp.fmax(os, oser).to_pandas(), np.fmax(ps, pser))

    def test_function_math_binary_fmax_array_with_dataframe(self):
        npa = np.array([1])
        dnpa = dnp.array([1])
        pdf = pd.DataFrame({'A': [4, 6, 9]})
        odf = orca.DataFrame({'A': [4, 6, 9]})
        # TODO: fmax bug
        # assert_frame_equal(dnp.fmax(odf, dnpa).to_pandas(), np.fmax(pdf, npa))
        # assert_frame_equal(dnp.fmax(dnpa, odf).to_pandas(), np.fmax(npa, pdf))
        #
        # pdfrm = pd.DataFrame({'A': [0, 7, 1]})
        # odfrm = orca.DataFrame({'A': [0, 7, 1]})
        # assert_frame_equal(dnp.fmax(odf, odfrm).to_pandas(), np.fmax(pdf, pdfrm))


if __name__ == '__main__':
    unittest.main()
