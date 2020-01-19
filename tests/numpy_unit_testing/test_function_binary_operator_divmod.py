import unittest
from setup.settings import *
from numpy.testing import *
from pandas.util.testing import *
import numpy as np
import dolphindb_numpy as dnp
import pandas as pd
import orca


class FunctionDivmodTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_function_math_binary_divmod_scalar(self):
        self.assertEqual(repr(dnp.divmod(0.5, 9)), repr(np.divmod(0.5, 9)))
        self.assertEqual(repr(dnp.divmod(-1, 8.5)), repr(np.divmod(-1, 8.5)))
        self.assertEqual(repr(dnp.divmod(1, -5)), repr(np.divmod(1, -5)))
        self.assertEqual(repr(dnp.divmod(0, 9)), repr(np.divmod(0, 9)))
        self.assertEqual(repr(dnp.divmod(dnp.nan, -5)), repr(np.divmod(dnp.nan, -5)))

    def test_function_math_binary_divmod_list(self):
        lst1 = [1, 2, 3]
        lst2 = [4, 6, 9]

        assert_array_equal(dnp.divmod(lst1, lst2), np.divmod(lst1, lst2))

    def test_function_math_binary_divmod_array_with_scalar(self):
        npa = np.array([1, 2, 3])
        dnpa = dnp.array([1, 2, 3])

        assert_array_equal(dnp.divmod(dnpa, 1), np.divmod(npa, 1))
        assert_array_equal(dnp.divmod(dnpa, dnp.nan), np.divmod(npa, np.nan))

        assert_array_equal(dnp.divmod(1, dnpa), np.divmod(1, npa))

    def test_function_math_binary_divmod_array_with_array(self):
        npa1 = np.array([1, 2, 3])
        npa2 = np.array([4, 6, 9])

        dnpa1 = dnp.array([1, 2, 3])
        dnpa2 = dnp.array([4, 6, 9])

        assert_array_equal(dnp.divmod(dnpa1, dnpa2), np.divmod(npa1, npa2))

    def test_function_math_binary_divmod_array_with_array_param_out(self):
        npa1 = np.array([1, 2, 3])
        npa2 = np.array([4, 6, 9])
        npa = np.zeros(shape=(1, 3))
        npb = np.zeros(shape=(1, 3))

        dnpa1 = dnp.array([1, 2, 3])
        dnpa2 = dnp.array([4, 6, 9])
        dnpa = dnp.zeros(shape=(1, 3))
        dnpb = dnp.zeros(shape=(1, 3))

        np.divmod(npa1, npa2, out=(npa, npb))
        dnp.divmod(dnpa1, dnpa2, out=(dnpa, dnpb))

        assert_array_equal(dnpa, npa)
        assert_array_equal(dnpb, npb)

    def test_function_math_binary_divmod_array_with_series(self):
        npa = np.array([1, 2, 3])
        dnpa = dnp.array([1, 2, 3])
        ps = pd.Series([4, 6, 9])
        os = orca.Series([4, 6, 9])

        odf1, odf2 = dnp.divmod(dnpa, os)
        pdf1, pdf2 = np.divmod(npa, ps)
        assert_series_equal(odf1.to_pandas(), pdf1)
        assert_series_equal(odf2.to_pandas(), pdf2)

        odf1, odf2 = dnp.divmod(os, dnpa)
        pdf1, pdf2 = np.divmod(ps, npa)
        assert_series_equal(odf1.to_pandas(), pdf1)
        assert_series_equal(odf2.to_pandas(), pdf2)


if __name__ == '__main__':
    unittest.main()
