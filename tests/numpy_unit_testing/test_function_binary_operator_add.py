import unittest
from setup.settings import *
from numpy.testing import *
from pandas.util.testing import *
import numpy as np
import dolphindb_numpy as dnp
import pandas as pd
import orca


class FunctionAddTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_function_math_binary_add_scalar(self):
        self.assertEqual(dnp.add(1.2 + 1j, 1.2 - 1j), np.add(1.2 + 1j, 1.2 - 1j))
        self.assertEqual(dnp.add(0.5, 9), np.add(0.5, 9))
        self.assertEqual(dnp.add(-1, 8.5), np.add(-1, 8.5))

        self.assertEqual(dnp.add(1, 4), 5)
        self.assertEqual(np.add(1, 4), 5)
        self.assertEqual(dnp.add(1, 4), np.add(1, 4))

        self.assertEqual(dnp.add(1, -5), -4)
        self.assertEqual(np.add(1, -5), -4)
        self.assertEqual(dnp.add(1, -5), np.add(1, -5))

        self.assertEqual(dnp.add(0, 9), 9)
        self.assertEqual(np.add(0, 9), 9)
        self.assertEqual(dnp.add(0, 9), np.add(0, 9))

        self.assertEqual(dnp.isnan(dnp.add(dnp.nan, -5)), True)
        self.assertEqual(np.isnan(np.add(dnp.nan, -5)), True)

    def test_function_math_binary_add_list(self):
        lst1 = [1, 2, 3]
        lst2 = [4, 6, 9]
        assert_array_equal(dnp.add(lst1, lst2), np.add(lst1, lst2))

    def test_function_math_binary_add_array_with_scalar(self):
        npa = np.array([1, 2, 3])
        dnpa = dnp.array([1, 2, 3])
        assert_array_equal(dnp.add(dnpa, 1), np.add(npa, 1))
        assert_array_equal(dnp.add(dnpa, dnp.nan), np.add(npa, np.nan))
        assert_array_equal(dnp.add(1, dnpa), np.add(1, npa))

    def test_function_math_binary_add_array_with_array(self):
        npa1 = np.array([1, 2, 3])
        npa2 = np.array([4, 6, 9])

        dnpa1 = dnp.array([1, 2, 3])
        dnpa2 = dnp.array([4, 6, 9])
        assert_array_equal(dnp.add(dnpa1, dnpa2), np.add(npa1, npa2))

    def test_function_math_binary_add_array_with_array_param_out(self):
        npa1 = np.array([1, 2, 3])
        npa2 = np.array([4, 6, 9])
        npa = np.zeros(shape=(1, 3))

        dnpa1 = dnp.array([1, 2, 3])
        dnpa2 = dnp.array([4, 6, 9])
        dnpa = dnp.zeros(shape=(1, 3))

        np.add(npa1, npa2, out=npa)
        dnp.add(dnpa1, dnpa2, out=dnpa)
        # TODO: dolphindb numpy add bug
        # assert_array_equal(dnpa.to_numpy(), npa)

    def test_function_math_binary_add_array_with_series(self):
        npa = np.array([1, 2, 3])
        dnpa = dnp.array([1, 2, 3])
        ps = pd.Series([4, 6, 9])
        os = orca.Series([4, 6, 9])
        assert_series_equal(dnp.add(dnpa, os).to_pandas(), np.add(npa, ps))
        assert_series_equal(dnp.add(os, dnpa).to_pandas(), np.add(ps, npa))

        pser = pd.Series([1, 2, 4])
        oser = orca.Series([1, 2, 4])
        assert_series_equal(dnp.add(os, oser).to_pandas(), np.add(ps, pser))

    def test_function_math_binary_add_array_with_dataframe(self):
        npa = np.array([1, 2, 3])
        dnpa = dnp.array([1, 2, 3])
        pdf = pd.DataFrame({'A': [4, 6, 9]})
        odf = orca.DataFrame({'A': [4, 6, 9]})
        # TODO: orca add bug
        # assert_frame_equal(odf.add(dnpa, axis=0).to_pandas(), pdf.add(npa, axis=0))


if __name__ == '__main__':
    unittest.main()
