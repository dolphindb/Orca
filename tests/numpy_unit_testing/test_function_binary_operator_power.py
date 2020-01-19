import unittest
from setup.settings import *
from numpy.testing import *
from pandas.util.testing import *
import numpy as np
import dolphindb_numpy as dnp
import pandas as pd
import orca


class FunctionPowerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_function_math_binary_power_scalar(self):
        self.assertEqual(dnp.power(1.2 + 1j, 1.2 - 1j), np.power(1.2 + 1j, 1.2 - 1j))
        self.assertEqual(dnp.power(0.5, 9), np.power(0.5, 9))

        self.assertEqual(dnp.isnan(dnp.power(-1, 8.5)), True)
        self.assertEqual(np.isnan(np.power(-1, 8.5)), True)

        self.assertEqual(dnp.power(1, 4), 1)
        self.assertEqual(np.power(1, 4), 1)
        self.assertEqual(dnp.power(1, 4), np.power(1, 4))

        with self.assertRaises(ValueError):
            dnp.power(1, -5)
        with self.assertRaises(ValueError):
            np.power(1, -5)

        self.assertEqual(dnp.power(0, 9), 0)
        self.assertEqual(np.power(0, 9), 0)
        self.assertEqual(dnp.power(0, 9), np.power(0, 9))

        self.assertEqual(dnp.isnan(dnp.power(dnp.nan, -5)), True)
        self.assertEqual(np.isnan(np.power(dnp.nan, -5)), True)

    def test_function_math_binary_power_list(self):
        lst1 = [1, 2, 3]
        lst2 = [4, 6, 9]
        assert_array_equal(dnp.power(lst1, lst2), np.power(lst1, lst2))

    def test_function_math_binary_power_array_with_scalar(self):
        npa = np.array([1, 2, 3])
        dnpa = dnp.array([1, 2, 3])
        assert_array_equal(dnp.power(dnpa, 1), np.power(npa, 1))
        assert_array_equal(dnp.power(dnpa, dnp.nan), np.power(npa, np.nan))
        assert_array_equal(dnp.power(1, dnpa), np.power(1, npa))

    def test_function_math_binary_power_array_with_array(self):
        npa1 = np.array([1, 2, 3])
        npa2 = np.array([4, 6, 9])

        dnpa1 = dnp.array([1, 2, 3])
        dnpa2 = dnp.array([4, 6, 9])
        assert_array_equal(dnp.power(dnpa1, dnpa2), np.power(npa1, npa2))

    def test_function_math_binary_power_array_with_array_param_out(self):
        npa1 = np.array([1, 2, 3])
        npa2 = np.array([4, 6, 9])
        npa = np.zeros(shape=(1, 3))

        dnpa1 = dnp.array([1, 2, 3])
        dnpa2 = dnp.array([4, 6, 9])
        dnpa = dnp.zeros(shape=(1, 3))

        np.power(npa1, npa2, out=npa)
        dnp.power(dnpa1, dnpa2, out=dnpa)
        assert_array_equal(dnpa, npa)

    # TODO: Orca support power vec
    # def test_function_math_binary_power_array_with_series(self):
    #     npa = np.array([1, 2, 3])
    #     dnpa = dnp.array([1, 2, 3])
    #     ps = pd.Series([4, 6, 9])
    #     os = orca.Series([4, 6, 9])
    #     assert_series_equal(dnp.power(dnpa, os).to_pandas(), np.power(npa, ps))
    #     assert_series_equal(dnp.power(os, dnpa).to_pandas(), np.power(ps, npa))

    # TODO: Orca support power vec
    # def test_function_math_binary_power_array_with_dataframe(self):
    #     npa = np.array([1, 2, 3])
    #     dnpa = dnp.array([1, 2, 3])
    #     pdf = pd.DataFrame({'A': [4, 6, 9]})
    #     odf = orca.DataFrame({'A': [4, 6, 9]})
    #     assert_frame_equal(odf.pow(dnpa, axis=0).to_pandas(), pdf.pow(npa, axis=0))


if __name__ == '__main__':
    unittest.main()
