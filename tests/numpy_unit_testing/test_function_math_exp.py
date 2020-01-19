import unittest
from setup.settings import *
from numpy.testing import *
from pandas.util.testing import *
import numpy as np
import dolphindb_numpy as dnp
import pandas as pd
import orca


class FunctionExpTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_function_math_exp_scalar(self):
        self.assertEqual(dnp.exp(1.2 + 1j), np.exp(1.2 + 1j))
        self.assertEqual(dnp.exp(0.5), np.exp(0.5))
        self.assertEqual(dnp.exp(1), np.exp(1))
        self.assertEqual(dnp.exp(-1), np.exp(-1))
        self.assertEqual(dnp.exp(0), np.exp(0))
        self.assertEqual(dnp.isnan(dnp.exp(dnp.nan)), True)
        self.assertEqual(np.isnan(np.exp(np.nan)), True)

    def test_function_math_exp_list(self):
        npa = np.exp([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0, np.nan])
        dnpa = dnp.exp([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0, dnp.nan])
        assert_array_equal(dnpa, npa)

    def test_function_math_exp_array(self):
        npa = np.exp(np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0, np.nan]))
        dnpa = dnp.exp(dnp.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0, dnp.nan]))
        assert_array_equal(dnpa, npa)
        # TODO: np.linspace not implemented yet
        # npx = np.linspace(-2 * np.pi, 2 * np.pi, 100)
        # npxx = npx + 1j * npx[:, np.newaxis]  # a + ib over complex plane
        # dnpx = dnp.linspace(-2 * dnp.pi, 2 * dnp.pi, 100)
        # dnpxx = dnpx + 1j * dnpx[:, dnp.newaxis]  # a + ib over complex plane
        # assert_array_equal(dnp.exp(dnpxx), np.exp(npxx))

    def test_function_math_exp_series(self):
        ps = pd.Series([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0, np.nan])
        os = orca.Series(ps)
        assert_series_equal(dnp.exp(os).to_pandas(), np.exp(ps))

    def test_function_math_exp_dataframe(self):
        pdf = pd.DataFrame({"cola": [-1.7, -1.5, -0.2, np.nan, 0.2, 1.5, 1.7, 2.0, np.nan],
                            "colb": [-1.7, -1.5, -0.2, np.nan, 0.2, 1.5, 1.7, np.nan, 2.0]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(dnp.exp(odf).to_pandas(), np.exp(pdf))


if __name__ == '__main__':
    unittest.main()
