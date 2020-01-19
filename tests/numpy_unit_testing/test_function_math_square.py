import unittest
from setup.settings import *
from numpy.testing import *
from pandas.util.testing import *
import numpy as np
import dolphindb_numpy as dnp
import pandas as pd
import orca


class FunctionSquareTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_function_math_square_scalar(self):
        self.assertEqual(dnp.square(1.2 + 1j), np.square(1.2 + 1j))
        self.assertEqual(dnp.square(1e-10), np.square(1e-10))
        self.assertEqual(dnp.square(0.5), np.square(0.5))

        self.assertEqual(dnp.square(1), 1)
        self.assertEqual(np.square(1), 1)
        self.assertEqual(dnp.square(1), np.square(1))

        self.assertEqual(dnp.square(-1), 1)
        self.assertEqual(np.square(-1), 1)
        self.assertEqual(dnp.square(-1), np.square(-1))

        self.assertEqual(dnp.square(0), 0)
        self.assertEqual(np.square(0), 0)
        self.assertEqual(dnp.square(0), np.square(0))

        self.assertEqual(dnp.isnan(dnp.square(dnp.nan)), np.isnan(np.square(np.nan)))

    def test_function_math_square_list(self):
        npa = np.square([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, np.inf, np.nan])
        dnpa = dnp.square([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, dnp.inf, dnp.nan])
        assert_array_equal(dnpa, npa)

    def test_function_math_square_array(self):
        npa = np.square(np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, np.inf, np.nan]))
        dnpa = dnp.square(dnp.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, dnp.inf, dnp.nan]))
        assert_array_equal(dnpa, npa)

    def test_function_math_square_series(self):
        ps = pd.Series([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, np.inf, np.nan])
        os = orca.Series(ps)
        assert_series_equal(dnp.square(os).to_pandas(), np.square(ps))

    def test_function_math_square_dataframe(self):
        pdf = pd.DataFrame({"cola": [-1.7, -1.5, -0.2, 0.2, np.nan, 1.5, 1.7, np.inf, 2.0, np.nan],
                            "colb": [-1.7, -1.5, -0.2, 0.2, np.nan, 1.5, 1.7, np.inf, np.nan, 2.0]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(dnp.square(odf).to_pandas(), np.square(pdf))


if __name__ == '__main__':
    unittest.main()
