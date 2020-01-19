import unittest
from setup.settings import *
from numpy.testing import *
from pandas.util.testing import *
import numpy as np
import dolphindb_numpy as dnp
import pandas as pd
import orca


class FunctionNegativeTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_function_math_negative_scalar(self):
        self.assertEqual(dnp.negative(1.2 + 1j), np.negative(1.2 + 1j))
        self.assertEqual(dnp.negative(1e-10), np.negative(1e-10))
        self.assertEqual(dnp.negative(0.5), np.negative(0.5))

        self.assertEqual(dnp.negative(1), -1)
        self.assertEqual(np.negative(1), -1)
        self.assertEqual(dnp.negative(1), np.negative(1))

        self.assertEqual(dnp.negative(-1), 1)
        self.assertEqual(np.negative(-1), 1)
        self.assertEqual(dnp.negative(-1), np.negative(-1))

        self.assertEqual(dnp.negative(0), 0)
        self.assertEqual(np.negative(0), 0)
        self.assertEqual(dnp.negative(0), np.negative(0))

        self.assertEqual(dnp.isnan(dnp.negative(dnp.nan)), np.isnan(np.negative(np.nan)))

    def test_function_math_negative_list(self):
        npa = np.negative([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0, np.nan])
        dnpa = dnp.negative([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0, dnp.nan])
        assert_array_equal(dnpa, npa)

    def test_function_math_negative_array(self):
        npa = np.negative(np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0, np.nan]))
        dnpa = dnp.negative(dnp.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0, dnp.nan]))
        assert_array_equal(dnpa, npa)

    def test_function_math_negative_series(self):
        ps = pd.Series([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0, np.nan])
        os = orca.Series(ps)
        assert_series_equal(dnp.negative(os).to_pandas(), np.negative(ps))

    def test_function_math_negative_dataframe(self):
        pdf = pd.DataFrame({"cola": [-1.7, -1.5, -0.2, 0.2, np.nan, 1.5, 1.7, 2.0, 2.0, np.nan],
                            "colb": [-1.7, -1.5, -0.2, 0.2, np.nan, 1.5, 1.7, 2.0, np.nan, 2.0]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(dnp.negative(odf).to_pandas(), np.negative(pdf))


if __name__ == '__main__':
    unittest.main()
