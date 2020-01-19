import unittest
from setup.settings import *
from numpy.testing import *
from pandas.util.testing import *
import numpy as np
import dolphindb_numpy as dnp
import pandas as pd
import orca


class FunctionPositiveTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_function_math_positive_scalar(self):
        self.assertEqual(dnp.positive(1.2 + 1j), np.positive(1.2 + 1j))
        self.assertEqual(dnp.positive(1e-10), np.positive(1e-10))
        self.assertEqual(dnp.positive(0.5), np.positive(0.5))

        self.assertEqual(dnp.positive(1), 1)
        self.assertEqual(np.positive(1), 1)
        self.assertEqual(dnp.positive(1), np.positive(1))

        self.assertEqual(dnp.positive(-1), -1)
        self.assertEqual(np.positive(-1), -1)
        self.assertEqual(dnp.positive(-1), np.positive(-1))

        self.assertEqual(dnp.positive(0), 0)
        self.assertEqual(np.positive(0), 0)
        self.assertEqual(dnp.positive(0), np.positive(0))

        self.assertEqual(dnp.isnan(dnp.positive(dnp.nan)), np.isnan(np.positive(np.nan)))

    def test_function_math_positive_list(self):
        npa = np.positive([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0, np.nan])
        dnpa = dnp.positive([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0, dnp.nan])
        assert_array_equal(dnpa, npa)

    def test_function_math_positive_array(self):
        npa = np.positive(np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0, np.nan]))
        dnpa = dnp.positive(dnp.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0, dnp.nan]))
        assert_array_equal(dnpa, npa)

    def test_function_math_positive_series(self):
        ps = pd.Series([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0, np.nan])
        os = orca.Series(ps)
        assert_series_equal(dnp.positive(os).to_pandas(), np.positive(ps))

    def test_function_math_positive_dataframe(self):
        pdf = pd.DataFrame({"cola": [-1.7, -1.5, -0.2, 0.2, np.nan, 1.5, 1.7, 2.0, 2.0, np.nan],
                            "colb": [-1.7, -1.5, -0.2, 0.2, np.nan, 1.5, 1.7, 2.0, np.nan, 2.0]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(dnp.positive(odf).to_pandas(), np.positive(pdf))


if __name__ == '__main__':
    unittest.main()
