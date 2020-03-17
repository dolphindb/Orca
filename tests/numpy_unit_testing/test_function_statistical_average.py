import unittest
from setup.settings import *
from numpy.testing import *
import numpy as np
import dolphindb_numpy as dnp
import pandas as pd
import orca


class FunctionAverageTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_function_math_average_scalar(self):
        self.assertEqual(dnp.average(0.5), np.average(0.5))
        self.assertEqual(dnp.average(1), np.average(1))
        self.assertEqual(dnp.average(-1), np.average(-1))
        self.assertEqual(dnp.average(0), np.average(0))
        self.assertEqual(dnp.isnan(dnp.average(dnp.nan)), True)
        self.assertEqual(np.isnan(np.average(np.nan)), True)

    def test_function_math_average_list(self):
        npa = np.average([1, 8, 27, -27, 0, 5, np.nan])
        dnpa = dnp.average([1, 8, 27, -27, 0, 5, dnp.nan])
        assert_array_equal(dnpa, npa)

    def test_function_math_average_array(self):
        npa = np.average(np.array([1, 8, 27, -27, 0, 5, np.nan]))
        dnpa = dnp.average(dnp.array([1, 8, 27, -27, 0, 5, dnp.nan]))
        assert_array_equal(dnpa, npa)

    def test_function_math_average_series(self):
        ps = pd.Series([-1, 8, 27, -27, 0, 5, np.nan])
        os = orca.Series(ps)
        self.assertEqual(dnp.isnan(dnp.average(os)), True)
        self.assertEqual(np.isnan(np.average(ps)), True)

        ps = pd.Series([-1, 8, 27, -27, 0, 5])
        os = orca.Series(ps)
        self.assertEqual(dnp.average(os), np.average(ps))

    def test_function_math_average_dataframe(self):
        pdf = pd.DataFrame({"cola": [-1, 8, 27, -27, 0, 5, np.nan, 1.5, 1.7, 2.0, np.nan],
                            "colb": [-1, 8, 27, -27, 0, 5, np.nan, 1.5, 1.7, np.nan, 2.0]})
        odf = orca.DataFrame(pdf)
        self.assertEqual(dnp.isnan(dnp.average(odf)), True)
        self.assertEqual(np.isnan(np.average(pdf)), True)

        pdf = pd.DataFrame({"cola": [-1, 8, 27, -27, 0, 5, 1.5, 1.7, 2.0],
                            "colb": [-1, 8, 27, -27, 0, 5, 1.5, 1.7, 2.0]})
        odf = orca.DataFrame(pdf)
        self.assertEqual(dnp.average(odf), np.average(pdf))


if __name__ == '__main__':
    unittest.main()
