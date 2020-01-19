import unittest
from setup.settings import *
from numpy.testing import *
from pandas.util.testing import *
import numpy as np
import dolphindb_numpy as dnp
import pandas as pd
import orca


class FunctionAbsoluteTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_function_math_absolute(self):
        self.assertEqual(dnp.absolute(1.2 + 1j), np.absolute(1.2 + 1j))
        self.assertEqual(dnp.absolute(0.5), np.absolute(0.5))

        self.assertEqual(dnp.absolute(1), 1)
        self.assertEqual(np.absolute(1), 1)
        # self.assertEqual(dnp.absolute(1), np.absolute(1))

        self.assertEqual(dnp.absolute(-1), 1)
        self.assertEqual(np.absolute(-1), 1)
        # self.assertEqual(dnp.absolute(-1), np.absolute(-1))

        self.assertEqual(dnp.absolute(0), 0)
        self.assertEqual(np.absolute(0), 0)
        # self.assertEqual(dnp.absolute(0), np.absolute(0))

        self.assertEqual(dnp.isnan(dnp.absolute(dnp.nan)), np.isnan(np.absolute(np.nan)))

    def test_function_math_absolute_list(self):
        npa = np.absolute([-1.2, 1.2, 1.2 + 1j, np.nan])
        dnpa = dnp.absolute([-1.2, 1.2, 1.2 + 1j, dnp.nan])
        assert_array_equal(dnpa, npa)

    def test_function_math_absolute_array(self):
        npa = np.absolute(np.array([-1.2, 1.2, 1.2 + 1j, np.nan]))
        dnpa = dnp.absolute(dnp.array([-1.2, 1.2, 1.2 + 1j, dnp.nan]))
        assert_array_equal(dnpa, npa)

    def test_function_math_absolute_series(self):
        ps = pd.Series([-1.2, 1.2, np.nan])
        os = orca.Series(ps)
        assert_series_equal(dnp.absolute(os).to_pandas(), np.absolute(ps))

    def test_function_math_absolute_dataframe(self):
        pdf = pd.DataFrame({"cola": [-1.7, -1.5, -0.2, np.nan, 1.5, 1.7, 2.0, np.nan],
                            "colb": [-1.7, -1.5, -0.2, np.nan, 1.5, 1.7, np.nan, 2.0]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(dnp.absolute(odf).to_pandas(), np.absolute(pdf))


if __name__ == '__main__':
    unittest.main()
