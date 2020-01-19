import unittest
from setup.settings import *
from numpy.testing import *
from pandas.util.testing import *
import numpy as np
import dolphindb_numpy as dnp
import pandas as pd
import orca


class FunctionFloorTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_function_math_floor_scalar(self):
        self.assertEqual(dnp.floor(1e-10), np.floor(1e-10))
        self.assertEqual(dnp.floor(0.5), np.floor(0.5))
        self.assertEqual(dnp.floor(1), np.floor(1))
        self.assertEqual(dnp.floor(0), np.floor(0))
        self.assertEqual(dnp.isnan(dnp.floor(dnp.nan)), True)
        self.assertEqual(np.isnan(np.floor(np.nan)), True)

    def test_function_math_floor_list(self):
        npa = np.floor([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0, np.nan])
        dnpa = dnp.floor([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0, dnp.nan])
        assert_array_equal(dnpa, npa)

    def test_function_math_floor_array(self):
        npa = np.floor(np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0, np.nan]))
        dnpa = dnp.floor(dnp.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0, dnp.nan]))
        assert_array_equal(dnpa, npa)

    def test_function_math_floor_series(self):
        ps = pd.Series([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0, np.nan])
        os = orca.Series(ps)
        # TODO: BUG
        # assert_series_equal(dnp.floor(os).to_pandas(), np.floor(ps))

    def test_function_math_floor_dataframe(self):
        pdf = pd.DataFrame({"cola": [-1.7, -1.5, -0.2, 0.2, np.nan, 1.5, 1.7, 2.0, 2.0, np.nan],
                            "colb": [-1.7, -1.5, -0.2, 0.2, np.nan, 1.5, 1.7, 2.0, np.nan, 2.0]})
        odf = orca.DataFrame(pdf)
        # TODO: BUG
        # assert_frame_equal(dnp.floor(odf).to_pandas(), np.floor(pdf))


if __name__ == '__main__':
    unittest.main()
