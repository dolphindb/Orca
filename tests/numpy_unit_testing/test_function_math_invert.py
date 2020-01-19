import unittest
from setup.settings import *
from numpy.testing import *
from pandas.util.testing import *
import numpy as np
import dolphindb_numpy as dnp
import pandas as pd
import orca


class FunctionInvertTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_function_math_invert_scalar(self):
        self.assertEqual(dnp.invert(1), -2)
        self.assertEqual(np.invert(1), -2)
        # self.assertEqual(dnp.invert(1), np.invert(1))

        self.assertEqual(dnp.invert(-1), 0)
        self.assertEqual(np.invert(-1), 0)
        # self.assertEqual(dnp.invert(-1), np.invert(-1))

        self.assertEqual(dnp.invert(0), -1)
        self.assertEqual(np.invert(0), -1)
        # self.assertEqual(dnp.invert(0), np.invert(0))

        with self.assertRaises(TypeError):
            np.invert(np.nan)
        with self.assertRaises(TypeError):
            dnp.invert(dnp.nan)

    def test_function_math_invert_list(self):
        npa = np.invert([-2, -20, -5, 0, 5, 7, 2])
        dnpa = dnp.invert([-2, -20, -5, 0, 5, 7, 2])
        assert_array_equal(dnpa, npa)

    def test_function_math_invert_array(self):
        npa = np.invert(np.array([-2, -20, -5, 0, 5, 7, 2]))
        dnpa = dnp.invert(dnp.array([-2, -20, -5, 0, 5, 7, 2]))
        assert_array_equal(dnpa, npa)

    def test_function_math_invert_series(self):
        ps = pd.Series([-2, -20, -5, 0, 5, 7, 2])
        os = orca.Series(ps)
        # TODO: invert bug
        # assert_series_equal(dnp.invert(os).to_pandas(), np.invert(ps))

    def test_function_math_invert_dataframe(self):
        pdf = pd.DataFrame({"cola": [7, -5, -2, 2, 0, 15, 17],
                            "colb": [7, -5, -2, 2, 0, 15, 17]})
        odf = orca.DataFrame(pdf)
        # TODO: invert bug
        # assert_frame_equal(dnp.invert(odf).to_pandas(), np.invert(pdf))


if __name__ == '__main__':
    unittest.main()
