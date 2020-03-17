import unittest
from setup.settings import *
from numpy.testing import *
import numpy as np
import dolphindb_numpy as dnp
import pandas as pd
import orca


class FunctionBincountTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_function_math_bincount_list(self):
        npa = np.bincount([1, 8, 27, 0, 5])
        dnpa = dnp.bincount([1, 8, 27, 0, 5,])
        assert_array_equal(dnpa, npa)

    def test_function_math_bincount_array(self):
        npa = np.bincount(np.array([1, 8, 27, 0, 5]))
        dnpa = dnp.bincount(dnp.array([1, 8, 27, 0, 5,]))
        assert_array_equal(dnpa, npa)

    def test_function_math_bincount_series(self):
        ps = pd.Series([8, 27, 0, 5])
        os = orca.Series(ps)
        assert_array_equal(dnp.bincount(os), np.bincount(ps))


if __name__ == '__main__':
    unittest.main()
