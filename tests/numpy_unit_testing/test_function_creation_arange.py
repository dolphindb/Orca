import unittest
from setup.settings import *
from numpy.testing import *
import numpy as np
import dolphindb_numpy as dnp
import pandas as pd
import orca


class FunctionArangeTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_function_arange(self):
        npa = np.arange(24)
        dnpa = dnp.arange(24)
        assert_array_equal(dnpa, npa)

        npa = np.arange(1, 11, 1)
        dnpa = dnp.arange(1, 11, 1)
        assert_array_equal(dnpa, npa)


if __name__ == '__main__':
    unittest.main()
