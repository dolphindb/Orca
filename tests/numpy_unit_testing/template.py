import unittest
from setup.settings import *
from numpy.testing import *
import numpy as np
import dolphindb_numpy as dnp
import pandas as pd
import orca


class FunctionSinTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_function_sin(self):
        npa = np.array([1, 2])
        dnpa = dnp.array([1, 2])
        assert_array_equal(dnpa, npa)


if __name__ == '__main__':
    unittest.main()
