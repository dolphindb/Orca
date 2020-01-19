import unittest
from setup.settings import *
from numpy.testing import *
import numpy as np
import dolphindb_numpy as dnp
import pandas as pd
import orca


class FunctionLinspaceTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_function_linspace(self):
        npa = np.linspace(2.0, 3.0, num=5)
        # TODO: NOT IMPLEMENTED
        # dnpa = dnp.linspace(2.0, 3.0, num=5)
        # assert_equal(dnpa, npa)


if __name__ == '__main__':
    unittest.main()
