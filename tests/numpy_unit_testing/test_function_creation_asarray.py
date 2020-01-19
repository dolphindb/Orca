import unittest
from setup.settings import *
from numpy.testing import *
import numpy as np
import dolphindb_numpy as dnp
import pandas as pd
import orca


class TopicOnesZerosTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_function_asarray(self):
        x = [1, 2, 3]
        npa = np.asarray(x)
        dnpa = dnp.asarray(x)
        assert_equal(dnpa, npa)


if __name__ == '__main__':
    unittest.main()
