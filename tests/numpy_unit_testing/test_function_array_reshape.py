import unittest
from setup.settings import *
from numpy.testing import *
import numpy as np
import dolphindb_numpy as dnp
import pandas as pd
import orca


class FunctionReshapeTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_function_reshape(self):
        npa = np.arange(24)
        dnpa = dnp.arange(24)

        npb = npa.reshape(8, 3)
        dnpb = dnpa.reshape(8, 3)
        assert_array_equal(dnpb, npb)

        npb = npa.reshape(2, 4, 3)  # b 现在拥有三个维度
        dnpb = dnpa.reshape(2, 4, 3)
        assert_array_equal(dnpb, npb)


if __name__ == '__main__':
    unittest.main()
