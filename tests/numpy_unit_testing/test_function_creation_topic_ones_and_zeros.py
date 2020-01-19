import unittest
from setup.settings import *
from numpy.testing import *
import numpy as np
import dolphindb_numpy as dnp
import pandas as pd
import orca


class FunctionAsarrayTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_topic_ones_and_zeros_function_empty(self):
        npa = np.empty([3, 2], dtype=int)
        dnpa = dnp.empty([3, 2], dtype=int)
        # TODO: DIFFERENCES
        # assert_array_equal(dnpa, npa)

    def test_topic_ones_and_zeros_function_zeros(self):
        npa = np.zeros((5,), dtype=np.int)
        dnpa = dnp.zeros((5,), dtype=dnp.int)
        assert_array_equal(dnpa, npa)

        # TODO: unknown difference
        # npa = np.zeros((2, 2), dtype=[('x', 'i4'), ('y', 'i4')])
        # dnpa = dnp.zeros((2, 2), dtype=[('x', 'i4'), ('y', 'i4')])
        # assert_array_equal(dnpa, npa)

    def test_topic_ones_and_zeros_function_ones(self):
        npa = np.ones([2, 2], dtype=int)
        dnpa = dnp.ones([2, 2], dtype=int)
        assert_array_equal(dnpa, npa)


if __name__ == '__main__':
    unittest.main()
