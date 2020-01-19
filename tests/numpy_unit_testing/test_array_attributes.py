import unittest
from setup.settings import *
from numpy.testing import *
import numpy as np
import dolphindb_numpy as dnp
import pandas as pd
import orca


class ArrayAttributesTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_array_attributes_itemsize(self):

        npa = np.array([1, 2, 3, 4, 5])
        dnpa = dnp.array([1, 2, 3, 4, 5])
        self.assertEqual(dnpa.itemsize, npa.itemsize)

        # TODO: dtype bug
        # npa = np.array([1, 2, 3, 4, 5], dtype=np.int8)
        # dnpa = dnp.array([1, 2, 3, 4, 5], dtype=np.int8)
        # self.assertEqual(dnpa.itemsize, npa.itemsize)


if __name__ == '__main__':
    unittest.main()
