import unittest
from setup.settings import *
from numpy.testing import *
import numpy as np
import dolphindb_numpy as dnp
import pandas as pd
import orca


class FunctionViewTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_function_view(self):
        npa = np.array([(1, 2)], dtype=np.int8)
        # BUG CASES
        # dnpa = dnp.array([(1, 2)], dtype=dnp.int8)
        #
        # npm = npa.view(type=np.matrix)
        # dnpm = dnpa.view(type=dnp.matrix)
        #
        # assert_array_equal(dnpm, npm)
        # self.assertEqual(type(dnpm), type(npm))
        #
        # npa = np.array([(1, 2), (3, 4)])
        # dnpa = dnp.array([(1, 2), (3, 4)])
        #
        # npv = npa.view().reshape(-1, 2)
        # dnpv = dnpa.view().reshape(-1, 2)
        # assert_array_equal(dnpm, npm)