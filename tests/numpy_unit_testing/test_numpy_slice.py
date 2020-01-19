import unittest
from setup.settings import *
from numpy.testing import *
import numpy as np
import dolphindb_numpy as dnp
import pandas as pd
import orca


class SliceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_slice(self):
        a = np.arange(10)
        np.arccos(a)
        s = slice(2, 7, 2)
        a_s = a[s]
        b = dnp.arange(10)
        bs = b[s]
        # print(bs)



