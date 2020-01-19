import unittest
from setup.settings import *
from numpy.testing import *
import numpy as np
import dolphindb_numpy as dnp
import pandas as pd
import orca


class FunctionFrombufferTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_function_frombuffer(self):
        s = b'hello world'
        npa = np.frombuffer(s, dtype='S1', count=5, offset=6)
        dnpa = dnp.frombuffer(s, dtype='S1', count=5, offset=6)
        assert_array_equal(dnpa, npa)


if __name__ == '__main__':
    unittest.main()
