import sys
import unittest
from setup.settings import *
from numpy.testing import *
import numpy as np
import dolphindb_numpy as dnp
import pandas as pd
import orca


class DtypeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_dtype(self):
        a = np.dtype('int16')
        b = dnp.dtype('int16')

    def test_dtype_struct(self):
        dt = np.dtype([('age', np.int8)])
        ddt = dnp.dtype([('age', np.int8)])
        # print(dt.__str__())
        # assert_equal(repr(dt.__str__()),repr(ddt.__str__()))

    def test_dtype_operator(self):
        a = np.dtype('int16')
        a1=np.dtype('int8')
        print(type(a))
        print(a*5)
        print(type(a*5))

        b = dnp.dtype('int16')
        print(type(b))
        print(b*5)
        print(type(b*5))


    def test_dtype_type(self):
        a=np.int(3)
        a1=np.int8(4)
        print(type(a*a1))

        a = dnp.int(3)
        a1 = dnp.int8(4)
        print(type(a*a1))

        b=np.int64

    def test_dtype_newbyteorder(self):
         sys_is_le = sys.byteorder == 'little'
         native_code = sys_is_le and '<' or '>'
         swapped_code = sys_is_le and '>' or '<'
         native_dt = np.dtype(native_code + 'i2')
         swapped_dt = np.dtype(swapped_code + 'i2')
         print(native_dt.newbyteorder('S') == swapped_dt)

         a = np.dtype('int16')
         b = dnp.dtype('int16')

    def test_dtype_read(self):
        pass
        # TODO: MAKE NO SENSE
        # npdt = np.dtype([('age', np.int8)])
        # dnpdt = dnp.dtype([('age', np.int8)])
        # npa = np.array([(10,), (20,), (30,)], dtype=npdt)
        # npb = dnp.array([(10,), (20,), (30,)], dtype=dnpdt)