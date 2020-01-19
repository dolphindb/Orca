import unittest
from setup.settings import *
from numpy.testing import *
import numpy as np
import dolphindb_numpy as dnp
import pandas as pd
import orca


class FunctionArrayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_array_param_object(self):
        # init with list
        lst = [1, 2, 3]
        npa = np.array(lst)
        dnpa = dnp.array(lst)
        assert_array_equal(dnpa, npa)

        # init with pandas series
        ps = pd.Series(['lama', 'cow', 'lama', 'beetle', 'lama', 'hippo'], name='animal')
        npa = np.array(ps)
        dnpa = dnp.array(ps)
        assert_array_equal(dnpa, npa)

        # init with orca series
        os = orca.Series(ps)
        dnpa = dnp.array(os)
        assert_array_equal(dnpa, npa)

        # init with 2-dim list
        npa = np.array([[1, 2], [3, 4]])
        dnpa = dnp.array([[1, 2], [3, 4]])
        assert_array_equal(dnpa, npa)

        # init with np.mat
        npa = np.array(np.mat('1 2; 3 4'))
        dnpa = dnp.array(dnp.mat('1 2; 3 4'))
        assert_array_equal(dnpa, npa)

        # init with dataframe
        pdf = pd.DataFrame([[1, 2], [3, 4]])
        odf = orca.DataFrame(pdf)
        npa = np.array(pdf)
        dnpa = dnp.array(odf)
        assert_array_equal(dnpa, npa)

    def test_array_param_dtype(self):
        lst = [1, 2, 3]

        # np.int8
        npa = np.array(lst, dtype=np.int8)
        dnpa = dnp.array(lst, dtype=dnp.int8)
        assert_array_equal(dnpa, npa)

        # np.int16
        npa = np.array(lst, dtype=np.int16)
        dnpa = dnp.array(lst, dtype=dnp.int16)
        assert_array_equal(dnpa, npa)

        # np.int32
        npa = np.array(lst, dtype=np.int32)
        dnpa = dnp.array(lst, dtype=dnp.int32)
        assert_array_equal(dnpa, npa)

        # np.int64
        npa = np.array(lst, dtype=np.int64)
        dnpa = dnp.array(lst, dtype=dnp.int64)
        assert_array_equal(dnpa, npa)

        # np.uint8
        npa = np.array(lst, dtype=np.uint8)
        dnpa = dnp.array(lst, dtype=dnp.uint8)
        assert_array_equal(dnpa, npa)

        # np.uint16
        npa = np.array(lst, dtype=np.uint16)
        dnpa = dnp.array(lst, dtype=dnp.uint16)
        assert_array_equal(dnpa, npa)

        # np.uint32
        npa = np.array(lst, dtype=np.uint32)
        dnpa = dnp.array(lst, dtype=dnp.uint32)
        assert_array_equal(dnpa, npa)

        # np.uint64
        npa = np.array(lst, dtype=np.uint64)
        dnpa = dnp.array(lst, dtype=dnp.uint64)
        assert_array_equal(dnpa, npa)

        # np.intp
        npa = np.array(lst, dtype=np.intp)
        dnpa = dnp.array(lst, dtype=dnp.intp)
        assert_array_equal(dnpa, npa)

        # np.uintp
        npa = np.array(lst, dtype=np.uintp)
        dnpa = dnp.array(lst, dtype=dnp.uintp)
        assert_array_equal(dnpa, npa)

        # np.float32
        npa = np.array(lst, dtype=np.float32)
        dnpa = dnp.array(lst, dtype=dnp.float32)
        assert_array_equal(dnpa, npa)

        # np.float64
        npa = np.array(lst, dtype=np.float64)
        dnpa = dnp.array(lst, dtype=dnp.float64)
        assert_array_equal(dnpa, npa)

        # np.complex64
        npa = np.array(lst, dtype=np.complex64)
        dnpa = dnp.array(lst, dtype=dnp.complex64)
        assert_array_equal(dnpa, npa)

        # np.complex128
        npa = np.array(lst, dtype=np.complex128)
        dnpa = dnp.array(lst, dtype=dnp.complex128)
        assert_array_equal(dnpa, npa)

        # complex
        npa = np.array(lst, dtype=complex)
        dnpa = dnp.array(lst, dtype=complex)
        assert_array_equal(dnpa, npa)

    def test_array_param_copy(self):
        pass

    def test_array_param_order(self):
        pass

    def test_array_param_subok(self):
        npa = np.array(np.mat('1 2; 3 4'), subok=True)
        dnpa = dnp.array(np.mat('1 2; 3 4'), subok=True)
        assert_array_equal(dnpa, npa)

    def test_array_param_ndmin(self):
        lst = [1, 2, 3]
        npa = np.array(lst, ndmin=2)
        dnpa = dnp.array(lst, ndmin=2)
        assert_array_equal(dnpa, npa)

        # init with dataframe
        pdf = pd.DataFrame([[1, 2], [3, 4]])
        odf = orca.DataFrame(pdf)
        npa = np.array(pdf, ndmin=2)
        dnpa = dnp.array(odf, ndmin=2)
        assert_array_equal(dnpa, npa)