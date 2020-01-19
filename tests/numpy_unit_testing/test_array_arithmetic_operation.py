import unittest
from setup.settings import *
from numpy.testing import *
from pandas.util.testing import *
import numpy as np
import dolphindb_numpy as dnp
import pandas as pd
import orca


class ArithmeticLogicOperationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_arithmetic_add(self):
        npa = np.array([1, 2, 3]) + np.array([4, 5, 6])
        dnpa = dnp.array([1, 2, 3]) + dnp.array([4, 5, 6])
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3]) + pd.Series([4, 5, 6], dtype=np.int8)
        dnpa = dnp.array([1, 2, 3]) + orca.Series([4, 5, 6], dtype=np.int8)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = pd.Series([4, 5, 6], dtype=np.int8) + np.array([1, 2, 3])
        dnpa = orca.Series([4, 5, 6], dtype=np.int8) + dnp.array([1, 2, 3])
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = np.array([1, 2, 3], dtype=np.int8) + pd.Series([4, 5, 6], dtype=np.int8)
        dnpa = dnp.array([1, 2, 3], dtype=np.int8) + orca.Series([4, 5, 6], dtype=np.int8)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.Series([4, 5, 6], dtype=np.int8) + np.array([1, 2, 3], dtype=np.int8)
        dnpa = orca.Series([4, 5, 6], dtype=np.int8) + dnp.array([1, 2, 3], dtype=np.int8)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1], dtype=np.int8) + pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int8)
        dnpa = dnp.array([1], dtype=np.int8) + orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int8)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int8) + np.array([1], dtype=np.int8)
        dnpa = orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int8) + dnp.array([1], dtype=np.int8)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1, 2, 3], dtype=np.int16) + np.array([4, 5, 6], dtype=np.int16)
        dnpa = dnp.array([1, 2, 3], dtype=np.int16) + dnp.array([4, 5, 6], dtype=np.int16)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.int16) + pd.Series([4, 5, 6], dtype=np.int16)
        dnpa = dnp.array([1, 2, 3], dtype=np.int16) + orca.Series([4, 5, 6], dtype=np.int16)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.Series([4, 5, 6], dtype=np.int16) + np.array([1, 2, 3], dtype=np.int16)
        dnpa = orca.Series([4, 5, 6], dtype=np.int16) + dnp.array([1, 2, 3], dtype=np.int16)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1], dtype=np.int16) + pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int16)
        dnpa = dnp.array([1], dtype=np.int16) + orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int16)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int16) + np.array([1], dtype=np.int16)
        dnpa = orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int16) + dnp.array([1], dtype=np.int16)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1, 2, 3], dtype=np.int32) + np.array([4, 5, 6], dtype=np.int32)
        dnpa = dnp.array([1, 2, 3], dtype=np.int32) + dnp.array([4, 5, 6], dtype=np.int32)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.int32) + pd.Series([4, 5, 6], dtype=np.int32)
        dnpa = dnp.array([1, 2, 3], dtype=np.int32) + orca.Series([4, 5, 6], dtype=np.int32)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.Series([4, 5, 6], dtype=np.int32) + np.array([1, 2, 3], dtype=np.int32)
        dnpa = orca.Series([4, 5, 6], dtype=np.int32) + dnp.array([1, 2, 3], dtype=np.int32)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1], dtype=np.int32) + pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int32)
        dnpa = dnp.array([1], dtype=np.int32) + orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int32)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int32) + np.array([1], dtype=np.int32)
        dnpa = orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int32) + dnp.array([1], dtype=np.int32)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1, 2, 3], dtype=np.int64) + np.array([4, 5, 6], dtype=np.int64)
        dnpa = dnp.array([1, 2, 3], dtype=np.int64) + dnp.array([4, 5, 6], dtype=np.int64)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.int64) + pd.Series([4, 5, 6], dtype=np.int64)
        dnpa = dnp.array([1, 2, 3], dtype=np.int64) + orca.Series([4, 5, 6], dtype=np.int64)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.Series([4, 5, 6], dtype=np.int64) + np.array([1, 2, 3], dtype=np.int64)
        dnpa = orca.Series([4, 5, 6], dtype=np.int64) + dnp.array([1, 2, 3], dtype=np.int64)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1], dtype=np.int64) + pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int64)
        dnpa = dnp.array([1], dtype=np.int64) + orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int64)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int64) + np.array([1], dtype=np.int64)
        dnpa = orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int64) + dnp.array([1], dtype=np.int64)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1, 2, 3], dtype=np.uint8) + np.array([4, 5, 6], dtype=np.uint8)
        dnpa = dnp.array([1, 2, 3], dtype=np.uint8) + dnp.array([4, 5, 6], dtype=np.uint8)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.uint16) + np.array([4, 5, 6], dtype=np.uint16)
        dnpa = dnp.array([1, 2, 3], dtype=np.uint16) + dnp.array([4, 5, 6], dtype=np.uint16)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.uint32) + np.array([4, 5, 6], dtype=np.uint32)
        dnpa = dnp.array([1, 2, 3], dtype=np.uint32) + dnp.array([4, 5, 6], dtype=np.uint32)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.uint64) + np.array([4, 5, 6], dtype=np.uint64)
        dnpa = dnp.array([1, 2, 3], dtype=np.uint64) + dnp.array([4, 5, 6], dtype=np.uint64)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.intp) + np.array([4, 5, 6], dtype=np.intp)
        dnpa = dnp.array([1, 2, 3], dtype=np.intp) + dnp.array([4, 5, 6], dtype=np.intp)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.intp) + pd.Series([4, 5, 6], dtype=np.intp)
        dnpa = dnp.array([1, 2, 3], dtype=np.intp) + orca.Series([4, 5, 6], dtype=np.intp)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.Series([4, 5, 6], dtype=np.intp) + np.array([1, 2, 3], dtype=np.intp)
        dnpa = orca.Series([4, 5, 6], dtype=np.intp) + dnp.array([1, 2, 3], dtype=np.intp)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1], dtype=np.intp) + pd.DataFrame({"a": [4, 5, 6]}, dtype=np.intp)
        dnpa = dnp.array([1], dtype=np.intp) + orca.DataFrame({"a": [4, 5, 6]}, dtype=np.intp)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.DataFrame({"a": [4, 5, 6]}, dtype=np.intp) + np.array([1], dtype=np.intp)
        dnpa = orca.DataFrame({"a": [4, 5, 6]}, dtype=np.intp) + dnp.array([1], dtype=np.intp)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1, 2, 3], dtype=np.uintp) + np.array([4, 5, 6], dtype=np.uintp)
        dnpa = dnp.array([1, 2, 3], dtype=np.uintp) + dnp.array([4, 5, 6], dtype=np.uintp)
        assert_array_equal(dnpa, npa)

        npa = np.array([12.0, np.nan, 15.0], dtype=np.float32) + np.array([14.7, 5.5, 6.8], dtype=np.float32)
        dnpa = dnp.array([12.0, np.nan, 15.0], dtype=np.float32) + dnp.array([14.7, 5.5, 6.8], dtype=np.float32)
        assert_array_equal(dnpa, npa)

        npa = np.array([12.0, np.nan, 15.0], dtype=np.float32) + pd.Series([14.7, 5.5, 6.8], dtype=np.float32)
        dnpa = dnp.array([12.0, np.nan, 15.0], dtype=np.float32) + orca.Series([14.7, 5.5, 6.8], dtype=np.float32)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.Series([14.7, 5.5, 6.8], dtype=np.float32) + np.array([12.0, np.nan, 15.0], dtype=np.float32)
        dnpa = orca.Series([14.7, 5.5, 6.8], dtype=np.float32) + dnp.array([12.0, np.nan, 15.0], dtype=np.float32)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([12.0], dtype=np.float32) + pd.DataFrame({'A': [14.7, np.nan, 6.8]}, dtype=np.float32)
        dnpa = dnp.array([12.0], dtype=np.float32) + orca.DataFrame({'A': [14.7, np.nan, 6.8]}, dtype=np.float32)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.DataFrame({'A': [14.7, np.nan, 6.8]}, dtype=np.float32) + np.array([12.0], dtype=np.float32)
        dnpa = orca.DataFrame({'A': [14.7, np.nan, 6.8]}, dtype=np.float32) + dnp.array([12.0], dtype=np.float32)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([12.0, np.nan, 15.0], dtype=np.float64) + np.array([14.7, 5.5, 6.8], dtype=np.float64)
        dnpa = dnp.array([12.0, np.nan, 15.0], dtype=np.float64) + dnp.array([14.7, 5.5, 6.8], dtype=np.float64)
        assert_array_equal(dnpa, npa)

        npa = np.array([12.0, np.nan, 15.0], dtype=np.float64) + pd.Series([14.7, 5.5, 6.8], dtype=np.float64)
        dnpa = dnp.array([12.0, np.nan, 15.0], dtype=np.float64) + orca.Series([14.7, 5.5, 6.8], dtype=np.float64)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.Series([14.7, 5.5, 6.8], dtype=np.float64) + np.array([12.0, np.nan, 15.0], dtype=np.float64)
        dnpa = orca.Series([14.7, 5.5, 6.8], dtype=np.float64) + dnp.array([12.0, np.nan, 15.0], dtype=np.float64)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([12.0], dtype=np.float64) + pd.DataFrame({'A': [14.7, np.nan, 6.8]}, dtype=np.float64)
        dnpa = dnp.array([12.0], dtype=np.float64) + orca.DataFrame({'A': [14.7, np.nan, 6.8]},
                                                                                  dtype=np.float64)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.DataFrame({'A': [14.7, np.nan, 6.8]}, dtype=np.float64) + np.array([12.0], dtype=np.float64)
        dnpa = orca.DataFrame({'A': [14.7, np.nan, 6.8]}, dtype=np.float64) + dnp.array([12.0],
                                                                                     dtype=np.float64)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([12.0 + 1j, np.nan, 15.0], dtype=np.complex64) + np.array([14.7, 5.5, 6.8 + 3j],
                                                                                 dtype=np.complex64)
        dnpa = dnp.array([12.0 + 1j, np.nan, 15.0], dtype=np.complex64) + dnp.array([14.7, 5.5, 6.8 + 3j],
                                                                                    dtype=np.complex64)
        assert_array_equal(dnpa, npa)

        npa = np.array([12.0 + 1j, np.nan, 15.0], dtype=np.complex128) + np.array([14.7, 5.5, 6.8 + 3j],
                                                                                  dtype=np.complex128)
        dnpa = dnp.array([12.0 + 1j, np.nan, 15.0], dtype=np.complex128) + dnp.array([14.7, 5.5, 6.8 + 3j],
                                                                                     dtype=np.complex128)
        assert_array_equal(dnpa, npa)

    def test_arithmetic_sub(self):
        npa = np.array([1, 2, 3]) - np.array([4, 5, 6])
        dnpa = dnp.array([1, 2, 3]) - dnp.array([4, 5, 6])
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3]) - pd.Series([4, 5, 6], dtype=np.int8)
        dnpa = dnp.array([1, 2, 3]) - orca.Series([4, 5, 6], dtype=np.int8)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = pd.Series([4, 5, 6], dtype=np.int8) - np.array([1, 2, 3])
        dnpa = orca.Series([4, 5, 6], dtype=np.int8) - dnp.array([1, 2, 3])
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = np.array([1, 2, 3], dtype=np.int8) - pd.Series([4, 5, 6], dtype=np.int8)
        dnpa = dnp.array([1, 2, 3], dtype=np.int8) - orca.Series([4, 5, 6], dtype=np.int8)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.Series([4, 5, 6], dtype=np.int8) - np.array([1, 2, 3], dtype=np.int8)
        dnpa = orca.Series([4, 5, 6], dtype=np.int8) - dnp.array([1, 2, 3], dtype=np.int8)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1], dtype=np.int8) - pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int8)
        dnpa = dnp.array([1], dtype=np.int8) - orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int8)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int8) - np.array([1], dtype=np.int8)
        dnpa = orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int8) - dnp.array([1], dtype=np.int8)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1, 2, 3], dtype=np.int16) - np.array([4, 5, 6], dtype=np.int16)
        dnpa = dnp.array([1, 2, 3], dtype=np.int16) - dnp.array([4, 5, 6], dtype=np.int16)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.int16) - pd.Series([4, 5, 6], dtype=np.int16)
        dnpa = dnp.array([1, 2, 3], dtype=np.int16) - orca.Series([4, 5, 6], dtype=np.int16)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.Series([4, 5, 6], dtype=np.int16) - np.array([1, 2, 3], dtype=np.int16)
        dnpa = orca.Series([4, 5, 6], dtype=np.int16) - dnp.array([1, 2, 3], dtype=np.int16)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1], dtype=np.int16) - pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int16)
        dnpa = dnp.array([1], dtype=np.int16) - orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int16)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int16) - np.array([1], dtype=np.int16)
        dnpa = orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int16) - dnp.array([1], dtype=np.int16)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1, 2, 3], dtype=np.int32) - np.array([4, 5, 6], dtype=np.int32)
        dnpa = dnp.array([1, 2, 3], dtype=np.int32) - dnp.array([4, 5, 6], dtype=np.int32)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.int32) - pd.Series([4, 5, 6], dtype=np.int32)
        dnpa = dnp.array([1, 2, 3], dtype=np.int32) - orca.Series([4, 5, 6], dtype=np.int32)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.Series([4, 5, 6], dtype=np.int32) - np.array([1, 2, 3], dtype=np.int32)
        dnpa = orca.Series([4, 5, 6], dtype=np.int32) - dnp.array([1, 2, 3], dtype=np.int32)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1], dtype=np.int32) - pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int32)
        dnpa = dnp.array([1], dtype=np.int32) - orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int32)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int32) - np.array([1], dtype=np.int32)
        dnpa = orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int32) - dnp.array([1], dtype=np.int32)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1, 2, 3], dtype=np.int64) - np.array([4, 5, 6], dtype=np.int64)
        dnpa = dnp.array([1, 2, 3], dtype=np.int64) - dnp.array([4, 5, 6], dtype=np.int64)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.int64) - pd.Series([4, 5, 6], dtype=np.int64)
        dnpa = dnp.array([1, 2, 3], dtype=np.int64) - orca.Series([4, 5, 6], dtype=np.int64)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.Series([4, 5, 6], dtype=np.int64) - np.array([1, 2, 3], dtype=np.int64)
        dnpa = orca.Series([4, 5, 6], dtype=np.int64) - dnp.array([1, 2, 3], dtype=np.int64)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1], dtype=np.int64) - pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int64)
        dnpa = dnp.array([1], dtype=np.int64) - orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int64)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int64) - np.array([1], dtype=np.int64)
        dnpa = orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int64) - dnp.array([1], dtype=np.int64)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1, 2, 3], dtype=np.uint8) - np.array([4, 5, 6], dtype=np.uint8)
        dnpa = dnp.array([1, 2, 3], dtype=np.uint8) - dnp.array([4, 5, 6], dtype=np.uint8)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.uint16) - np.array([4, 5, 6], dtype=np.uint16)
        dnpa = dnp.array([1, 2, 3], dtype=np.uint16) - dnp.array([4, 5, 6], dtype=np.uint16)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.uint32) - np.array([4, 5, 6], dtype=np.uint32)
        dnpa = dnp.array([1, 2, 3], dtype=np.uint32) - dnp.array([4, 5, 6], dtype=np.uint32)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.uint64) - np.array([4, 5, 6], dtype=np.uint64)
        dnpa = dnp.array([1, 2, 3], dtype=np.uint64) - dnp.array([4, 5, 6], dtype=np.uint64)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.intp) - np.array([4, 5, 6], dtype=np.intp)
        dnpa = dnp.array([1, 2, 3], dtype=np.intp) - dnp.array([4, 5, 6], dtype=np.intp)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.intp) - pd.Series([4, 5, 6], dtype=np.intp)
        dnpa = dnp.array([1, 2, 3], dtype=np.intp) - orca.Series([4, 5, 6], dtype=np.intp)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.Series([4, 5, 6], dtype=np.intp) - np.array([1, 2, 3], dtype=np.intp)
        dnpa = orca.Series([4, 5, 6], dtype=np.intp) - dnp.array([1, 2, 3], dtype=np.intp)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1], dtype=np.intp) - pd.DataFrame({"a": [4, 5, 6]}, dtype=np.intp)
        dnpa = dnp.array([1], dtype=np.intp) - orca.DataFrame({"a": [4, 5, 6]}, dtype=np.intp)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.DataFrame({"a": [4, 5, 6]}, dtype=np.intp) - np.array([1], dtype=np.intp)
        dnpa = orca.DataFrame({"a": [4, 5, 6]}, dtype=np.intp) - dnp.array([1], dtype=np.intp)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1, 2, 3], dtype=np.uintp) - np.array([4, 5, 6], dtype=np.uintp)
        dnpa = dnp.array([1, 2, 3], dtype=np.uintp) - dnp.array([4, 5, 6], dtype=np.uintp)
        assert_array_equal(dnpa, npa)

        npa = np.array([12.0, np.nan, 15.0], dtype=np.float32) - np.array([14.7, 5.5, 6.8], dtype=np.float32)
        dnpa = dnp.array([12.0, np.nan, 15.0], dtype=np.float32) - dnp.array([14.7, 5.5, 6.8], dtype=np.float32)
        assert_array_equal(dnpa, npa)

        npa = np.array([12.0, np.nan, 15.0], dtype=np.float32) - pd.Series([14.7, 5.5, 6.8], dtype=np.float32)
        dnpa = dnp.array([12.0, np.nan, 15.0], dtype=np.float32) - orca.Series([14.7, 5.5, 6.8], dtype=np.float32)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.Series([14.7, 5.5, 6.8], dtype=np.float32) - np.array([12.0, np.nan, 15.0], dtype=np.float32)
        dnpa = orca.Series([14.7, 5.5, 6.8], dtype=np.float32) - dnp.array([12.0, np.nan, 15.0], dtype=np.float32)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([12.0], dtype=np.float32) - pd.DataFrame({'A': [14.7, np.nan, 6.8]}, dtype=np.float32)
        dnpa = dnp.array([12.0], dtype=np.float32) - orca.DataFrame({'A': [14.7, np.nan, 6.8]}, dtype=np.float32)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.DataFrame({'A': [14.7, np.nan, 6.8]}, dtype=np.float32) - np.array([12.0], dtype=np.float32)
        dnpa = orca.DataFrame({'A': [14.7, np.nan, 6.8]}, dtype=np.float32) - dnp.array([12.0], dtype=np.float32)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([12.0, np.nan, 15.0], dtype=np.float64) - np.array([14.7, 5.5, 6.8], dtype=np.float64)
        dnpa = dnp.array([12.0, np.nan, 15.0], dtype=np.float64) - dnp.array([14.7, 5.5, 6.8], dtype=np.float64)
        assert_array_equal(dnpa, npa)

        npa = np.array([12.0, np.nan, 15.0], dtype=np.float64) - pd.Series([14.7, 5.5, 6.8], dtype=np.float64)
        dnpa = dnp.array([12.0, np.nan, 15.0], dtype=np.float64) - orca.Series([14.7, 5.5, 6.8], dtype=np.float64)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.Series([14.7, 5.5, 6.8], dtype=np.float64) - np.array([12.0, np.nan, 15.0], dtype=np.float64)
        dnpa = orca.Series([14.7, 5.5, 6.8], dtype=np.float64) - dnp.array([12.0, np.nan, 15.0], dtype=np.float64)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([12.0], dtype=np.float64) - pd.DataFrame({'A': [14.7, np.nan, 6.8]}, dtype=np.float64)
        dnpa = dnp.array([12.0], dtype=np.float64) - orca.DataFrame({'A': [14.7, np.nan, 6.8]},
                                                                                  dtype=np.float64)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.DataFrame({'A': [14.7, np.nan, 6.8]}, dtype=np.float64) - np.array([12.0], dtype=np.float64)
        dnpa = orca.DataFrame({'A': [14.7, np.nan, 6.8]}, dtype=np.float64) - dnp.array([12.0],
                                                                                     dtype=np.float64)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([12.0 - 1j, np.nan, 15.0], dtype=np.complex64) - np.array([14.7, 5.5, 6.8 - 3j],
                                                                                 dtype=np.complex64)
        dnpa = dnp.array([12.0 - 1j, np.nan, 15.0], dtype=np.complex64) - dnp.array([14.7, 5.5, 6.8 - 3j],
                                                                                    dtype=np.complex64)
        assert_array_equal(dnpa, npa)

        npa = np.array([12.0 - 1j, np.nan, 15.0], dtype=np.complex128) - np.array([14.7, 5.5, 6.8 - 3j],
                                                                                  dtype=np.complex128)
        dnpa = dnp.array([12.0 - 1j, np.nan, 15.0], dtype=np.complex128) - dnp.array([14.7, 5.5, 6.8 - 3j],
                                                                                     dtype=np.complex128)
        assert_array_equal(dnpa, npa)

    def test_arithmetic_multiplicate(self):
        npa = np.array([1, 2, 3]) * np.array([4, 5, 6])
        dnpa = dnp.array([1, 2, 3]) * dnp.array([4, 5, 6])
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3]) * pd.Series([4, 5, 6], dtype=np.int8)
        dnpa = dnp.array([1, 2, 3]) * orca.Series([4, 5, 6], dtype=np.int8)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = pd.Series([4, 5, 6], dtype=np.int8) * np.array([1, 2, 3])
        dnpa = orca.Series([4, 5, 6], dtype=np.int8) * dnp.array([1, 2, 3])
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = np.array([1, 2, 3], dtype=np.int8) * pd.Series([4, 5, 6], dtype=np.int8)
        dnpa = dnp.array([1, 2, 3], dtype=np.int8) * orca.Series([4, 5, 6], dtype=np.int8)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.Series([4, 5, 6], dtype=np.int8) * np.array([1, 2, 3], dtype=np.int8)
        dnpa = orca.Series([4, 5, 6], dtype=np.int8) * dnp.array([1, 2, 3], dtype=np.int8)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1], dtype=np.int8) * pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int8)
        dnpa = dnp.array([1], dtype=np.int8) * orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int8)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int8) * np.array([1], dtype=np.int8)
        dnpa = orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int8) * dnp.array([1], dtype=np.int8)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1, 2, 3], dtype=np.int16) * np.array([4, 5, 6], dtype=np.int16)
        dnpa = dnp.array([1, 2, 3], dtype=np.int16) * dnp.array([4, 5, 6], dtype=np.int16)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.int16) * pd.Series([4, 5, 6], dtype=np.int16)
        dnpa = dnp.array([1, 2, 3], dtype=np.int16) * orca.Series([4, 5, 6], dtype=np.int16)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.Series([4, 5, 6], dtype=np.int16) * np.array([1, 2, 3], dtype=np.int16)
        dnpa = orca.Series([4, 5, 6], dtype=np.int16) * dnp.array([1, 2, 3], dtype=np.int16)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1], dtype=np.int16) * pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int16)
        dnpa = dnp.array([1], dtype=np.int16) * orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int16)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int16) * np.array([1], dtype=np.int16)
        dnpa = orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int16) * dnp.array([1], dtype=np.int16)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1, 2, 3], dtype=np.int32) * np.array([4, 5, 6], dtype=np.int32)
        dnpa = dnp.array([1, 2, 3], dtype=np.int32) * dnp.array([4, 5, 6], dtype=np.int32)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.int32) * pd.Series([4, 5, 6], dtype=np.int32)
        dnpa = dnp.array([1, 2, 3], dtype=np.int32) * orca.Series([4, 5, 6], dtype=np.int32)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.Series([4, 5, 6], dtype=np.int32) * np.array([1, 2, 3], dtype=np.int32)
        dnpa = orca.Series([4, 5, 6], dtype=np.int32) * dnp.array([1, 2, 3], dtype=np.int32)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1], dtype=np.int32) * pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int32)
        dnpa = dnp.array([1], dtype=np.int32) * orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int32)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int32) * np.array([1], dtype=np.int32)
        dnpa = orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int32) * dnp.array([1], dtype=np.int32)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1, 2, 3], dtype=np.int64) * np.array([4, 5, 6], dtype=np.int64)
        dnpa = dnp.array([1, 2, 3], dtype=np.int64) * dnp.array([4, 5, 6], dtype=np.int64)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.int64) * pd.Series([4, 5, 6], dtype=np.int64)
        dnpa = dnp.array([1, 2, 3], dtype=np.int64) * orca.Series([4, 5, 6], dtype=np.int64)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.Series([4, 5, 6], dtype=np.int64) * np.array([1, 2, 3], dtype=np.int64)
        dnpa = orca.Series([4, 5, 6], dtype=np.int64) * dnp.array([1, 2, 3], dtype=np.int64)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1], dtype=np.int64) * pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int64)
        dnpa = dnp.array([1], dtype=np.int64) * orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int64)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int64) * np.array([1], dtype=np.int64)
        dnpa = orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int64) * dnp.array([1], dtype=np.int64)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1, 2, 3], dtype=np.uint8) * np.array([4, 5, 6], dtype=np.uint8)
        dnpa = dnp.array([1, 2, 3], dtype=np.uint8) * dnp.array([4, 5, 6], dtype=np.uint8)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.uint16) * np.array([4, 5, 6], dtype=np.uint16)
        dnpa = dnp.array([1, 2, 3], dtype=np.uint16) * dnp.array([4, 5, 6], dtype=np.uint16)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.uint32) * np.array([4, 5, 6], dtype=np.uint32)
        dnpa = dnp.array([1, 2, 3], dtype=np.uint32) * dnp.array([4, 5, 6], dtype=np.uint32)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.uint64) * np.array([4, 5, 6], dtype=np.uint64)
        dnpa = dnp.array([1, 2, 3], dtype=np.uint64) * dnp.array([4, 5, 6], dtype=np.uint64)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.intp) * np.array([4, 5, 6], dtype=np.intp)
        dnpa = dnp.array([1, 2, 3], dtype=np.intp) * dnp.array([4, 5, 6], dtype=np.intp)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.intp) * pd.Series([4, 5, 6], dtype=np.intp)
        dnpa = dnp.array([1, 2, 3], dtype=np.intp) * orca.Series([4, 5, 6], dtype=np.intp)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.Series([4, 5, 6], dtype=np.intp) * np.array([1, 2, 3], dtype=np.intp)
        dnpa = orca.Series([4, 5, 6], dtype=np.intp) * dnp.array([1, 2, 3], dtype=np.intp)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1], dtype=np.intp) * pd.DataFrame({"a": [4, 5, 6]}, dtype=np.intp)
        dnpa = dnp.array([1], dtype=np.intp) * orca.DataFrame({"a": [4, 5, 6]}, dtype=np.intp)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.DataFrame({"a": [4, 5, 6]}, dtype=np.intp) * np.array([1], dtype=np.intp)
        dnpa = orca.DataFrame({"a": [4, 5, 6]}, dtype=np.intp) * dnp.array([1], dtype=np.intp)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1, 2, 3], dtype=np.uintp) * np.array([4, 5, 6], dtype=np.uintp)
        dnpa = dnp.array([1, 2, 3], dtype=np.uintp) * dnp.array([4, 5, 6], dtype=np.uintp)
        assert_array_equal(dnpa, npa)

        npa = np.array([12.0, np.nan, 15.0], dtype=np.float32) * np.array([14.7, 5.5, 6.8], dtype=np.float32)
        dnpa = dnp.array([12.0, np.nan, 15.0], dtype=np.float32) * dnp.array([14.7, 5.5, 6.8], dtype=np.float32)
        assert_array_equal(dnpa, npa)

        npa = np.array([12.0, np.nan, 15.0], dtype=np.float32) * pd.Series([14.7, 5.5, 6.8], dtype=np.float32)
        dnpa = dnp.array([12.0, np.nan, 15.0], dtype=np.float32) * orca.Series([14.7, 5.5, 6.8], dtype=np.float32)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.Series([14.7, 5.5, 6.8], dtype=np.float32) * np.array([12.0, np.nan, 15.0], dtype=np.float32)
        dnpa = orca.Series([14.7, 5.5, 6.8], dtype=np.float32) * dnp.array([12.0, np.nan, 15.0], dtype=np.float32)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([12.0], dtype=np.float32) * pd.DataFrame({'A': [14.7, np.nan, 6.8]}, dtype=np.float32)
        dnpa = dnp.array([12.0], dtype=np.float32) * orca.DataFrame({'A': [14.7, np.nan, 6.8]}, dtype=np.float32)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.DataFrame({'A': [14.7, np.nan, 6.8]}, dtype=np.float32) * np.array([12.0], dtype=np.float32)
        dnpa = orca.DataFrame({'A': [14.7, np.nan, 6.8]}, dtype=np.float32) * dnp.array([12.0], dtype=np.float32)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([12.0, np.nan, 15.0], dtype=np.float64) * np.array([14.7, 5.5, 6.8], dtype=np.float64)
        dnpa = dnp.array([12.0, np.nan, 15.0], dtype=np.float64) * dnp.array([14.7, 5.5, 6.8], dtype=np.float64)
        assert_array_equal(dnpa, npa)

        npa = np.array([12.0, np.nan, 15.0], dtype=np.float64) * pd.Series([14.7, 5.5, 6.8], dtype=np.float64)
        dnpa = dnp.array([12.0, np.nan, 15.0], dtype=np.float64) * orca.Series([14.7, 5.5, 6.8], dtype=np.float64)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.Series([14.7, 5.5, 6.8], dtype=np.float64) * np.array([12.0, np.nan, 15.0], dtype=np.float64)
        dnpa = orca.Series([14.7, 5.5, 6.8], dtype=np.float64) * dnp.array([12.0, np.nan, 15.0], dtype=np.float64)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([12.0], dtype=np.float64) * pd.DataFrame({'A': [14.7, np.nan, 6.8]}, dtype=np.float64)
        dnpa = dnp.array([12.0], dtype=np.float64) * orca.DataFrame({'A': [14.7, np.nan, 6.8]},
                                                                                  dtype=np.float64)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.DataFrame({'A': [14.7, np.nan, 6.8]}, dtype=np.float64) * np.array([12.0], dtype=np.float64)
        dnpa = orca.DataFrame({'A': [14.7, np.nan, 6.8]}, dtype=np.float64) * dnp.array([12.0],
                                                                                     dtype=np.float64)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([12.0 + 1j, np.nan, 15.0], dtype=np.complex64) * np.array([14.7, 5.5, 6.8 + 3j],
                                                                                 dtype=np.complex64)
        dnpa = dnp.array([12.0 + 1j, np.nan, 15.0], dtype=np.complex64) * dnp.array([14.7, 5.5, 6.8 + 3j],
                                                                                    dtype=np.complex64)
        assert_array_equal(dnpa, npa)

        npa = np.array([12.0 + 1j, np.nan, 15.0], dtype=np.complex128) * np.array([14.7, 5.5, 6.8 + 3j],
                                                                                  dtype=np.complex128)
        dnpa = dnp.array([12.0 + 1j, np.nan, 15.0], dtype=np.complex128) * dnp.array([14.7, 5.5, 6.8 + 3j],
                                                                                     dtype=np.complex128)
        assert_array_equal(dnpa, npa)

    def test_arithmetic_truediv(self):
        npa = np.array([1, 2, 3]) / np.array([4, 5, 6])
        dnpa = dnp.array([1, 2, 3]) / dnp.array([4, 5, 6])
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3]) / pd.Series([4, 5, 6], dtype=np.int8)
        dnpa = dnp.array([1, 2, 3]) / orca.Series([4, 5, 6], dtype=np.int8)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = pd.Series([4, 5, 6], dtype=np.int8) / np.array([1, 2, 3])
        dnpa = orca.Series([4, 5, 6], dtype=np.int8) / dnp.array([1, 2, 3])
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = np.array([1, 2, 3], dtype=np.int8) / pd.Series([4, 5, 6], dtype=np.int8)
        dnpa = dnp.array([1, 2, 3], dtype=np.int8) / orca.Series([4, 5, 6], dtype=np.int8)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.Series([4, 5, 6], dtype=np.int8) / np.array([1, 2, 3], dtype=np.int8)
        dnpa = orca.Series([4, 5, 6], dtype=np.int8) / dnp.array([1, 2, 3], dtype=np.int8)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1], dtype=np.int8) / pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int8)
        dnpa = dnp.array([1], dtype=np.int8) / orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int8)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int8) / np.array([1], dtype=np.int8)
        dnpa = orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int8) / dnp.array([1], dtype=np.int8)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1, 2, 3], dtype=np.int16) / np.array([4, 5, 6], dtype=np.int16)
        dnpa = dnp.array([1, 2, 3], dtype=np.int16) / dnp.array([4, 5, 6], dtype=np.int16)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.int16) / pd.Series([4, 5, 6], dtype=np.int16)
        dnpa = dnp.array([1, 2, 3], dtype=np.int16) / orca.Series([4, 5, 6], dtype=np.int16)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.Series([4, 5, 6], dtype=np.int16) / np.array([1, 2, 3], dtype=np.int16)
        dnpa = orca.Series([4, 5, 6], dtype=np.int16) / dnp.array([1, 2, 3], dtype=np.int16)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1], dtype=np.int16) / pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int16)
        dnpa = dnp.array([1], dtype=np.int16) / orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int16)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int16) / np.array([1], dtype=np.int16)
        dnpa = orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int16) / dnp.array([1], dtype=np.int16)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1, 2, 3], dtype=np.int32) / np.array([4, 5, 6], dtype=np.int32)
        dnpa = dnp.array([1, 2, 3], dtype=np.int32) / dnp.array([4, 5, 6], dtype=np.int32)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.int32) / pd.Series([4, 5, 6], dtype=np.int32)
        dnpa = dnp.array([1, 2, 3], dtype=np.int32) / orca.Series([4, 5, 6], dtype=np.int32)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.Series([4, 5, 6], dtype=np.int32) / np.array([1, 2, 3], dtype=np.int32)
        dnpa = orca.Series([4, 5, 6], dtype=np.int32) / dnp.array([1, 2, 3], dtype=np.int32)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1], dtype=np.int32) / pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int32)
        dnpa = dnp.array([1], dtype=np.int32) / orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int32)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int32) / np.array([1], dtype=np.int32)
        dnpa = orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int32) / dnp.array([1], dtype=np.int32)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1, 2, 3], dtype=np.int64) / np.array([4, 5, 6], dtype=np.int64)
        dnpa = dnp.array([1, 2, 3], dtype=np.int64) / dnp.array([4, 5, 6], dtype=np.int64)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.int64) / pd.Series([4, 5, 6], dtype=np.int64)
        dnpa = dnp.array([1, 2, 3], dtype=np.int64) / orca.Series([4, 5, 6], dtype=np.int64)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.Series([4, 5, 6], dtype=np.int64) / np.array([1, 2, 3], dtype=np.int64)
        dnpa = orca.Series([4, 5, 6], dtype=np.int64) / dnp.array([1, 2, 3], dtype=np.int64)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1], dtype=np.int64) / pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int64)
        dnpa = dnp.array([1], dtype=np.int64) / orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int64)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int64) / np.array([1], dtype=np.int64)
        dnpa = orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int64) / dnp.array([1], dtype=np.int64)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1, 2, 3], dtype=np.uint8) / np.array([4, 5, 6], dtype=np.uint8)
        dnpa = dnp.array([1, 2, 3], dtype=np.uint8) / dnp.array([4, 5, 6], dtype=np.uint8)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.uint16) / np.array([4, 5, 6], dtype=np.uint16)
        dnpa = dnp.array([1, 2, 3], dtype=np.uint16) / dnp.array([4, 5, 6], dtype=np.uint16)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.uint32) / np.array([4, 5, 6], dtype=np.uint32)
        dnpa = dnp.array([1, 2, 3], dtype=np.uint32) / dnp.array([4, 5, 6], dtype=np.uint32)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.uint64) / np.array([4, 5, 6], dtype=np.uint64)
        dnpa = dnp.array([1, 2, 3], dtype=np.uint64) / dnp.array([4, 5, 6], dtype=np.uint64)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.intp) / np.array([4, 5, 6], dtype=np.intp)
        dnpa = dnp.array([1, 2, 3], dtype=np.intp) / dnp.array([4, 5, 6], dtype=np.intp)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.intp) / pd.Series([4, 5, 6], dtype=np.intp)
        dnpa = dnp.array([1, 2, 3], dtype=np.intp) / orca.Series([4, 5, 6], dtype=np.intp)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.Series([4, 5, 6], dtype=np.intp) / np.array([1, 2, 3], dtype=np.intp)
        dnpa = orca.Series([4, 5, 6], dtype=np.intp) / dnp.array([1, 2, 3], dtype=np.intp)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1], dtype=np.intp) / pd.DataFrame({"a": [4, 5, 6]}, dtype=np.intp)
        dnpa = dnp.array([1], dtype=np.intp) / orca.DataFrame({"a": [4, 5, 6]}, dtype=np.intp)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.DataFrame({"a": [4, 5, 6]}, dtype=np.intp) / np.array([1], dtype=np.intp)
        dnpa = orca.DataFrame({"a": [4, 5, 6]}, dtype=np.intp) / dnp.array([1], dtype=np.intp)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1, 2, 3], dtype=np.uintp) / np.array([4, 5, 6], dtype=np.uintp)
        dnpa = dnp.array([1, 2, 3], dtype=np.uintp) / dnp.array([4, 5, 6], dtype=np.uintp)
        assert_array_equal(dnpa, npa)

        npa = np.array([12.0, np.nan, 15.0], dtype=np.float32) / np.array([14.7, 5.5, 6.8], dtype=np.float32)
        dnpa = dnp.array([12.0, np.nan, 15.0], dtype=np.float32) / dnp.array([14.7, 5.5, 6.8], dtype=np.float32)
        assert_array_equal(dnpa, npa)

        npa = np.array([12.0, np.nan, 15.0], dtype=np.float32) / pd.Series([14.7, 5.5, 6.8], dtype=np.float32)
        dnpa = dnp.array([12.0, np.nan, 15.0], dtype=np.float32) / orca.Series([14.7, 5.5, 6.8], dtype=np.float32)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.Series([14.7, 5.5, 6.8], dtype=np.float32) / np.array([12.0, np.nan, 15.0], dtype=np.float32)
        dnpa = orca.Series([14.7, 5.5, 6.8], dtype=np.float32) / dnp.array([12.0, np.nan, 15.0], dtype=np.float32)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([12.0], dtype=np.float32) / pd.DataFrame({'A': [14.7, np.nan, 6.8]}, dtype=np.float32)
        dnpa = dnp.array([12.0], dtype=np.float32) / orca.DataFrame({'A': [14.7, np.nan, 6.8]}, dtype=np.float32)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.DataFrame({'A': [14.7, np.nan, 6.8]}, dtype=np.float32) / np.array([12.0], dtype=np.float32)
        dnpa = orca.DataFrame({'A': [14.7, np.nan, 6.8]}, dtype=np.float32) / dnp.array([12.0], dtype=np.float32)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([12.0, np.nan, 15.0], dtype=np.float64) / np.array([14.7, 5.5, 6.8], dtype=np.float64)
        dnpa = dnp.array([12.0, np.nan, 15.0], dtype=np.float64) / dnp.array([14.7, 5.5, 6.8], dtype=np.float64)
        assert_array_equal(dnpa, npa)

        npa = np.array([12.0, np.nan, 15.0], dtype=np.float64) / pd.Series([14.7, 5.5, 6.8], dtype=np.float64)
        dnpa = dnp.array([12.0, np.nan, 15.0], dtype=np.float64) / orca.Series([14.7, 5.5, 6.8], dtype=np.float64)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.Series([14.7, 5.5, 6.8], dtype=np.float64) / np.array([12.0, np.nan, 15.0], dtype=np.float64)
        dnpa = orca.Series([14.7, 5.5, 6.8], dtype=np.float64) / dnp.array([12.0, np.nan, 15.0], dtype=np.float64)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([12.0], dtype=np.float64) / pd.DataFrame({'A': [14.7, np.nan, 6.8]}, dtype=np.float64)
        dnpa = dnp.array([12.0], dtype=np.float64) / orca.DataFrame({'A': [14.7, np.nan, 6.8]},
                                                                                  dtype=np.float64)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.DataFrame({'A': [14.7, np.nan, 6.8]}, dtype=np.float64) / np.array([12.0], dtype=np.float64)
        dnpa = orca.DataFrame({'A': [14.7, np.nan, 6.8]}, dtype=np.float64) / dnp.array([12.0],
                                                                                     dtype=np.float64)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([12.0 + 1j, np.nan, 15.0], dtype=np.complex64) / np.array([14.7, 5.5, 6.8 + 3j],
                                                                                 dtype=np.complex64)
        dnpa = dnp.array([12.0 + 1j, np.nan, 15.0], dtype=np.complex64) / dnp.array([14.7, 5.5, 6.8 + 3j],
                                                                                    dtype=np.complex64)
        assert_array_equal(dnpa, npa)

        npa = np.array([12.0 + 1j, np.nan, 15.0], dtype=np.complex128) / np.array([14.7, 5.5, 6.8 + 3j],
                                                                                  dtype=np.complex128)
        dnpa = dnp.array([12.0 + 1j, np.nan, 15.0], dtype=np.complex128) / dnp.array([14.7, 5.5, 6.8 + 3j],
                                                                                     dtype=np.complex128)
        assert_array_equal(dnpa, npa)

    def test_arithmetic_floordiv(self):
        npa = np.array([1, 2, 3]) // np.array([4, 5, 6])
        dnpa = dnp.array([1, 2, 3]) // dnp.array([4, 5, 6])
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3]) // pd.Series([4, 5, 6], dtype=np.int8)
        dnpa = dnp.array([1, 2, 3]) // orca.Series([4, 5, 6], dtype=np.int8)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = pd.Series([4, 5, 6], dtype=np.int8) // np.array([1, 2, 3])
        dnpa = orca.Series([4, 5, 6], dtype=np.int8) // dnp.array([1, 2, 3])
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = np.array([1, 2, 3], dtype=np.int8) // pd.Series([4, 5, 6], dtype=np.int8)
        dnpa = dnp.array([1, 2, 3], dtype=np.int8) // orca.Series([4, 5, 6], dtype=np.int8)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.Series([4, 5, 6], dtype=np.int8) // np.array([1, 2, 3], dtype=np.int8)
        dnpa = orca.Series([4, 5, 6], dtype=np.int8) // dnp.array([1, 2, 3], dtype=np.int8)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1], dtype=np.int8) // pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int8)
        dnpa = dnp.array([1], dtype=np.int8) // orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int8)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int8) // np.array([1], dtype=np.int8)
        dnpa = orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int8) // dnp.array([1], dtype=np.int8)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1, 2, 3], dtype=np.int16) // np.array([4, 5, 6], dtype=np.int16)
        dnpa = dnp.array([1, 2, 3], dtype=np.int16) // dnp.array([4, 5, 6], dtype=np.int16)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.int16) // pd.Series([4, 5, 6], dtype=np.int16)
        dnpa = dnp.array([1, 2, 3], dtype=np.int16) // orca.Series([4, 5, 6], dtype=np.int16)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.Series([4, 5, 6], dtype=np.int16) // np.array([1, 2, 3], dtype=np.int16)
        dnpa = orca.Series([4, 5, 6], dtype=np.int16) // dnp.array([1, 2, 3], dtype=np.int16)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1], dtype=np.int16) // pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int16)
        dnpa = dnp.array([1], dtype=np.int16) // orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int16)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int16) // np.array([1], dtype=np.int16)
        dnpa = orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int16) // dnp.array([1], dtype=np.int16)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1, 2, 3], dtype=np.int32) // np.array([4, 5, 6], dtype=np.int32)
        dnpa = dnp.array([1, 2, 3], dtype=np.int32) // dnp.array([4, 5, 6], dtype=np.int32)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.int32) // pd.Series([4, 5, 6], dtype=np.int32)
        dnpa = dnp.array([1, 2, 3], dtype=np.int32) // orca.Series([4, 5, 6], dtype=np.int32)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.Series([4, 5, 6], dtype=np.int32) // np.array([1, 2, 3], dtype=np.int32)
        dnpa = orca.Series([4, 5, 6], dtype=np.int32) // dnp.array([1, 2, 3], dtype=np.int32)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1], dtype=np.int32) // pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int32)
        dnpa = dnp.array([1], dtype=np.int32) // orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int32)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int32) // np.array([1], dtype=np.int32)
        dnpa = orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int32) // dnp.array([1], dtype=np.int32)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1, 2, 3], dtype=np.int64) // np.array([4, 5, 6], dtype=np.int64)
        dnpa = dnp.array([1, 2, 3], dtype=np.int64) // dnp.array([4, 5, 6], dtype=np.int64)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.int64) // pd.Series([4, 5, 6], dtype=np.int64)
        dnpa = dnp.array([1, 2, 3], dtype=np.int64) // orca.Series([4, 5, 6], dtype=np.int64)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.Series([4, 5, 6], dtype=np.int64) // np.array([1, 2, 3], dtype=np.int64)
        dnpa = orca.Series([4, 5, 6], dtype=np.int64) // dnp.array([1, 2, 3], dtype=np.int64)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1], dtype=np.int64) // pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int64)
        dnpa = dnp.array([1], dtype=np.int64) // orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int64)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int64) // np.array([1], dtype=np.int64)
        dnpa = orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int64) // dnp.array([1], dtype=np.int64)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1, 2, 3], dtype=np.uint8) // np.array([4, 5, 6], dtype=np.uint8)
        dnpa = dnp.array([1, 2, 3], dtype=np.uint8) // dnp.array([4, 5, 6], dtype=np.uint8)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.uint16) // np.array([4, 5, 6], dtype=np.uint16)
        dnpa = dnp.array([1, 2, 3], dtype=np.uint16) // dnp.array([4, 5, 6], dtype=np.uint16)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.uint32) // np.array([4, 5, 6], dtype=np.uint32)
        dnpa = dnp.array([1, 2, 3], dtype=np.uint32) // dnp.array([4, 5, 6], dtype=np.uint32)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.uint64) // np.array([4, 5, 6], dtype=np.uint64)
        dnpa = dnp.array([1, 2, 3], dtype=np.uint64) // dnp.array([4, 5, 6], dtype=np.uint64)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.intp) // np.array([4, 5, 6], dtype=np.intp)
        dnpa = dnp.array([1, 2, 3], dtype=np.intp) // dnp.array([4, 5, 6], dtype=np.intp)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.intp) // pd.Series([4, 5, 6], dtype=np.intp)
        dnpa = dnp.array([1, 2, 3], dtype=np.intp) // orca.Series([4, 5, 6], dtype=np.intp)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.Series([4, 5, 6], dtype=np.intp) // np.array([1, 2, 3], dtype=np.intp)
        dnpa = orca.Series([4, 5, 6], dtype=np.intp) // dnp.array([1, 2, 3], dtype=np.intp)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1], dtype=np.intp) // pd.DataFrame({"a": [4, 5, 6]}, dtype=np.intp)
        dnpa = dnp.array([1], dtype=np.intp) // orca.DataFrame({"a": [4, 5, 6]}, dtype=np.intp)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.DataFrame({"a": [4, 5, 6]}, dtype=np.intp) // np.array([1], dtype=np.intp)
        dnpa = orca.DataFrame({"a": [4, 5, 6]}, dtype=np.intp) // dnp.array([1], dtype=np.intp)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1, 2, 3], dtype=np.uintp) // np.array([4, 5, 6], dtype=np.uintp)
        dnpa = dnp.array([1, 2, 3], dtype=np.uintp) // dnp.array([4, 5, 6], dtype=np.uintp)
        assert_array_equal(dnpa, npa)

        npa = np.array([12.0, np.nan, 15.0], dtype=np.float32) // np.array([14.7, 5.5, 6.8], dtype=np.float32)
        dnpa = dnp.array([12.0, np.nan, 15.0], dtype=np.float32) // dnp.array([14.7, 5.5, 6.8], dtype=np.float32)
        assert_array_equal(dnpa, npa)

        npa = np.array([12.0, np.nan, 15.0], dtype=np.float32) // pd.Series([14.7, 5.5, 6.8], dtype=np.float32)
        dnpa = dnp.array([12.0, np.nan, 15.0], dtype=np.float32) // orca.Series([14.7, 5.5, 6.8], dtype=np.float32)
        # TODO： floorfiv bug
        # assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.Series([14.7, 5.5, 6.8], dtype=np.float32) // np.array([12.0, np.nan, 15.0], dtype=np.float32)
        dnpa = orca.Series([14.7, 5.5, 6.8], dtype=np.float32) // dnp.array([12.0, np.nan, 15.0], dtype=np.float32)
        # TODO： floorfiv bug
        # assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([12.0], dtype=np.float32) // pd.DataFrame({'A': [14.7, np.nan, 6.8]}, dtype=np.float32)
        dnpa = dnp.array([12.0], dtype=np.float32) // orca.DataFrame({'A': [14.7, np.nan, 6.8]}, dtype=np.float32)
        # TODO： floorfiv bug
        # assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.DataFrame({'A': [14.7, np.nan, 6.8]}, dtype=np.float32) // np.array([12.0], dtype=np.float32)
        dnpa = orca.DataFrame({'A': [14.7, np.nan, 6.8]}, dtype=np.float32) // dnp.array([12.0], dtype=np.float32)
        # TODO： floorfiv bug
        # assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([12.0, np.nan, 15.0], dtype=np.float64) // np.array([14.7, 5.5, 6.8], dtype=np.float64)
        dnpa = dnp.array([12.0, np.nan, 15.0], dtype=np.float64) // dnp.array([14.7, 5.5, 6.8], dtype=np.float64)
        assert_array_equal(dnpa, npa)

        npa = np.array([12.0, np.nan, 15.0], dtype=np.float64) // pd.Series([14.7, 5.5, 6.8], dtype=np.float64)
        dnpa = dnp.array([12.0, np.nan, 15.0], dtype=np.float64) // orca.Series([14.7, 5.5, 6.8], dtype=np.float64)
        # TODO： floorfiv bug
        # assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.Series([14.7, 5.5, 6.8], dtype=np.float64) // np.array([12.0, np.nan, 15.0], dtype=np.float64)
        dnpa = orca.Series([14.7, 5.5, 6.8], dtype=np.float64) // dnp.array([12.0, np.nan, 15.0], dtype=np.float64)
        # TODO： floorfiv bug
        # assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([12.0], dtype=np.float64) // pd.DataFrame({'A': [14.7, np.nan, 6.8]}, dtype=np.float64)
        dnpa = dnp.array([12.0], dtype=np.float64) // orca.DataFrame({'A': [14.7, np.nan, 6.8]},
                                                                     dtype=np.float64)
        # TODO： floorfiv bug
        # assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.DataFrame({'A': [14.7, np.nan, 6.8]}, dtype=np.float64) // np.array([12.0], dtype=np.float64)
        dnpa = orca.DataFrame({'A': [14.7, np.nan, 6.8]}, dtype=np.float64) // dnp.array([12.0],
                                                                                         dtype=np.float64)
        # TODO： floorfiv bug
        # assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([12.0 + 1j, np.nan, 15.0], dtype=np.complex64) // np.array([14.7, 5.5, 6.8 + 3j],
                                                                                   dtype=np.complex64)
        dnpa = dnp.array([12.0 + 1j, np.nan, 15.0], dtype=np.complex64) // dnp.array([14.7, 5.5, 6.8 + 3j],
                                                                                      dtype=np.complex64)
        assert_array_equal(dnpa, npa)

        npa = np.array([12.0 + 1j, np.nan, 15.0], dtype=np.complex128) // np.array([14.7, 5.5, 6.8 + 3j],
                                                                                    dtype=np.complex128)
        dnpa = dnp.array([12.0 + 1j, np.nan, 15.0], dtype=np.complex128) // dnp.array([14.7, 5.5, 6.8 + 3j],
                                                                                       dtype=np.complex128)
        assert_array_equal(dnpa, npa)

    def test_arithmetic_mod(self):
        npa = np.array([1, 2, 3]) % np.array([4, 5, 6])
        dnpa = dnp.array([1, 2, 3]) % dnp.array([4, 5, 6])
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3]) % pd.Series([4, 5, 6], dtype=np.int8)
        dnpa = dnp.array([1, 2, 3]) % orca.Series([4, 5, 6], dtype=np.int8)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = pd.Series([4, 5, 6], dtype=np.int8) % np.array([1, 2, 3])
        dnpa = orca.Series([4, 5, 6], dtype=np.int8) % dnp.array([1, 2, 3])
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = np.array([1, 2, 3], dtype=np.int8) % pd.Series([4, 5, 6], dtype=np.int8)
        dnpa = dnp.array([1, 2, 3], dtype=np.int8) % orca.Series([4, 5, 6], dtype=np.int8)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.Series([4, 5, 6], dtype=np.int8) % np.array([1, 2, 3], dtype=np.int8)
        dnpa = orca.Series([4, 5, 6], dtype=np.int8) % dnp.array([1, 2, 3], dtype=np.int8)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1], dtype=np.int8) % pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int8)
        dnpa = dnp.array([1], dtype=np.int8) % orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int8)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int8) % np.array([1], dtype=np.int8)
        dnpa = orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int8) % dnp.array([1], dtype=np.int8)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1, 2, 3], dtype=np.int16) % np.array([4, 5, 6], dtype=np.int16)
        dnpa = dnp.array([1, 2, 3], dtype=np.int16) % dnp.array([4, 5, 6], dtype=np.int16)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.int16) % pd.Series([4, 5, 6], dtype=np.int16)
        dnpa = dnp.array([1, 2, 3], dtype=np.int16) % orca.Series([4, 5, 6], dtype=np.int16)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.Series([4, 5, 6], dtype=np.int16) % np.array([1, 2, 3], dtype=np.int16)
        dnpa = orca.Series([4, 5, 6], dtype=np.int16) % dnp.array([1, 2, 3], dtype=np.int16)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1], dtype=np.int16) % pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int16)
        dnpa = dnp.array([1], dtype=np.int16) % orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int16)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int16) % np.array([1], dtype=np.int16)
        dnpa = orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int16) % dnp.array([1], dtype=np.int16)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1, 2, 3], dtype=np.int32) % np.array([4, 5, 6], dtype=np.int32)
        dnpa = dnp.array([1, 2, 3], dtype=np.int32) % dnp.array([4, 5, 6], dtype=np.int32)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.int32) % pd.Series([4, 5, 6], dtype=np.int32)
        dnpa = dnp.array([1, 2, 3], dtype=np.int32) % orca.Series([4, 5, 6], dtype=np.int32)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.Series([4, 5, 6], dtype=np.int32) % np.array([1, 2, 3], dtype=np.int32)
        dnpa = orca.Series([4, 5, 6], dtype=np.int32) % dnp.array([1, 2, 3], dtype=np.int32)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1], dtype=np.int32) % pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int32)
        dnpa = dnp.array([1], dtype=np.int32) % orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int32)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int32) % np.array([1], dtype=np.int32)
        dnpa = orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int32) % dnp.array([1], dtype=np.int32)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1, 2, 3], dtype=np.int64) % np.array([4, 5, 6], dtype=np.int64)
        dnpa = dnp.array([1, 2, 3], dtype=np.int64) % dnp.array([4, 5, 6], dtype=np.int64)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.int64) % pd.Series([4, 5, 6], dtype=np.int64)
        dnpa = dnp.array([1, 2, 3], dtype=np.int64) % orca.Series([4, 5, 6], dtype=np.int64)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.Series([4, 5, 6], dtype=np.int64) % np.array([1, 2, 3], dtype=np.int64)
        dnpa = orca.Series([4, 5, 6], dtype=np.int64) % dnp.array([1, 2, 3], dtype=np.int64)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1], dtype=np.int64) % pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int64)
        dnpa = dnp.array([1], dtype=np.int64) % orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int64)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int64) % np.array([1], dtype=np.int64)
        dnpa = orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int64) % dnp.array([1], dtype=np.int64)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1, 2, 3], dtype=np.uint8) % np.array([4, 5, 6], dtype=np.uint8)
        dnpa = dnp.array([1, 2, 3], dtype=np.uint8) % dnp.array([4, 5, 6], dtype=np.uint8)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.uint16) % np.array([4, 5, 6], dtype=np.uint16)
        dnpa = dnp.array([1, 2, 3], dtype=np.uint16) % dnp.array([4, 5, 6], dtype=np.uint16)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.uint32) % np.array([4, 5, 6], dtype=np.uint32)
        dnpa = dnp.array([1, 2, 3], dtype=np.uint32) % dnp.array([4, 5, 6], dtype=np.uint32)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.uint64) % np.array([4, 5, 6], dtype=np.uint64)
        dnpa = dnp.array([1, 2, 3], dtype=np.uint64) % dnp.array([4, 5, 6], dtype=np.uint64)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.intp) % np.array([4, 5, 6], dtype=np.intp)
        dnpa = dnp.array([1, 2, 3], dtype=np.intp) % dnp.array([4, 5, 6], dtype=np.intp)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.intp) % pd.Series([4, 5, 6], dtype=np.intp)
        dnpa = dnp.array([1, 2, 3], dtype=np.intp) % orca.Series([4, 5, 6], dtype=np.intp)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.Series([4, 5, 6], dtype=np.intp) % np.array([1, 2, 3], dtype=np.intp)
        dnpa = orca.Series([4, 5, 6], dtype=np.intp) % dnp.array([1, 2, 3], dtype=np.intp)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1], dtype=np.intp) % pd.DataFrame({"a": [4, 5, 6]}, dtype=np.intp)
        dnpa = dnp.array([1], dtype=np.intp) % orca.DataFrame({"a": [4, 5, 6]}, dtype=np.intp)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.DataFrame({"a": [4, 5, 6]}, dtype=np.intp) % np.array([1], dtype=np.intp)
        dnpa = orca.DataFrame({"a": [4, 5, 6]}, dtype=np.intp) % dnp.array([1], dtype=np.intp)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1, 2, 3], dtype=np.uintp) % np.array([4, 5, 6], dtype=np.uintp)
        dnpa = dnp.array([1, 2, 3], dtype=np.uintp) % dnp.array([4, 5, 6], dtype=np.uintp)
        assert_array_equal(dnpa, npa)

        npa = np.array([12.0, np.nan, 15.0], dtype=np.float32) % np.array([14.7, 5.5, 6.8], dtype=np.float32)
        dnpa = dnp.array([12.0, np.nan, 15.0], dtype=np.float32) % dnp.array([14.7, 5.5, 6.8], dtype=np.float32)
        assert_array_equal(dnpa, npa)

        npa = np.array([12.0, np.nan, 15.0], dtype=np.float32) % pd.Series([14.7, 5.5, 6.8], dtype=np.float32)
        dnpa = dnp.array([12.0, np.nan, 15.0], dtype=np.float32) % orca.Series([14.7, 5.5, 6.8], dtype=np.float32)
        # TODO: mod bug
        # assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.Series([14.7, 5.5, 6.8], dtype=np.float32) % np.array([12.0, np.nan, 15.0], dtype=np.float32)
        dnpa = orca.Series([14.7, 5.5, 6.8], dtype=np.float32) % dnp.array([12.0, np.nan, 15.0], dtype=np.float32)
        # TODO: mod bug
        # assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([12.0], dtype=np.float32) % pd.DataFrame({'A': [14.7, np.nan, 6.8]}, dtype=np.float32)
        dnpa = dnp.array([12.0], dtype=np.float32) % orca.DataFrame({'A': [14.7, np.nan, 6.8]}, dtype=np.float32)
        # TODO: mod bug
        # assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.DataFrame({'A': [14.7, np.nan, 6.8]}, dtype=np.float32) % np.array([12.0], dtype=np.float32)
        dnpa = orca.DataFrame({'A': [14.7, np.nan, 6.8]}, dtype=np.float32) % dnp.array([12.0], dtype=np.float32)
        # TODO: mod bug
        # assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([12.0, np.nan, 15.0], dtype=np.float64) % np.array([14.7, 5.5, 6.8], dtype=np.float64)
        dnpa = dnp.array([12.0, np.nan, 15.0], dtype=np.float64) % dnp.array([14.7, 5.5, 6.8], dtype=np.float64)
        assert_array_equal(dnpa, npa)

        npa = np.array([12.0, np.nan, 15.0], dtype=np.float64) % pd.Series([14.7, 5.5, 6.8], dtype=np.float64)
        dnpa = dnp.array([12.0, np.nan, 15.0], dtype=np.float64) % orca.Series([14.7, 5.5, 6.8], dtype=np.float64)
        # TODO: mod bug
        # assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.Series([14.7, 5.5, 6.8], dtype=np.float64) % np.array([12.0, np.nan, 15.0], dtype=np.float64)
        dnpa = orca.Series([14.7, 5.5, 6.8], dtype=np.float64) % dnp.array([12.0, np.nan, 15.0], dtype=np.float64)
        # TODO: mod bug
        # assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([12.0], dtype=np.float64) % pd.DataFrame({'A': [14.7, np.nan, 6.8]}, dtype=np.float64)
        dnpa = dnp.array([12.0], dtype=np.float64) % orca.DataFrame({'A': [14.7, np.nan, 6.8]},
                                                                    dtype=np.float64)
        # TODO: mod bug
        # assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.DataFrame({'A': [14.7, np.nan, 6.8]}, dtype=np.float64) % np.array([12.0], dtype=np.float64)
        dnpa = orca.DataFrame({'A': [14.7, np.nan, 6.8]}, dtype=np.float64) % dnp.array([12.0],
                                                                                        dtype=np.float64)
        # TODO: mod bug
        # assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

    def test_arithmetic_pow(self):
        npa = np.array([1, 2, 3]) ** np.array([4, 5, 6])
        dnpa = dnp.array([1, 2, 3]) ** dnp.array([4, 5, 6])
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3]) ** pd.Series([4, 5, 6], dtype=np.int8)
        dnpa = dnp.array([1, 2, 3]) ** orca.Series([4, 5, 6], dtype=np.int8)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.Series([4, 5, 6], dtype=np.int8) ** np.array([1, 2, 3])
        dnpa = orca.Series([4, 5, 6], dtype=np.int8) ** dnp.array([1, 2, 3])
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1, 2, 3], dtype=np.int8) ** pd.Series([4, 5, 6], dtype=np.int8)
        dnpa = dnp.array([1, 2, 3], dtype=np.int8) ** orca.Series([4, 5, 6], dtype=np.int8)
        # TODO: pow bug
        # assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.Series([4, 5, 6], dtype=np.int8) ** np.array([1, 2, 3], dtype=np.int8)
        dnpa = orca.Series([4, 5, 6], dtype=np.int8) ** dnp.array([1, 2, 3], dtype=np.int8)
        # TODO: pow bug
        # assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1], dtype=np.int8) ** pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int8)
        dnpa = dnp.array([1], dtype=np.int8) ** orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int8)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int8) ** np.array([1], dtype=np.int8)
        dnpa = orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int8) ** dnp.array([1], dtype=np.int8)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1, 2, 3], dtype=np.int16) ** np.array([4, 5, 6], dtype=np.int16)
        dnpa = dnp.array([1, 2, 3], dtype=np.int16) ** dnp.array([4, 5, 6], dtype=np.int16)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.int16) ** pd.Series([4, 5, 6], dtype=np.int16)
        dnpa = dnp.array([1, 2, 3], dtype=np.int16) ** orca.Series([4, 5, 6], dtype=np.int16)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.Series([4, 5, 6], dtype=np.int16) ** np.array([1, 2, 3], dtype=np.int16)
        dnpa = orca.Series([4, 5, 6], dtype=np.int16) ** dnp.array([1, 2, 3], dtype=np.int16)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1], dtype=np.int16) ** pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int16)
        dnpa = dnp.array([1], dtype=np.int16) ** orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int16)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int16) ** np.array([1], dtype=np.int16)
        dnpa = orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int16) ** dnp.array([1], dtype=np.int16)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1, 2, 3], dtype=np.int32) ** np.array([4, 5, 6], dtype=np.int32)
        dnpa = dnp.array([1, 2, 3], dtype=np.int32) ** dnp.array([4, 5, 6], dtype=np.int32)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.int32) ** pd.Series([4, 5, 6], dtype=np.int32)
        dnpa = dnp.array([1, 2, 3], dtype=np.int32) ** orca.Series([4, 5, 6], dtype=np.int32)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.Series([4, 5, 6], dtype=np.int32) ** np.array([1, 2, 3], dtype=np.int32)
        dnpa = orca.Series([4, 5, 6], dtype=np.int32) ** dnp.array([1, 2, 3], dtype=np.int32)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1], dtype=np.int32) ** pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int32)
        dnpa = dnp.array([1], dtype=np.int32) ** orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int32)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int32) ** np.array([1], dtype=np.int32)
        dnpa = orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int32) ** dnp.array([1], dtype=np.int32)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1, 2, 3], dtype=np.int64) ** np.array([4, 5, 6], dtype=np.int64)
        dnpa = dnp.array([1, 2, 3], dtype=np.int64) ** dnp.array([4, 5, 6], dtype=np.int64)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.int64) ** pd.Series([4, 5, 6], dtype=np.int64)
        dnpa = dnp.array([1, 2, 3], dtype=np.int64) ** orca.Series([4, 5, 6], dtype=np.int64)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.Series([4, 5, 6], dtype=np.int64) ** np.array([1, 2, 3], dtype=np.int64)
        dnpa = orca.Series([4, 5, 6], dtype=np.int64) ** dnp.array([1, 2, 3], dtype=np.int64)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1], dtype=np.int64) ** pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int64)
        dnpa = dnp.array([1], dtype=np.int64) ** orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int64)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.DataFrame({"a": [4, 5, 6]}, dtype=np.int64) ** np.array([1], dtype=np.int64)
        dnpa = orca.DataFrame({"a": [4, 5, 6]}, dtype=np.int64) ** dnp.array([1], dtype=np.int64)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1, 2, 3], dtype=np.uint8) ** np.array([4, 5, 6], dtype=np.uint8)
        dnpa = dnp.array([1, 2, 3], dtype=np.uint8) ** dnp.array([4, 5, 6], dtype=np.uint8)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.uint16) ** np.array([4, 5, 6], dtype=np.uint16)
        dnpa = dnp.array([1, 2, 3], dtype=np.uint16) ** dnp.array([4, 5, 6], dtype=np.uint16)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.uint32) ** np.array([4, 5, 6], dtype=np.uint32)
        dnpa = dnp.array([1, 2, 3], dtype=np.uint32) ** dnp.array([4, 5, 6], dtype=np.uint32)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.uint64) ** np.array([4, 5, 6], dtype=np.uint64)
        dnpa = dnp.array([1, 2, 3], dtype=np.uint64) ** dnp.array([4, 5, 6], dtype=np.uint64)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.intp) ** np.array([4, 5, 6], dtype=np.intp)
        dnpa = dnp.array([1, 2, 3], dtype=np.intp) ** dnp.array([4, 5, 6], dtype=np.intp)
        assert_array_equal(dnpa, npa)

        npa = np.array([1, 2, 3], dtype=np.intp) ** pd.Series([4, 5, 6], dtype=np.intp)
        dnpa = dnp.array([1, 2, 3], dtype=np.intp) ** orca.Series([4, 5, 6], dtype=np.intp)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.Series([4, 5, 6], dtype=np.intp) ** np.array([1, 2, 3], dtype=np.intp)
        dnpa = orca.Series([4, 5, 6], dtype=np.intp) ** dnp.array([1, 2, 3], dtype=np.intp)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1], dtype=np.intp) ** pd.DataFrame({"a": [4, 5, 6]}, dtype=np.intp)
        dnpa = dnp.array([1], dtype=np.intp) ** orca.DataFrame({"a": [4, 5, 6]}, dtype=np.intp)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.DataFrame({"a": [4, 5, 6]}, dtype=np.intp) ** np.array([1], dtype=np.intp)
        dnpa = orca.DataFrame({"a": [4, 5, 6]}, dtype=np.intp) ** dnp.array([1], dtype=np.intp)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([1, 2, 3], dtype=np.uintp) ** np.array([4, 5, 6], dtype=np.uintp)
        dnpa = dnp.array([1, 2, 3], dtype=np.uintp) ** dnp.array([4, 5, 6], dtype=np.uintp)
        assert_array_equal(dnpa, npa)

        npa = np.array([12.0, np.nan, 15.0], dtype=np.float32) ** np.array([14.7, 5.5, 6.8], dtype=np.float32)
        dnpa = dnp.array([12.0, np.nan, 15.0], dtype=np.float32) ** dnp.array([14.7, 5.5, 6.8], dtype=np.float32)
        assert_array_equal(dnpa, npa)

        npa = np.array([12.0, np.nan, 15.0], dtype=np.float32) ** pd.Series([14.7, 5.5, 6.8], dtype=np.float32)
        dnpa = dnp.array([12.0, np.nan, 15.0], dtype=np.float32) ** orca.Series([14.7, 5.5, 6.8], dtype=np.float32)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.Series([14.7, 5.5, 6.8], dtype=np.float32) ** np.array([12.0, np.nan, 15.0], dtype=np.float32)
        dnpa = orca.Series([14.7, 5.5, 6.8], dtype=np.float32) ** dnp.array([12.0, np.nan, 15.0], dtype=np.float32)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([12.0], dtype=np.float32) ** pd.DataFrame({'A': [14.7, np.nan, 6.8]}, dtype=np.float32)
        dnpa = dnp.array([12.0], dtype=np.float32) ** orca.DataFrame({'A': [14.7, np.nan, 6.8]}, dtype=np.float32)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.DataFrame({'A': [14.7, np.nan, 6.8]}, dtype=np.float32) ** np.array([12.0], dtype=np.float32)
        dnpa = orca.DataFrame({'A': [14.7, np.nan, 6.8]}, dtype=np.float32) ** dnp.array([12.0], dtype=np.float32)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([12.0, np.nan, 15.0], dtype=np.float64) ** np.array([14.7, 5.5, 6.8], dtype=np.float64)
        dnpa = dnp.array([12.0, np.nan, 15.0], dtype=np.float64) ** dnp.array([14.7, 5.5, 6.8], dtype=np.float64)
        assert_array_equal(dnpa, npa)

        npa = np.array([12.0, np.nan, 15.0], dtype=np.float64) ** pd.Series([14.7, 5.5, 6.8], dtype=np.float64)
        dnpa = dnp.array([12.0, np.nan, 15.0], dtype=np.float64) ** orca.Series([14.7, 5.5, 6.8], dtype=np.float64)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.Series([14.7, 5.5, 6.8], dtype=np.float64) ** np.array([12.0, np.nan, 15.0], dtype=np.float64)
        dnpa = orca.Series([14.7, 5.5, 6.8], dtype=np.float64) ** dnp.array([12.0, np.nan, 15.0], dtype=np.float64)
        assert_series_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([12.0], dtype=np.float64) ** pd.DataFrame({'A': [14.7, np.nan, 6.8]}, dtype=np.float64)
        dnpa = dnp.array([12.0], dtype=np.float64) ** orca.DataFrame({'A': [14.7, np.nan, 6.8]},
                                                                     dtype=np.float64)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = pd.DataFrame({'A': [14.7, np.nan, 6.8]}, dtype=np.float64) ** np.array([12.0], dtype=np.float64)
        dnpa = orca.DataFrame({'A': [14.7, np.nan, 6.8]}, dtype=np.float64) ** dnp.array([12.0],
                                                                                         dtype=np.float64)
        assert_frame_equal(dnpa.to_pandas(), npa, check_dtype=False)

        npa = np.array([12.0 + 1j, np.nan, 15.0], dtype=np.complex64) ** np.array([14.7, 5.5, 6.8 + 3j],
                                                                                  dtype=np.complex64)
        dnpa = dnp.array([12.0 + 1j, np.nan, 15.0], dtype=np.complex64) ** dnp.array([14.7, 5.5, 6.8 + 3j],
                                                                                     dtype=np.complex64)
        assert_array_equal(dnpa, npa)

        npa = np.array([12.0 + 1j, np.nan, 15.0], dtype=np.complex128) ** np.array([14.7, 5.5, 6.8 + 3j],
                                                                                   dtype=np.complex128)
        dnpa = dnp.array([12.0 + 1j, np.nan, 15.0], dtype=np.complex128) ** dnp.array([14.7, 5.5, 6.8 + 3j],
                                                                                      dtype=np.complex128)
        assert_array_equal(dnpa, npa)


if __name__ == '__main__':
    unittest.main()