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

    def test_logic_operation_greaterrthan(self):
        npa = np.array([6, 50, 12]) > np.array([6, 20, 18])
        dnpa = dnp.array([6, 50, 12]) > dnp.array([6, 20, 18])
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12]) > pd.Series([6, 20, 18], dtype=np.int8)
        dnpa = dnp.array([6, 50, 12]) > orca.Series([6, 20, 18], dtype=dnp.int8)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = pd.Series([6, 20, 18], dtype=np.int8) > np.array([6, 50, 12])
        dnpa = orca.Series([6, 20, 18], dtype=dnp.int8) > dnp.array([6, 50, 12])
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = np.array([6, 50, 12], dtype=np.int8) > np.array([6, 20, 18], dtype=np.int8)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.int8) > dnp.array([6, 20, 18], dtype=dnp.int8)
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12], dtype=np.int8) > pd.Series([6, 20, 18], dtype=np.int8)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.int8) > orca.Series([6, 20, 18], dtype=dnp.int8)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = pd.Series([6, 50, 12], dtype=np.int8) > np.array([6, 20, 18], dtype=np.int8)
        dnpa = orca.Series([6, 50, 12], dtype=dnp.int8) > dnp.array([6, 20, 18], dtype=dnp.int8)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = np.array([6], dtype=np.int8) > pd.DataFrame({"a": [6, 5, 18]}, dtype=np.int8)
        dnpa = dnp.array([6], dtype=dnp.int8) > orca.DataFrame({"a": [6, 5, 18]}, dtype=dnp.int8)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = pd.DataFrame({"a": [6, 5, 12]}, dtype=np.int8) > np.array([6], dtype=np.int8)
        dnpa = orca.DataFrame({"a": [6, 5, 12]}, dtype=dnp.int8) > dnp.array([6], dtype=dnp.int8)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = np.array([6, 50, 12], dtype=np.int16) > np.array([6, 20, 18], dtype=np.int16)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.int16) > dnp.array([6, 20, 18], dtype=dnp.int16)
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12], dtype=np.int16) > pd.Series([6, 20, 18], dtype=np.int16)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.int16) > orca.Series([6, 20, 18], dtype=dnp.int16)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = pd.Series([6, 50, 12], dtype=np.int16) > np.array([6, 20, 18], dtype=np.int16)
        dnpa = orca.Series([6, 50, 12], dtype=dnp.int16) > dnp.array([6, 20, 18], dtype=dnp.int16)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = np.array([6], dtype=np.int16) > pd.DataFrame({"a": [6, 5, 18]}, dtype=np.int16)
        dnpa = dnp.array([6], dtype=dnp.int16) > orca.DataFrame({"a": [6, 5, 18]}, dtype=dnp.int16)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = pd.DataFrame({"a": [6, 5, 12]}, dtype=np.int16) > np.array([6], dtype=np.int16)
        dnpa = orca.DataFrame({"a": [6, 5, 12]}, dtype=dnp.int16) > dnp.array([6], dtype=dnp.int16)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = np.array([6, 50, 12], dtype=np.int32) > np.array([6, 20, 18], dtype=np.int32)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.int32) > dnp.array([6, 20, 18], dtype=dnp.int32)
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12], dtype=np.int32) > pd.Series([6, 20, 18], dtype=np.int32)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.int32) > orca.Series([6, 20, 18], dtype=dnp.int32)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = pd.Series([6, 50, 12], dtype=np.int32) > np.array([6, 20, 18], dtype=np.int32)
        dnpa = orca.Series([6, 50, 12], dtype=dnp.int32) > dnp.array([6, 20, 18], dtype=dnp.int32)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = np.array([6], dtype=np.int32) > pd.DataFrame({"a": [6, 5, 18]}, dtype=np.int32)
        dnpa = dnp.array([6], dtype=dnp.int32) > orca.DataFrame({"a": [6, 5, 18]}, dtype=dnp.int32)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = pd.DataFrame({"a": [6, 50, 12]}, dtype=np.int32) > np.array([6], dtype=np.int32)
        dnpa = orca.DataFrame({"a": [6, 50, 12]}, dtype=dnp.int32) > dnp.array([6], dtype=dnp.int32)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = np.array([6, 50, 12], dtype=np.int64) > np.array([6, 20, 18], dtype=np.int64)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.int64) > dnp.array([6, 20, 18], dtype=dnp.int64)
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12], dtype=np.int64) > pd.Series([6, 20, 18], dtype=np.int64)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.int64) > orca.Series([6, 20, 18], dtype=dnp.int64)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = pd.Series([6, 50, 12], dtype=np.int64) > np.array([6, 20, 18], dtype=np.int64)
        dnpa = orca.Series([6, 50, 12], dtype=dnp.int64) > dnp.array([6, 20, 18], dtype=dnp.int64)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = np.array([6], dtype=np.int64) > pd.DataFrame({"a": [6, 5, 18]}, dtype=np.int64)
        dnpa = dnp.array([6], dtype=dnp.int64) > orca.DataFrame({"a": [6, 5, 18]}, dtype=dnp.int64)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = pd.DataFrame({"a": [6, 5, 12]}, dtype=np.int64) > np.array([6], dtype=np.int64)
        dnpa = orca.DataFrame({"a": [6, 5, 12]}, dtype=dnp.int64) > dnp.array([6], dtype=dnp.int64)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = np.array([6, 50, 12], dtype=np.uint8) > np.array([6, 20, 18], dtype=np.uint8)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.uint8) > dnp.array([6, 20, 18], dtype=dnp.uint8)
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12], dtype=np.uint16) > np.array([6, 20, 18], dtype=np.uint16)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.uint16) > dnp.array([6, 20, 18], dtype=dnp.uint16)
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12], dtype=np.uint32) > np.array([6, 20, 18], dtype=np.uint32)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.uint32) > dnp.array([6, 20, 18], dtype=dnp.uint32)
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12], dtype=np.uint64) > np.array([6, 20, 18], dtype=np.uint64)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.uint64) > dnp.array([6, 20, 18], dtype=dnp.uint64)
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12], dtype=np.intp) > np.array([6, 20, 18], dtype=np.intp)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.intp) > dnp.array([6, 20, 18], dtype=dnp.intp)
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12], dtype=np.intp) > pd.Series([6, 20, 18], dtype=np.intp)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.intp) > orca.Series([6, 20, 18], dtype=dnp.intp)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = pd.Series([6, 50, 12], dtype=np.intp) > np.array([6, 20, 18], dtype=np.intp)
        dnpa = orca.Series([6, 50, 12], dtype=dnp.intp) > dnp.array([6, 20, 18], dtype=dnp.intp)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = np.array([6], dtype=np.intp) > pd.DataFrame({"a": [6, 5, 18]}, dtype=np.intp)
        dnpa = dnp.array([6], dtype=dnp.intp) > orca.DataFrame({"a": [6, 5, 18]}, dtype=dnp.intp)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = pd.DataFrame({"a": [6, 5, 12]}, dtype=np.intp) > np.array([6], dtype=np.intp)
        dnpa = orca.DataFrame({"a": [6, 5, 12]}, dtype=dnp.intp) > dnp.array([6], dtype=dnp.intp)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = np.array([6, 50, 12], dtype=np.uintp) > np.array([6, 20, 18], dtype=np.uintp)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.uintp) > dnp.array([6, 20, 18], dtype=dnp.uintp)
        assert_array_equal(dnpa, npa)

        npa = np.array([12.0, 5.5, 15.0], dtype=np.float32) > np.array([14.7, 5.5, 6.8], dtype=np.float32)
        dnpa = dnp.array([12.0, 5.5, 15.0], dtype=dnp.float32) > dnp.array([14.7, 5.5, 6.8], dtype=dnp.float32)
        assert_array_equal(dnpa, npa)

        npa = np.array([12.0, 5.5, 15.0], dtype=np.float32) > pd.Series([14.7, 5.5, 6.8], dtype=np.float32)
        dnpa = dnp.array([12.0, 5.5, 15.0], dtype=dnp.float32) > orca.Series([14.7, 5.5, 6.8], dtype=dnp.float32)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = pd.Series([14.7, 5.5, 6.8], dtype=np.float32) > np.array([12.0, 5.5, 15.0], dtype=np.float32)
        dnpa = orca.Series([14.7, 5.5, 6.8], dtype=dnp.float32) > dnp.array([12.0, 5.5, 15.0], dtype=dnp.float32)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = np.array([12.0], dtype=np.float32) > pd.DataFrame({'A': [14.7, 5.5, 6.8]}, dtype=np.float32)
        dnpa = dnp.array([12.0], dtype=dnp.float32) > orca.DataFrame({'A': [14.7, 5.5, 6.8]}, dtype=dnp.float32)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = pd.DataFrame({'A': [14.7, 5.5, 6.8]}, dtype=np.float32) > np.array([12.0], dtype=np.float32)
        dnpa = orca.DataFrame({'A': [14.7, 5.5, 6.8]}, dtype=dnp.float32) > dnp.array([12.0], dtype=dnp.float32)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = np.array([12.0, 5.5, 15.0], dtype=np.float64) > np.array([14.7, 5.5, 6.8], dtype=np.float64)
        dnpa = dnp.array([12.0, 5.5, 15.0], dtype=dnp.float64) > dnp.array([14.7, 5.5, 6.8], dtype=dnp.float64)
        assert_array_equal(dnpa, npa)

        npa = np.array([12.0, 5.5, 15.0], dtype=np.float64) > pd.Series([14.7, 5.5, 6.8], dtype=np.float64)
        dnpa = dnp.array([12.0, 5.5, 15.0], dtype=dnp.float64) > orca.Series([14.7, 5.5, 6.8], dtype=dnp.float64)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = pd.Series([14.7, 5.5, 6.8], dtype=np.float64) > np.array([12.0, 5.5, 15.0], dtype=np.float64)
        dnpa = orca.Series([14.7, 5.5, 6.8], dtype=dnp.float64) > dnp.array([12.0, 5.5, 15.0], dtype=dnp.float64)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = np.array([12.0], dtype=np.float64) > pd.DataFrame({'A': [14.7, 5.5, 6.8]}, dtype=np.float64)
        dnpa = dnp.array([12.0], dtype=dnp.float64) > orca.DataFrame({'A': [14.7, 5.5, 6.8]}, dtype=dnp.float64)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = pd.DataFrame({'A': [14.7, 5.5, 6.8]}, dtype=np.float64) > np.array([12.0], dtype=np.float64)
        dnpa = orca.DataFrame({'A': [14.7, 5.5, 6.8]}, dtype=dnp.float64) > dnp.array([12.0], dtype=dnp.float64)
        assert_frame_equal(dnpa.to_pandas(), npa)

    def test_logic_operation_lessthan(self):
        npa = np.array([6, 50, 12]) < np.array([6, 20, 18])
        dnpa = dnp.array([6, 50, 12]) < dnp.array([6, 20, 18])
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12]) < pd.Series([6, 20, 18], dtype=np.int8)
        dnpa = dnp.array([6, 50, 12]) < orca.Series([6, 20, 18], dtype=dnp.int8)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = pd.Series([6, 20, 18], dtype=np.int8) < np.array([6, 50, 12])
        dnpa = orca.Series([6, 20, 18], dtype=dnp.int8) < dnp.array([6, 50, 12])
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = np.array([6, 50, 12], dtype=np.int8) < np.array([6, 20, 18], dtype=np.int8)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.int8) < dnp.array([6, 20, 18], dtype=dnp.int8)
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12], dtype=np.int8) < pd.Series([6, 20, 18], dtype=np.int8)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.int8) < orca.Series([6, 20, 18], dtype=dnp.int8)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = pd.Series([6, 50, 12], dtype=np.int8) < np.array([6, 20, 18], dtype=np.int8)
        dnpa = orca.Series([6, 50, 12], dtype=dnp.int8) < dnp.array([6, 20, 18], dtype=dnp.int8)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = np.array([6], dtype=np.int8) < pd.DataFrame({"a": [6, 5, 18]}, dtype=np.int8)
        dnpa = dnp.array([6], dtype=dnp.int8) < orca.DataFrame({"a": [6, 5, 18]}, dtype=dnp.int8)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = pd.DataFrame({"a": [6, 5, 12]}, dtype=np.int8) < np.array([6], dtype=np.int8)
        dnpa = orca.DataFrame({"a": [6, 5, 12]}, dtype=dnp.int8) < dnp.array([6], dtype=dnp.int8)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = np.array([6, 50, 12], dtype=np.int16) < np.array([6, 20, 18], dtype=np.int16)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.int16) < dnp.array([6, 20, 18], dtype=dnp.int16)
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12], dtype=np.int16) < pd.Series([6, 20, 18], dtype=np.int16)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.int16) < orca.Series([6, 20, 18], dtype=dnp.int16)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = pd.Series([6, 50, 12], dtype=np.int16) < np.array([6, 20, 18], dtype=np.int16)
        dnpa = orca.Series([6, 50, 12], dtype=dnp.int16) < dnp.array([6, 20, 18], dtype=dnp.int16)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = np.array([6], dtype=np.int16) < pd.DataFrame({"a": [6, 5, 18]}, dtype=np.int16)
        dnpa = dnp.array([6], dtype=dnp.int16) < orca.DataFrame({"a": [6, 5, 18]}, dtype=dnp.int16)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = pd.DataFrame({"a": [6, 5, 12]}, dtype=np.int16) < np.array([6], dtype=np.int16)
        dnpa = orca.DataFrame({"a": [6, 5, 12]}, dtype=dnp.int16) < dnp.array([6], dtype=dnp.int16)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = np.array([6, 50, 12], dtype=np.int32) < np.array([6, 20, 18], dtype=np.int32)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.int32) < dnp.array([6, 20, 18], dtype=dnp.int32)
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12], dtype=np.int32) < pd.Series([6, 20, 18], dtype=np.int32)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.int32) < orca.Series([6, 20, 18], dtype=dnp.int32)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = pd.Series([6, 50, 12], dtype=np.int32) < np.array([6, 20, 18], dtype=np.int32)
        dnpa = orca.Series([6, 50, 12], dtype=dnp.int32) < dnp.array([6, 20, 18], dtype=dnp.int32)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = np.array([6], dtype=np.int32) < pd.DataFrame({"a": [6, 5, 18]}, dtype=np.int32)
        dnpa = dnp.array([6], dtype=dnp.int32) < orca.DataFrame({"a": [6, 5, 18]}, dtype=dnp.int32)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = pd.DataFrame({"a": [6, 50, 12]}, dtype=np.int32) < np.array([6], dtype=np.int32)
        dnpa = orca.DataFrame({"a": [6, 50, 12]}, dtype=dnp.int32) < dnp.array([6], dtype=dnp.int32)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = np.array([6, 50, 12], dtype=np.int64) < np.array([6, 20, 18], dtype=np.int64)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.int64) < dnp.array([6, 20, 18], dtype=dnp.int64)
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12], dtype=np.int64) < pd.Series([6, 20, 18], dtype=np.int64)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.int64) < orca.Series([6, 20, 18], dtype=dnp.int64)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = pd.Series([6, 50, 12], dtype=np.int64) < np.array([6, 20, 18], dtype=np.int64)
        dnpa = orca.Series([6, 50, 12], dtype=dnp.int64) < dnp.array([6, 20, 18], dtype=dnp.int64)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = np.array([6], dtype=np.int64) < pd.DataFrame({"a": [6, 5, 18]}, dtype=np.int64)
        dnpa = dnp.array([6], dtype=dnp.int64) < orca.DataFrame({"a": [6, 5, 18]}, dtype=dnp.int64)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = pd.DataFrame({"a": [6, 5, 12]}, dtype=np.int64) < np.array([6], dtype=np.int64)
        dnpa = orca.DataFrame({"a": [6, 5, 12]}, dtype=dnp.int64) < dnp.array([6], dtype=dnp.int64)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = np.array([6, 50, 12], dtype=np.uint8) < np.array([6, 20, 18], dtype=np.uint8)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.uint8) < dnp.array([6, 20, 18], dtype=dnp.uint8)
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12], dtype=np.uint16) < np.array([6, 20, 18], dtype=np.uint16)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.uint16) < dnp.array([6, 20, 18], dtype=dnp.uint16)
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12], dtype=np.uint32) < np.array([6, 20, 18], dtype=np.uint32)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.uint32) < dnp.array([6, 20, 18], dtype=dnp.uint32)
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12], dtype=np.uint64) < np.array([6, 20, 18], dtype=np.uint64)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.uint64) < dnp.array([6, 20, 18], dtype=dnp.uint64)
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12], dtype=np.intp) < np.array([6, 20, 18], dtype=np.intp)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.intp) < dnp.array([6, 20, 18], dtype=dnp.intp)
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12], dtype=np.intp) < pd.Series([6, 20, 18], dtype=np.intp)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.intp) < orca.Series([6, 20, 18], dtype=dnp.intp)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = pd.Series([6, 50, 12], dtype=np.intp) < np.array([6, 20, 18], dtype=np.intp)
        dnpa = orca.Series([6, 50, 12], dtype=dnp.intp) < dnp.array([6, 20, 18], dtype=dnp.intp)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = np.array([6], dtype=np.intp) < pd.DataFrame({"a": [6, 5, 18]}, dtype=np.intp)
        dnpa = dnp.array([6], dtype=dnp.intp) < orca.DataFrame({"a": [6, 5, 18]}, dtype=dnp.intp)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = pd.DataFrame({"a": [6, 5, 12]}, dtype=np.intp) < np.array([6], dtype=np.intp)
        dnpa = orca.DataFrame({"a": [6, 5, 12]}, dtype=dnp.intp) < dnp.array([6], dtype=dnp.intp)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = np.array([6, 50, 12], dtype=np.uintp) < np.array([6, 20, 18], dtype=np.uintp)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.uintp) < dnp.array([6, 20, 18], dtype=dnp.uintp)
        assert_array_equal(dnpa, npa)

        npa = np.array([12.0, 5.5, 15.0], dtype=np.float32) < np.array([14.7, 5.5, 6.8], dtype=np.float32)
        dnpa = dnp.array([12.0, 5.5, 15.0], dtype=dnp.float32) < dnp.array([14.7, 5.5, 6.8], dtype=dnp.float32)
        assert_array_equal(dnpa, npa)

        npa = np.array([12.0, 5.5, 15.0], dtype=np.float32) < pd.Series([14.7, 5.5, 6.8], dtype=np.float32)
        dnpa = dnp.array([12.0, 5.5, 15.0], dtype=dnp.float32) < orca.Series([14.7, 5.5, 6.8], dtype=dnp.float32)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = pd.Series([14.7, 5.5, 6.8], dtype=np.float32) < np.array([12.0, 5.5, 15.0], dtype=np.float32)
        dnpa = orca.Series([14.7, 5.5, 6.8], dtype=dnp.float32) < dnp.array([12.0, 5.5, 15.0], dtype=dnp.float32)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = np.array([12.0], dtype=np.float32) < pd.DataFrame({'A': [14.7, 5.5, 6.8]}, dtype=np.float32)
        dnpa = dnp.array([12.0], dtype=dnp.float32) < orca.DataFrame({'A': [14.7, 5.5, 6.8]}, dtype=dnp.float32)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = pd.DataFrame({'A': [14.7, 5.5, 6.8]}, dtype=np.float32) < np.array([12.0], dtype=np.float32)
        dnpa = orca.DataFrame({'A': [14.7, 5.5, 6.8]}, dtype=dnp.float32) < dnp.array([12.0], dtype=dnp.float32)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = np.array([12.0, 5.5, 15.0], dtype=np.float64) < np.array([14.7, 5.5, 6.8], dtype=np.float64)
        dnpa = dnp.array([12.0, 5.5, 15.0], dtype=dnp.float64) < dnp.array([14.7, 5.5, 6.8], dtype=dnp.float64)
        assert_array_equal(dnpa, npa)

        npa = np.array([12.0, 5.5, 15.0], dtype=np.float64) < pd.Series([14.7, 5.5, 6.8], dtype=np.float64)
        dnpa = dnp.array([12.0, 5.5, 15.0], dtype=dnp.float64) < orca.Series([14.7, 5.5, 6.8], dtype=dnp.float64)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = pd.Series([14.7, 5.5, 6.8], dtype=np.float64) < np.array([12.0, 5.5, 15.0], dtype=np.float64)
        dnpa = orca.Series([14.7, 5.5, 6.8], dtype=dnp.float64) < dnp.array([12.0, 5.5, 15.0], dtype=dnp.float64)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = np.array([12.0], dtype=np.float64) < pd.DataFrame({'A': [14.7, 5.5, 6.8]}, dtype=np.float64)
        dnpa = dnp.array([12.0], dtype=dnp.float64) < orca.DataFrame({'A': [14.7, 5.5, 6.8]}, dtype=dnp.float64)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = pd.DataFrame({'A': [14.7, 5.5, 6.8]}, dtype=np.float64) < np.array([12.0], dtype=np.float64)
        dnpa = orca.DataFrame({'A': [14.7, 5.5, 6.8]}, dtype=dnp.float64) < dnp.array([12.0], dtype=dnp.float64)
        assert_frame_equal(dnpa.to_pandas(), npa)

    def test_logic_operation_greaterequalthan(self):
        npa = np.array([6, 50, 12]) >= np.array([6, 20, 18])
        dnpa = dnp.array([6, 50, 12]) >= dnp.array([6, 20, 18])
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12]) >= pd.Series([6, 20, 18], dtype=np.int8)
        dnpa = dnp.array([6, 50, 12]) >= orca.Series([6, 20, 18], dtype=dnp.int8)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = pd.Series([6, 20, 18], dtype=np.int8) >= np.array([6, 50, 12])
        dnpa = orca.Series([6, 20, 18], dtype=dnp.int8) >= dnp.array([6, 50, 12])
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = np.array([6, 50, 12], dtype=np.int8) >= np.array([6, 20, 18], dtype=np.int8)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.int8) >= dnp.array([6, 20, 18], dtype=dnp.int8)
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12], dtype=np.int8) >= pd.Series([6, 20, 18], dtype=np.int8)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.int8) >= orca.Series([6, 20, 18], dtype=dnp.int8)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = pd.Series([6, 50, 12], dtype=np.int8) >= np.array([6, 20, 18], dtype=np.int8)
        dnpa = orca.Series([6, 50, 12], dtype=dnp.int8) >= dnp.array([6, 20, 18], dtype=dnp.int8)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = np.array([6], dtype=np.int8) >= pd.DataFrame({"a": [6, 5, 18]}, dtype=np.int8)
        dnpa = dnp.array([6], dtype=dnp.int8) >= orca.DataFrame({"a": [6, 5, 18]}, dtype=dnp.int8)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = pd.DataFrame({"a": [6, 5, 12]}, dtype=np.int8) >= np.array([6], dtype=np.int8)
        dnpa = orca.DataFrame({"a": [6, 5, 12]}, dtype=dnp.int8) >= dnp.array([6], dtype=dnp.int8)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = np.array([6, 50, 12], dtype=np.int16) >= np.array([6, 20, 18], dtype=np.int16)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.int16) >= dnp.array([6, 20, 18], dtype=dnp.int16)
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12], dtype=np.int16) >= pd.Series([6, 20, 18], dtype=np.int16)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.int16) >= orca.Series([6, 20, 18], dtype=dnp.int16)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = pd.Series([6, 50, 12], dtype=np.int16) >= np.array([6, 20, 18], dtype=np.int16)
        dnpa = orca.Series([6, 50, 12], dtype=dnp.int16) >= dnp.array([6, 20, 18], dtype=dnp.int16)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = np.array([6], dtype=np.int16) >= pd.DataFrame({"a": [6, 5, 18]}, dtype=np.int16)
        dnpa = dnp.array([6], dtype=dnp.int16) >= orca.DataFrame({"a": [6, 5, 18]}, dtype=dnp.int16)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = pd.DataFrame({"a": [6, 5, 12]}, dtype=np.int16) >= np.array([6], dtype=np.int16)
        dnpa = orca.DataFrame({"a": [6, 5, 12]}, dtype=dnp.int16) >= dnp.array([6], dtype=dnp.int16)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = np.array([6, 50, 12], dtype=np.int32) >= np.array([6, 20, 18], dtype=np.int32)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.int32) >= dnp.array([6, 20, 18], dtype=dnp.int32)
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12], dtype=np.int32) >= pd.Series([6, 20, 18], dtype=np.int32)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.int32) >= orca.Series([6, 20, 18], dtype=dnp.int32)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = pd.Series([6, 50, 12], dtype=np.int32) >= np.array([6, 20, 18], dtype=np.int32)
        dnpa = orca.Series([6, 50, 12], dtype=dnp.int32) >= dnp.array([6, 20, 18], dtype=dnp.int32)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = np.array([6], dtype=np.int32) >= pd.DataFrame({"a": [6, 5, 18]}, dtype=np.int32)
        dnpa = dnp.array([6], dtype=dnp.int32) >= orca.DataFrame({"a": [6, 5, 18]}, dtype=dnp.int32)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = pd.DataFrame({"a": [6, 50, 12]}, dtype=np.int32) >= np.array([6], dtype=np.int32)
        dnpa = orca.DataFrame({"a": [6, 50, 12]}, dtype=dnp.int32) >= dnp.array([6], dtype=dnp.int32)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = np.array([6, 50, 12], dtype=np.int64) >= np.array([6, 20, 18], dtype=np.int64)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.int64) >= dnp.array([6, 20, 18], dtype=dnp.int64)
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12], dtype=np.int64) >= pd.Series([6, 20, 18], dtype=np.int64)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.int64) >= orca.Series([6, 20, 18], dtype=dnp.int64)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = pd.Series([6, 50, 12], dtype=np.int64) >= np.array([6, 20, 18], dtype=np.int64)
        dnpa = orca.Series([6, 50, 12], dtype=dnp.int64) >= dnp.array([6, 20, 18], dtype=dnp.int64)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = np.array([6], dtype=np.int64) >= pd.DataFrame({"a": [6, 5, 18]}, dtype=np.int64)
        dnpa = dnp.array([6], dtype=dnp.int64) >= orca.DataFrame({"a": [6, 5, 18]}, dtype=dnp.int64)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = pd.DataFrame({"a": [6, 5, 12]}, dtype=np.int64) >= np.array([6], dtype=np.int64)
        dnpa = orca.DataFrame({"a": [6, 5, 12]}, dtype=dnp.int64) >= dnp.array([6], dtype=dnp.int64)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = np.array([6, 50, 12], dtype=np.uint8) >= np.array([6, 20, 18], dtype=np.uint8)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.uint8) >= dnp.array([6, 20, 18], dtype=dnp.uint8)
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12], dtype=np.uint16) >= np.array([6, 20, 18], dtype=np.uint16)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.uint16) >= dnp.array([6, 20, 18], dtype=dnp.uint16)
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12], dtype=np.uint32) >= np.array([6, 20, 18], dtype=np.uint32)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.uint32) >= dnp.array([6, 20, 18], dtype=dnp.uint32)
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12], dtype=np.uint64) >= np.array([6, 20, 18], dtype=np.uint64)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.uint64) >= dnp.array([6, 20, 18], dtype=dnp.uint64)
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12], dtype=np.intp) >= np.array([6, 20, 18], dtype=np.intp)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.intp) >= dnp.array([6, 20, 18], dtype=dnp.intp)
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12], dtype=np.intp) >= pd.Series([6, 20, 18], dtype=np.intp)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.intp) >= orca.Series([6, 20, 18], dtype=dnp.intp)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = pd.Series([6, 50, 12], dtype=np.intp) >= np.array([6, 20, 18], dtype=np.intp)
        dnpa = orca.Series([6, 50, 12], dtype=dnp.intp) >= dnp.array([6, 20, 18], dtype=dnp.intp)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = np.array([6], dtype=np.intp) >= pd.DataFrame({"a": [6, 5, 18]}, dtype=np.intp)
        dnpa = dnp.array([6], dtype=dnp.intp) >= orca.DataFrame({"a": [6, 5, 18]}, dtype=dnp.intp)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = pd.DataFrame({"a": [6, 5, 12]}, dtype=np.intp) >= np.array([6], dtype=np.intp)
        dnpa = orca.DataFrame({"a": [6, 5, 12]}, dtype=dnp.intp) >= dnp.array([6], dtype=dnp.intp)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = np.array([6, 50, 12], dtype=np.uintp) >= np.array([6, 20, 18], dtype=np.uintp)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.uintp) >= dnp.array([6, 20, 18], dtype=dnp.uintp)
        assert_array_equal(dnpa, npa)

        npa = np.array([12.0, 5.5, 15.0], dtype=np.float32) >= np.array([14.7, 5.5, 6.8], dtype=np.float32)
        dnpa = dnp.array([12.0, 5.5, 15.0], dtype=dnp.float32) >= dnp.array([14.7, 5.5, 6.8], dtype=dnp.float32)
        assert_array_equal(dnpa, npa)

        npa = np.array([12.0, 5.5, 15.0], dtype=np.float32) >= pd.Series([14.7, 5.5, 6.8], dtype=np.float32)
        dnpa = dnp.array([12.0, 5.5, 15.0], dtype=dnp.float32) >= orca.Series([14.7, 5.5, 6.8], dtype=dnp.float32)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = pd.Series([14.7, 5.5, 6.8], dtype=np.float32) >= np.array([12.0, 5.5, 15.0], dtype=np.float32)
        dnpa = orca.Series([14.7, 5.5, 6.8], dtype=dnp.float32) >= dnp.array([12.0, 5.5, 15.0], dtype=dnp.float32)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = np.array([12.0], dtype=np.float32) >= pd.DataFrame({'A': [14.7, 5.5, 6.8]}, dtype=np.float32)
        dnpa = dnp.array([12.0], dtype=dnp.float32) >= orca.DataFrame({'A': [14.7, 5.5, 6.8]}, dtype=dnp.float32)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = pd.DataFrame({'A': [14.7, 5.5, 6.8]}, dtype=np.float32) >= np.array([12.0], dtype=np.float32)
        dnpa = orca.DataFrame({'A': [14.7, 5.5, 6.8]}, dtype=dnp.float32) >= dnp.array([12.0], dtype=dnp.float32)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = np.array([12.0, 5.5, 15.0], dtype=np.float64) >= np.array([14.7, 5.5, 6.8], dtype=np.float64)
        dnpa = dnp.array([12.0, 5.5, 15.0], dtype=dnp.float64) >= dnp.array([14.7, 5.5, 6.8], dtype=dnp.float64)
        assert_array_equal(dnpa, npa)

        npa = np.array([12.0, 5.5, 15.0], dtype=np.float64) >= pd.Series([14.7, 5.5, 6.8], dtype=np.float64)
        dnpa = dnp.array([12.0, 5.5, 15.0], dtype=dnp.float64) >= orca.Series([14.7, 5.5, 6.8], dtype=dnp.float64)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = pd.Series([14.7, 5.5, 6.8], dtype=np.float64) >= np.array([12.0, 5.5, 15.0], dtype=np.float64)
        dnpa = orca.Series([14.7, 5.5, 6.8], dtype=dnp.float64) >= dnp.array([12.0, 5.5, 15.0], dtype=dnp.float64)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = np.array([12.0], dtype=np.float64) >= pd.DataFrame({'A': [14.7, 5.5, 6.8]}, dtype=np.float64)
        dnpa = dnp.array([12.0], dtype=dnp.float64) >= orca.DataFrame({'A': [14.7, 5.5, 6.8]}, dtype=dnp.float64)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = pd.DataFrame({'A': [14.7, 5.5, 6.8]}, dtype=np.float64) >= np.array([12.0], dtype=np.float64)
        dnpa = orca.DataFrame({'A': [14.7, 5.5, 6.8]}, dtype=dnp.float64) >= dnp.array([12.0], dtype=dnp.float64)
        assert_frame_equal(dnpa.to_pandas(), npa)

    def test_logic_operation_lessequalthan(self):
        npa = np.array([6, 50, 12]) <= np.array([6, 20, 18])
        dnpa = dnp.array([6, 50, 12]) <= dnp.array([6, 20, 18])
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12]) <= pd.Series([6, 20, 18], dtype=np.int8)
        dnpa = dnp.array([6, 50, 12]) <= orca.Series([6, 20, 18], dtype=dnp.int8)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = pd.Series([6, 20, 18], dtype=np.int8) <= np.array([6, 50, 12])
        dnpa = orca.Series([6, 20, 18], dtype=dnp.int8) <= dnp.array([6, 50, 12])
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = np.array([6, 50, 12], dtype=np.int8) <= np.array([6, 20, 18], dtype=np.int8)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.int8) <= dnp.array([6, 20, 18], dtype=dnp.int8)
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12], dtype=np.int8) <= pd.Series([6, 20, 18], dtype=np.int8)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.int8) <= orca.Series([6, 20, 18], dtype=dnp.int8)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = pd.Series([6, 50, 12], dtype=np.int8) <= np.array([6, 20, 18], dtype=np.int8)
        dnpa = orca.Series([6, 50, 12], dtype=dnp.int8) <= dnp.array([6, 20, 18], dtype=dnp.int8)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = np.array([6], dtype=np.int8) <= pd.DataFrame({"a": [6, 5, 18]}, dtype=np.int8)
        dnpa = dnp.array([6], dtype=dnp.int8) <= orca.DataFrame({"a": [6, 5, 18]}, dtype=dnp.int8)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = pd.DataFrame({"a": [6, 5, 12]}, dtype=np.int8) <= np.array([6], dtype=np.int8)
        dnpa = orca.DataFrame({"a": [6, 5, 12]}, dtype=dnp.int8) <= dnp.array([6], dtype=dnp.int8)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = np.array([6, 50, 12], dtype=np.int16) <= np.array([6, 20, 18], dtype=np.int16)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.int16) <= dnp.array([6, 20, 18], dtype=dnp.int16)
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12], dtype=np.int16) <= pd.Series([6, 20, 18], dtype=np.int16)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.int16) <= orca.Series([6, 20, 18], dtype=dnp.int16)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = pd.Series([6, 50, 12], dtype=np.int16) <= np.array([6, 20, 18], dtype=np.int16)
        dnpa = orca.Series([6, 50, 12], dtype=dnp.int16) <= dnp.array([6, 20, 18], dtype=dnp.int16)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = np.array([6], dtype=np.int16) <= pd.DataFrame({"a": [6, 5, 18]}, dtype=np.int16)
        dnpa = dnp.array([6], dtype=dnp.int16) <= orca.DataFrame({"a": [6, 5, 18]}, dtype=dnp.int16)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = pd.DataFrame({"a": [6, 5, 12]}, dtype=np.int16) <= np.array([6], dtype=np.int16)
        dnpa = orca.DataFrame({"a": [6, 5, 12]}, dtype=dnp.int16) <= dnp.array([6], dtype=dnp.int16)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = np.array([6, 50, 12], dtype=np.int32) <= np.array([6, 20, 18], dtype=np.int32)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.int32) <= dnp.array([6, 20, 18], dtype=dnp.int32)
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12], dtype=np.int32) <= pd.Series([6, 20, 18], dtype=np.int32)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.int32) <= orca.Series([6, 20, 18], dtype=dnp.int32)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = pd.Series([6, 50, 12], dtype=np.int32) <= np.array([6, 20, 18], dtype=np.int32)
        dnpa = orca.Series([6, 50, 12], dtype=dnp.int32) <= dnp.array([6, 20, 18], dtype=dnp.int32)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = np.array([6], dtype=np.int32) <= pd.DataFrame({"a": [6, 5, 18]}, dtype=np.int32)
        dnpa = dnp.array([6], dtype=dnp.int32) <= orca.DataFrame({"a": [6, 5, 18]}, dtype=dnp.int32)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = pd.DataFrame({"a": [6, 50, 12]}, dtype=np.int32) <= np.array([6], dtype=np.int32)
        dnpa = orca.DataFrame({"a": [6, 50, 12]}, dtype=dnp.int32) <= dnp.array([6], dtype=dnp.int32)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = np.array([6, 50, 12], dtype=np.int64) <= np.array([6, 20, 18], dtype=np.int64)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.int64) <= dnp.array([6, 20, 18], dtype=dnp.int64)
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12], dtype=np.int64) <= pd.Series([6, 20, 18], dtype=np.int64)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.int64) <= orca.Series([6, 20, 18], dtype=dnp.int64)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = pd.Series([6, 50, 12], dtype=np.int64) <= np.array([6, 20, 18], dtype=np.int64)
        dnpa = orca.Series([6, 50, 12], dtype=dnp.int64) <= dnp.array([6, 20, 18], dtype=dnp.int64)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = np.array([6], dtype=np.int64) <= pd.DataFrame({"a": [6, 5, 18]}, dtype=np.int64)
        dnpa = dnp.array([6], dtype=dnp.int64) <= orca.DataFrame({"a": [6, 5, 18]}, dtype=dnp.int64)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = pd.DataFrame({"a": [6, 5, 12]}, dtype=np.int64) <= np.array([6], dtype=np.int64)
        dnpa = orca.DataFrame({"a": [6, 5, 12]}, dtype=dnp.int64) <= dnp.array([6], dtype=dnp.int64)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = np.array([6, 50, 12], dtype=np.uint8) <= np.array([6, 20, 18], dtype=np.uint8)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.uint8) <= dnp.array([6, 20, 18], dtype=dnp.uint8)
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12], dtype=np.uint16) <= np.array([6, 20, 18], dtype=np.uint16)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.uint16) <= dnp.array([6, 20, 18], dtype=dnp.uint16)
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12], dtype=np.uint32) <= np.array([6, 20, 18], dtype=np.uint32)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.uint32) <= dnp.array([6, 20, 18], dtype=dnp.uint32)
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12], dtype=np.uint64) <= np.array([6, 20, 18], dtype=np.uint64)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.uint64) <= dnp.array([6, 20, 18], dtype=dnp.uint64)
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12], dtype=np.intp) <= np.array([6, 20, 18], dtype=np.intp)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.intp) <= dnp.array([6, 20, 18], dtype=dnp.intp)
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12], dtype=np.intp) <= pd.Series([6, 20, 18], dtype=np.intp)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.intp) <= orca.Series([6, 20, 18], dtype=dnp.intp)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = pd.Series([6, 50, 12], dtype=np.intp) <= np.array([6, 20, 18], dtype=np.intp)
        dnpa = orca.Series([6, 50, 12], dtype=dnp.intp) <= dnp.array([6, 20, 18], dtype=dnp.intp)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = np.array([6], dtype=np.intp) <= pd.DataFrame({"a": [6, 5, 18]}, dtype=np.intp)
        dnpa = dnp.array([6], dtype=dnp.intp) <= orca.DataFrame({"a": [6, 5, 18]}, dtype=dnp.intp)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = pd.DataFrame({"a": [6, 5, 12]}, dtype=np.intp) <= np.array([6], dtype=np.intp)
        dnpa = orca.DataFrame({"a": [6, 5, 12]}, dtype=dnp.intp) <= dnp.array([6], dtype=dnp.intp)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = np.array([6, 50, 12], dtype=np.uintp) <= np.array([6, 20, 18], dtype=np.uintp)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.uintp) <= dnp.array([6, 20, 18], dtype=dnp.uintp)
        assert_array_equal(dnpa, npa)

        npa = np.array([12.0, 5.5, 15.0], dtype=np.float32) <= np.array([14.7, 5.5, 6.8], dtype=np.float32)
        dnpa = dnp.array([12.0, 5.5, 15.0], dtype=dnp.float32) <= dnp.array([14.7, 5.5, 6.8], dtype=dnp.float32)
        assert_array_equal(dnpa, npa)

        npa = np.array([12.0, 5.5, 15.0], dtype=np.float32) <= pd.Series([14.7, 5.5, 6.8], dtype=np.float32)
        dnpa = dnp.array([12.0, 5.5, 15.0], dtype=dnp.float32) <= orca.Series([14.7, 5.5, 6.8], dtype=dnp.float32)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = pd.Series([14.7, 5.5, 6.8], dtype=np.float32) <= np.array([12.0, 5.5, 15.0], dtype=np.float32)
        dnpa = orca.Series([14.7, 5.5, 6.8], dtype=dnp.float32) <= dnp.array([12.0, 5.5, 15.0], dtype=dnp.float32)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = np.array([12.0], dtype=np.float32) <= pd.DataFrame({'A': [14.7, 5.5, 6.8]}, dtype=np.float32)
        dnpa = dnp.array([12.0], dtype=dnp.float32) <= orca.DataFrame({'A': [14.7, 5.5, 6.8]}, dtype=dnp.float32)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = pd.DataFrame({'A': [14.7, 5.5, 6.8]}, dtype=np.float32) <= np.array([12.0], dtype=np.float32)
        dnpa = orca.DataFrame({'A': [14.7, 5.5, 6.8]}, dtype=dnp.float32) <= dnp.array([12.0], dtype=dnp.float32)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = np.array([12.0, 5.5, 15.0], dtype=np.float64) <= np.array([14.7, 5.5, 6.8], dtype=np.float64)
        dnpa = dnp.array([12.0, 5.5, 15.0], dtype=dnp.float64) <= dnp.array([14.7, 5.5, 6.8], dtype=dnp.float64)
        assert_array_equal(dnpa, npa)

        npa = np.array([12.0, 5.5, 15.0], dtype=np.float64) <= pd.Series([14.7, 5.5, 6.8], dtype=np.float64)
        dnpa = dnp.array([12.0, 5.5, 15.0], dtype=dnp.float64) <= orca.Series([14.7, 5.5, 6.8], dtype=dnp.float64)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = pd.Series([14.7, 5.5, 6.8], dtype=np.float64) <= np.array([12.0, 5.5, 15.0], dtype=np.float64)
        dnpa = orca.Series([14.7, 5.5, 6.8], dtype=dnp.float64) <= dnp.array([12.0, 5.5, 15.0], dtype=dnp.float64)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = np.array([12.0], dtype=np.float64) <= pd.DataFrame({'A': [14.7, 5.5, 6.8]}, dtype=np.float64)
        dnpa = dnp.array([12.0], dtype=dnp.float64) <= orca.DataFrame({'A': [14.7, 5.5, 6.8]}, dtype=dnp.float64)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = pd.DataFrame({'A': [14.7, 5.5, 6.8]}, dtype=np.float64) <= np.array([12.0], dtype=np.float64)
        dnpa = orca.DataFrame({'A': [14.7, 5.5, 6.8]}, dtype=dnp.float64) <= dnp.array([12.0], dtype=dnp.float64)
        assert_frame_equal(dnpa.to_pandas(), npa)

    def test_logic_operation_notequal(self):
        npa = np.array([6, 50, 12]) != np.array([6, 20, 18])
        dnpa = dnp.array([6, 50, 12]) != dnp.array([6, 20, 18])
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12]) != pd.Series([6, 20, 18], dtype=np.int8)
        dnpa = dnp.array([6, 50, 12]) != orca.Series([6, 20, 18], dtype=dnp.int8)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = pd.Series([6, 20, 18], dtype=np.int8) != np.array([6, 50, 12])
        dnpa = orca.Series([6, 20, 18], dtype=dnp.int8) != dnp.array([6, 50, 12])
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = np.array([6, 50, 12], dtype=np.int8) != np.array([6, 20, 18], dtype=np.int8)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.int8) != dnp.array([6, 20, 18], dtype=dnp.int8)
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12], dtype=np.int8) != pd.Series([6, 20, 18], dtype=np.int8)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.int8) != orca.Series([6, 20, 18], dtype=dnp.int8)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = pd.Series([6, 50, 12], dtype=np.int8) != np.array([6, 20, 18], dtype=np.int8)
        dnpa = orca.Series([6, 50, 12], dtype=dnp.int8) != dnp.array([6, 20, 18], dtype=dnp.int8)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = np.array([6], dtype=np.int8) != pd.DataFrame({"a": [6, 5, 18]}, dtype=np.int8)
        dnpa = dnp.array([6], dtype=dnp.int8) != orca.DataFrame({"a": [6, 5, 18]}, dtype=dnp.int8)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = pd.DataFrame({"a": [6, 5, 12]}, dtype=np.int8) != np.array([6], dtype=np.int8)
        dnpa = orca.DataFrame({"a": [6, 5, 12]}, dtype=dnp.int8) != dnp.array([6], dtype=dnp.int8)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = np.array([6, 50, 12], dtype=np.int16) != np.array([6, 20, 18], dtype=np.int16)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.int16) != dnp.array([6, 20, 18], dtype=dnp.int16)
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12], dtype=np.int16) != pd.Series([6, 20, 18], dtype=np.int16)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.int16) != orca.Series([6, 20, 18], dtype=dnp.int16)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = pd.Series([6, 50, 12], dtype=np.int16) != np.array([6, 20, 18], dtype=np.int16)
        dnpa = orca.Series([6, 50, 12], dtype=dnp.int16) != dnp.array([6, 20, 18], dtype=dnp.int16)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = np.array([6], dtype=np.int16) != pd.DataFrame({"a": [6, 5, 18]}, dtype=np.int16)
        dnpa = dnp.array([6], dtype=dnp.int16) != orca.DataFrame({"a": [6, 5, 18]}, dtype=dnp.int16)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = pd.DataFrame({"a": [6, 5, 12]}, dtype=np.int16) != np.array([6], dtype=np.int16)
        dnpa = orca.DataFrame({"a": [6, 5, 12]}, dtype=dnp.int16) != dnp.array([6], dtype=dnp.int16)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = np.array([6, 50, 12], dtype=np.int32) != np.array([6, 20, 18], dtype=np.int32)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.int32) != dnp.array([6, 20, 18], dtype=dnp.int32)
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12], dtype=np.int32) != pd.Series([6, 20, 18], dtype=np.int32)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.int32) != orca.Series([6, 20, 18], dtype=dnp.int32)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = pd.Series([6, 50, 12], dtype=np.int32) != np.array([6, 20, 18], dtype=np.int32)
        dnpa = orca.Series([6, 50, 12], dtype=dnp.int32) != dnp.array([6, 20, 18], dtype=dnp.int32)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = np.array([6], dtype=np.int32) != pd.DataFrame({"a": [6, 5, 18]}, dtype=np.int32)
        dnpa = dnp.array([6], dtype=dnp.int32) != orca.DataFrame({"a": [6, 5, 18]}, dtype=dnp.int32)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = pd.DataFrame({"a": [6, 50, 12]}, dtype=np.int32) != np.array([6], dtype=np.int32)
        dnpa = orca.DataFrame({"a": [6, 50, 12]}, dtype=dnp.int32) != dnp.array([6], dtype=dnp.int32)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = np.array([6, 50, 12], dtype=np.int64) != np.array([6, 20, 18], dtype=np.int64)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.int64) != dnp.array([6, 20, 18], dtype=dnp.int64)
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12], dtype=np.int64) != pd.Series([6, 20, 18], dtype=np.int64)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.int64) != orca.Series([6, 20, 18], dtype=dnp.int64)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = pd.Series([6, 50, 12], dtype=np.int64) != np.array([6, 20, 18], dtype=np.int64)
        dnpa = orca.Series([6, 50, 12], dtype=dnp.int64) != dnp.array([6, 20, 18], dtype=dnp.int64)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = np.array([6], dtype=np.int64) != pd.DataFrame({"a": [6, 5, 18]}, dtype=np.int64)
        dnpa = dnp.array([6], dtype=dnp.int64) != orca.DataFrame({"a": [6, 5, 18]}, dtype=dnp.int64)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = pd.DataFrame({"a": [6, 5, 12]}, dtype=np.int64) != np.array([6], dtype=np.int64)
        dnpa = orca.DataFrame({"a": [6, 5, 12]}, dtype=dnp.int64) != dnp.array([6], dtype=dnp.int64)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = np.array([6, 50, 12], dtype=np.uint8) != np.array([6, 20, 18], dtype=np.uint8)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.uint8) != dnp.array([6, 20, 18], dtype=dnp.uint8)
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12], dtype=np.uint16) != np.array([6, 20, 18], dtype=np.uint16)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.uint16) != dnp.array([6, 20, 18], dtype=dnp.uint16)
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12], dtype=np.uint32) != np.array([6, 20, 18], dtype=np.uint32)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.uint32) != dnp.array([6, 20, 18], dtype=dnp.uint32)
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12], dtype=np.uint64) != np.array([6, 20, 18], dtype=np.uint64)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.uint64) != dnp.array([6, 20, 18], dtype=dnp.uint64)
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12], dtype=np.intp) != np.array([6, 20, 18], dtype=np.intp)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.intp) != dnp.array([6, 20, 18], dtype=dnp.intp)
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12], dtype=np.intp) != pd.Series([6, 20, 18], dtype=np.intp)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.intp) != orca.Series([6, 20, 18], dtype=dnp.intp)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = pd.Series([6, 50, 12], dtype=np.intp) != np.array([6, 20, 18], dtype=np.intp)
        dnpa = orca.Series([6, 50, 12], dtype=dnp.intp) != dnp.array([6, 20, 18], dtype=dnp.intp)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = np.array([6], dtype=np.intp) != pd.DataFrame({"a": [6, 5, 18]}, dtype=np.intp)
        dnpa = dnp.array([6], dtype=dnp.intp) != orca.DataFrame({"a": [6, 5, 18]}, dtype=dnp.intp)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = pd.DataFrame({"a": [6, 5, 12]}, dtype=np.intp) != np.array([6], dtype=np.intp)
        dnpa = orca.DataFrame({"a": [6, 5, 12]}, dtype=dnp.intp) != dnp.array([6], dtype=dnp.intp)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = np.array([6, 50, 12], dtype=np.uintp) != np.array([6, 20, 18], dtype=np.uintp)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.uintp) != dnp.array([6, 20, 18], dtype=dnp.uintp)
        assert_array_equal(dnpa, npa)

        npa = np.array([12.0, 5.5, 15.0], dtype=np.float32) != np.array([14.7, 5.5, 6.8], dtype=np.float32)
        dnpa = dnp.array([12.0, 5.5, 15.0], dtype=dnp.float32) != dnp.array([14.7, 5.5, 6.8], dtype=dnp.float32)
        assert_array_equal(dnpa, npa)

        npa = np.array([12.0, 5.5, 15.0], dtype=np.float32) != pd.Series([14.7, 5.5, 6.8], dtype=np.float32)
        dnpa = dnp.array([12.0, 5.5, 15.0], dtype=dnp.float32) != orca.Series([14.7, 5.5, 6.8], dtype=dnp.float32)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = pd.Series([14.7, 5.5, 6.8], dtype=np.float32) != np.array([12.0, 5.5, 15.0], dtype=np.float32)
        dnpa = orca.Series([14.7, 5.5, 6.8], dtype=dnp.float32) != dnp.array([12.0, 5.5, 15.0], dtype=dnp.float32)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = np.array([12.0], dtype=np.float32) != pd.DataFrame({'A': [14.7, 5.5, 6.8]}, dtype=np.float32)
        dnpa = dnp.array([12.0], dtype=dnp.float32) != orca.DataFrame({'A': [14.7, 5.5, 6.8]}, dtype=dnp.float32)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = pd.DataFrame({'A': [14.7, 5.5, 6.8]}, dtype=np.float32) != np.array([12.0], dtype=np.float32)
        dnpa = orca.DataFrame({'A': [14.7, 5.5, 6.8]}, dtype=dnp.float32) != dnp.array([12.0], dtype=dnp.float32)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = np.array([12.0, 5.5, 15.0], dtype=np.float64) != np.array([14.7, 5.5, 6.8], dtype=np.float64)
        dnpa = dnp.array([12.0, 5.5, 15.0], dtype=dnp.float64) != dnp.array([14.7, 5.5, 6.8], dtype=dnp.float64)
        assert_array_equal(dnpa, npa)

        npa = np.array([12.0, 5.5, 15.0], dtype=np.float64) != pd.Series([14.7, 5.5, 6.8], dtype=np.float64)
        dnpa = dnp.array([12.0, 5.5, 15.0], dtype=dnp.float64) != orca.Series([14.7, 5.5, 6.8], dtype=dnp.float64)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = pd.Series([14.7, 5.5, 6.8], dtype=np.float64) != np.array([12.0, 5.5, 15.0], dtype=np.float64)
        dnpa = orca.Series([14.7, 5.5, 6.8], dtype=dnp.float64) != dnp.array([12.0, 5.5, 15.0], dtype=dnp.float64)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = np.array([12.0], dtype=np.float64) != pd.DataFrame({'A': [14.7, 5.5, 6.8]}, dtype=np.float64)
        dnpa = dnp.array([12.0], dtype=dnp.float64) != orca.DataFrame({'A': [14.7, 5.5, 6.8]}, dtype=dnp.float64)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = pd.DataFrame({'A': [14.7, 5.5, 6.8]}, dtype=np.float64) != np.array([12.0], dtype=np.float64)
        dnpa = orca.DataFrame({'A': [14.7, 5.5, 6.8]}, dtype=dnp.float64) != dnp.array([12.0], dtype=dnp.float64)
        assert_frame_equal(dnpa.to_pandas(), npa)

    def test_logic_operation_equal(self):
        npa = np.array([6, 50, 12]) == np.array([6, 20, 18])
        dnpa = dnp.array([6, 50, 12]) == dnp.array([6, 20, 18])
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12]) == pd.Series([6, 20, 18], dtype=np.int8)
        dnpa = dnp.array([6, 50, 12]) == orca.Series([6, 20, 18], dtype=dnp.int8)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = pd.Series([6, 20, 18], dtype=np.int8) == np.array([6, 50, 12])
        dnpa = orca.Series([6, 20, 18], dtype=dnp.int8) == dnp.array([6, 50, 12])
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = np.array([6, 50, 12], dtype=np.int8) == np.array([6, 20, 18], dtype=np.int8)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.int8) == dnp.array([6, 20, 18], dtype=dnp.int8)
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12], dtype=np.int8) == pd.Series([6, 20, 18], dtype=np.int8)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.int8) == orca.Series([6, 20, 18], dtype=dnp.int8)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = pd.Series([6, 50, 12], dtype=np.int8) == np.array([6, 20, 18], dtype=np.int8)
        dnpa = orca.Series([6, 50, 12], dtype=dnp.int8) == dnp.array([6, 20, 18], dtype=dnp.int8)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = np.array([6], dtype=np.int8) == pd.DataFrame({"a": [6, 5, 18]}, dtype=np.int8)
        dnpa = dnp.array([6], dtype=dnp.int8) == orca.DataFrame({"a": [6, 5, 18]}, dtype=dnp.int8)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = pd.DataFrame({"a": [6, 5, 12]}, dtype=np.int8) == np.array([6], dtype=np.int8)
        dnpa = orca.DataFrame({"a": [6, 5, 12]}, dtype=dnp.int8) == dnp.array([6], dtype=dnp.int8)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = np.array([6, 50, 12], dtype=np.int16) == np.array([6, 20, 18], dtype=np.int16)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.int16) == dnp.array([6, 20, 18], dtype=dnp.int16)
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12], dtype=np.int16) == pd.Series([6, 20, 18], dtype=np.int16)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.int16) == orca.Series([6, 20, 18], dtype=dnp.int16)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = pd.Series([6, 50, 12], dtype=np.int16) == np.array([6, 20, 18], dtype=np.int16)
        dnpa = orca.Series([6, 50, 12], dtype=dnp.int16) == dnp.array([6, 20, 18], dtype=dnp.int16)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = np.array([6], dtype=np.int16) == pd.DataFrame({"a": [6, 5, 18]}, dtype=np.int16)
        dnpa = dnp.array([6], dtype=dnp.int16) == orca.DataFrame({"a": [6, 5, 18]}, dtype=dnp.int16)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = pd.DataFrame({"a": [6, 5, 12]}, dtype=np.int16) == np.array([6], dtype=np.int16)
        dnpa = orca.DataFrame({"a": [6, 5, 12]}, dtype=dnp.int16) == dnp.array([6], dtype=dnp.int16)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = np.array([6, 50, 12], dtype=np.int32) == np.array([6, 20, 18], dtype=np.int32)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.int32) == dnp.array([6, 20, 18], dtype=dnp.int32)
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12], dtype=np.int32) == pd.Series([6, 20, 18], dtype=np.int32)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.int32) == orca.Series([6, 20, 18], dtype=dnp.int32)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = pd.Series([6, 50, 12], dtype=np.int32) == np.array([6, 20, 18], dtype=np.int32)
        dnpa = orca.Series([6, 50, 12], dtype=dnp.int32) == dnp.array([6, 20, 18], dtype=dnp.int32)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = np.array([6], dtype=np.int32) == pd.DataFrame({"a": [6, 5, 18]}, dtype=np.int32)
        dnpa = dnp.array([6], dtype=dnp.int32) == orca.DataFrame({"a": [6, 5, 18]}, dtype=dnp.int32)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = pd.DataFrame({"a": [6, 50, 12]}, dtype=np.int32) == np.array([6], dtype=np.int32)
        dnpa = orca.DataFrame({"a": [6, 50, 12]}, dtype=dnp.int32) == dnp.array([6], dtype=dnp.int32)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = np.array([6, 50, 12], dtype=np.int64) == np.array([6, 20, 18], dtype=np.int64)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.int64) == dnp.array([6, 20, 18], dtype=dnp.int64)
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12], dtype=np.int64) == pd.Series([6, 20, 18], dtype=np.int64)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.int64) == orca.Series([6, 20, 18], dtype=dnp.int64)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = pd.Series([6, 50, 12], dtype=np.int64) == np.array([6, 20, 18], dtype=np.int64)
        dnpa = orca.Series([6, 50, 12], dtype=dnp.int64) == dnp.array([6, 20, 18], dtype=dnp.int64)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = np.array([6], dtype=np.int64) == pd.DataFrame({"a": [6, 5, 18]}, dtype=np.int64)
        dnpa = dnp.array([6], dtype=dnp.int64) == orca.DataFrame({"a": [6, 5, 18]}, dtype=dnp.int64)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = pd.DataFrame({"a": [6, 5, 12]}, dtype=np.int64) == np.array([6], dtype=np.int64)
        dnpa = orca.DataFrame({"a": [6, 5, 12]}, dtype=dnp.int64) == dnp.array([6], dtype=dnp.int64)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = np.array([6, 50, 12], dtype=np.uint8) == np.array([6, 20, 18], dtype=np.uint8)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.uint8) == dnp.array([6, 20, 18], dtype=dnp.uint8)
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12], dtype=np.uint16) == np.array([6, 20, 18], dtype=np.uint16)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.uint16) == dnp.array([6, 20, 18], dtype=dnp.uint16)
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12], dtype=np.uint32) == np.array([6, 20, 18], dtype=np.uint32)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.uint32) == dnp.array([6, 20, 18], dtype=dnp.uint32)
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12], dtype=np.uint64) == np.array([6, 20, 18], dtype=np.uint64)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.uint64) == dnp.array([6, 20, 18], dtype=dnp.uint64)
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12], dtype=np.intp) == np.array([6, 20, 18], dtype=np.intp)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.intp) == dnp.array([6, 20, 18], dtype=dnp.intp)
        assert_array_equal(dnpa, npa)

        npa = np.array([6, 50, 12], dtype=np.intp) == pd.Series([6, 20, 18], dtype=np.intp)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.intp) == orca.Series([6, 20, 18], dtype=dnp.intp)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = pd.Series([6, 50, 12], dtype=np.intp) == np.array([6, 20, 18], dtype=np.intp)
        dnpa = orca.Series([6, 50, 12], dtype=dnp.intp) == dnp.array([6, 20, 18], dtype=dnp.intp)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = np.array([6], dtype=np.intp) == pd.DataFrame({"a": [6, 5, 18]}, dtype=np.intp)
        dnpa = dnp.array([6], dtype=dnp.intp) == orca.DataFrame({"a": [6, 5, 18]}, dtype=dnp.intp)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = pd.DataFrame({"a": [6, 5, 12]}, dtype=np.intp) == np.array([6], dtype=np.intp)
        dnpa = orca.DataFrame({"a": [6, 5, 12]}, dtype=dnp.intp) == dnp.array([6], dtype=dnp.intp)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = np.array([6, 50, 12], dtype=np.uintp) == np.array([6, 20, 18], dtype=np.uintp)
        dnpa = dnp.array([6, 50, 12], dtype=dnp.uintp) == dnp.array([6, 20, 18], dtype=dnp.uintp)
        assert_array_equal(dnpa, npa)

        npa = np.array([12.0, 5.5, 15.0], dtype=np.float32) == np.array([14.7, 5.5, 6.8], dtype=np.float32)
        dnpa = dnp.array([12.0, 5.5, 15.0], dtype=dnp.float32) == dnp.array([14.7, 5.5, 6.8], dtype=dnp.float32)
        assert_array_equal(dnpa, npa)

        npa = np.array([12.0, 5.5, 15.0], dtype=np.float32) == pd.Series([14.7, 5.5, 6.8], dtype=np.float32)
        dnpa = dnp.array([12.0, 5.5, 15.0], dtype=dnp.float32) == orca.Series([14.7, 5.5, 6.8], dtype=dnp.float32)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = pd.Series([14.7, 5.5, 6.8], dtype=np.float32) == np.array([12.0, 5.5, 15.0], dtype=np.float32)
        dnpa = orca.Series([14.7, 5.5, 6.8], dtype=dnp.float32) == dnp.array([12.0, 5.5, 15.0], dtype=dnp.float32)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = np.array([12.0], dtype=np.float32) == pd.DataFrame({'A': [14.7, 5.5, 6.8]}, dtype=np.float32)
        dnpa = dnp.array([12.0], dtype=dnp.float32) == orca.DataFrame({'A': [14.7, 5.5, 6.8]}, dtype=dnp.float32)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = pd.DataFrame({'A': [14.7, 5.5, 6.8]}, dtype=np.float32) == np.array([12.0], dtype=np.float32)
        dnpa = orca.DataFrame({'A': [14.7, 5.5, 6.8]}, dtype=dnp.float32) == dnp.array([12.0], dtype=dnp.float32)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = np.array([12.0, 5.5, 15.0], dtype=np.float64) == np.array([14.7, 5.5, 6.8], dtype=np.float64)
        dnpa = dnp.array([12.0, 5.5, 15.0], dtype=dnp.float64) == dnp.array([14.7, 5.5, 6.8], dtype=dnp.float64)
        assert_array_equal(dnpa, npa)

        npa = np.array([12.0, 5.5, 15.0], dtype=np.float64) == pd.Series([14.7, 5.5, 6.8], dtype=np.float64)
        dnpa = dnp.array([12.0, 5.5, 15.0], dtype=dnp.float64) == orca.Series([14.7, 5.5, 6.8], dtype=dnp.float64)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = pd.Series([14.7, 5.5, 6.8], dtype=np.float64) == np.array([12.0, 5.5, 15.0], dtype=np.float64)
        dnpa = orca.Series([14.7, 5.5, 6.8], dtype=dnp.float64) == dnp.array([12.0, 5.5, 15.0], dtype=dnp.float64)
        assert_series_equal(dnpa.to_pandas(), npa)

        npa = np.array([12.0], dtype=np.float64) == pd.DataFrame({'A': [14.7, 5.5, 6.8]}, dtype=np.float64)
        dnpa = dnp.array([12.0], dtype=dnp.float64) == orca.DataFrame({'A': [14.7, 5.5, 6.8]}, dtype=dnp.float64)
        assert_frame_equal(dnpa.to_pandas(), npa)

        npa = pd.DataFrame({'A': [14.7, 5.5, 6.8]}, dtype=np.float64) == np.array([12.0], dtype=np.float64)
        dnpa = orca.DataFrame({'A': [14.7, 5.5, 6.8]}, dtype=dnp.float64) == dnp.array([12.0], dtype=dnp.float64)
        assert_frame_equal(dnpa.to_pandas(), npa)


if __name__ == '__main__':
    unittest.main()
