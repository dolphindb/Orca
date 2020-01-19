import unittest
from setup.settings import *
from numpy.testing import *
from pandas.util.testing import *
import numpy as np
import dolphindb_numpy as dnp
import pandas as pd
import orca


class FunctionLogicalNotTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_function_math_binary_logical_not_array_with_array(self):
        npa1 = np.array([0, 1, 2])
        npa2 = np.array([4, 6, 9])

        dnpa1 = dnp.array([0, 1, 2])
        dnpa2 = dnp.array([4, 6, 9])
        assert_array_equal(dnp.logical_not(dnpa1, dnpa2), np.logical_not(npa1, npa2))

    def test_function_math_binary_logical_not_array_with_series(self):
        npa = np.array([0, 1, 2])
        dnpa = dnp.array([0, 1, 2])
        ps = pd.Series([4, 6, 9])
        os = orca.Series([4, 6, 9])
        # NUMPY NOT SUPPORT
        # assert_series_equal(dnp.logical_not(dnpa, os).to_pandas(), np.logical_not(npa, ps))
        # assert_series_equal(dnp.logical_not(os, dnpa).to_pandas(), np.logical_not(ps, npa))
        #
        # pser = pd.Series([1, 2, 4])
        # oser = orca.Series([1, 2, 4])
        # assert_series_equal(dnp.logical_not(os, oser).to_pandas(), np.logical_not(ps, pser))

    def test_function_math_binary_logical_not_array_with_dataframe(self):
        npa = np.array([[1, 1, 1], [1, 0, 1], [2, 0, 0]])
        dnpa = dnp.array([[1, 1, 1], [1, 0, 1], [2, 0, 0]])
        pdf = pd.DataFrame({'A': [4, 6, 9]})
        odf = orca.DataFrame({'A': [4, 6, 9]})
        # TODO: ORCA BUG
        # assert_frame_equal(dnp.logical_not(odf, dnpa).to_pandas(), np.logical_not(pdf, npa))
        # assert_frame_equal(dnp.logical_not(dnpa, odf).to_pandas(), np.logical_not(npa, pdf))
        #
        # pdfrm = pd.DataFrame({'A': [0, 7, 1]})
        # odfrm = orca.DataFrame({'A': [0, 7, 1]})
        # assert_frame_equal(dnp.logical_not(odf, odfrm).to_pandas(), np.logical_not(pdf, pdfrm))


if __name__ == '__main__':
    unittest.main()
