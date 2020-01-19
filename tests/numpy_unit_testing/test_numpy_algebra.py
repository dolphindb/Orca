import unittest
import orca
import os.path as path
from setup.settings import *
from pandas.util.testing import *
from numpy.distutils.tests import *
import pandas as pd
import dolphindb_numpy as dnp
import numpy as np


class AlgebraTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_dot(self):
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[11, 12], [13, 14]])
        np.dot(a, b)

    def test_trigonometric_functions(self):
        # print(np.arccos([1, -1]))
        dnp.arccos([1, -1])
        dt = orca.DataFrame([1,-1])
        dnp.arccos(dt).to_pandas()

    def test_clip(self):
        x = np.array([[1, 2, 3, 5, 6, 7, 8, 9], [0, 2, 3, 5, 6, 7, 8, 9]])
        # print(dnp.clip(x, 3, 8))
        dt = orca.DataFrame([[1, 2, 3, 5, 6, 7, 8, 9], [1, 2, 3, 5, 6, 7, 8, 9]])
        odf = orca.DataFrame(
            {'float': [1.0, 2.0, 3.5, 6.5], 'int': [1, 2, 7, 4], 'datetime': pd.date_range('2019-01-02', periods=4),
             'string': ['foo', 'ss', 'sw', 'qa']}, index=['a', 'b', 'c', 'c'])
        # np.isnan()
        # print(np.min(x))
        # print(odf.min(axis=1))

        x.min(axis=0)

        # print(dnp.clip(dt, 3, 8).to_pandas())


