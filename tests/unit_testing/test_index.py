import unittest
import orca
from setup.settings import *
from pandas.util.testing import *


class IndexTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    def test_index(self):
        pi = pd.Index([1, 2, 3])
        oi = orca.Index([1, 2, 3])
        assert_index_equal(pi, oi.to_pandas())

    def test_index_properties_values(self):
        pi = pd.Index([1, 2, 3])
        oi = orca.Index([1, 2, 3])
        self.assertEqual(repr(pi.values), repr(oi.values))

    def test_index_properties_is_monotonic(self):
        pi = pd.Index([1, 2, 3])
        oi = orca.Index([1, 2, 3])
        self.assertEqual(pi.is_monotonic, oi.is_monotonic)

    def test_index_properties_is_monotonic_increasing(self):
        pi = pd.Index([1, 2, 3])
        oi = orca.Index([1, 2, 3])
        self.assertEqual(pi.is_monotonic_increasing, oi.is_monotonic_increasing)

    def test_index_properties_is_monotonic_decreasing(self):
        pi = pd.Index([1, 2, 3])
        oi = orca.Index([1, 2, 3])
        self.assertEqual(pi.is_monotonic_decreasing, oi.is_monotonic_decreasing)

    def test_index_properties_is_unique(self):
        pi = pd.Index([1, 2, 3])
        oi = orca.Index([1, 2, 3])
        # FIXME：BUG
        # self.assertEqual(pi.is_unique, oi.is_unique)

        pi = pd.Index([1, 1, 1])
        oi = orca.Index([1, 1, 1])
        # self.assertEqual(pi.is_unique, oi.is_unique)

    def test_index_properties_has_duplicates(self):
        pi = pd.Index([1, 2, 3])
        oi = orca.Index([1, 2, 3])
        # TODO：NOT IMPLEMENTED
        # self.assertEqual(pi.has_duplicates, oi.has_duplicates)

        pi = pd.Index([1, 1, 1])
        oi = orca.Index([1, 1, 1])
        # TODO：NOT IMPLEMENTED
        # self.assertEqual(pi.has_duplicates, oi.has_duplicates)

    def test_index_properties_is_hasnans(self):
        pi = pd.Index([1, 2, 3])
        oi = orca.Index([1, 2, 3])
        self.assertEqual(pi.hasnans, oi.hasnans)

        pi = pd.Index([1, np.nan, 1])
        oi = orca.Index([1, np.nan, 1])
        self.assertEqual(pi.hasnans, oi.hasnans)

    def test_index_properties_dtype(self):
        pi = pd.Index([1, 2, 3])
        oi = orca.Index([1, 2, 3])
        self.assertEqual(pi.dtype, oi.dtype)

        pi = pd.Index([1.0, 2, 3])
        oi = pd.Index([1.0, 2, 3])
        self.assertEqual(pi.dtype, oi.dtype)

        pi = pd.Index(['a', 'b'], dtype=str)
        oi = pd.Index(['a', 'b'], dtype=str)
        self.assertEqual(pi.dtype, oi.dtype)

        pi = pd.Index([pd.to_datetime('1/1/2018'), pd.to_datetime('2/1/2018')])
        oi = pd.Index([pd.to_datetime('1/1/2018'), pd.to_datetime('2/1/2018')])
        self.assertEqual(pi.dtype, oi.dtype)
        # dtype.str
        pi = pd.Index([1, 2, 3])
        oi = orca.Index([1, 2, 3])
        self.assertEqual(pi.dtype.str, oi.dtype.str)

        pi = pd.Index([1.0, 2, 3])
        oi = pd.Index([1.0, 2, 3])
        self.assertEqual(pi.dtype.str, oi.dtype.str)

        pi = pd.Index(['a', 'b'], dtype=str)
        oi = pd.Index(['a', 'b'], dtype=str)
        self.assertEqual(pi.dtype.str, oi.dtype.str)

        pi = pd.Index([pd.to_datetime('1/1/2018'), pd.to_datetime('2/1/2018')])
        oi = pd.Index([pd.to_datetime('1/1/2018'), pd.to_datetime('2/1/2018')])
        self.assertEqual(pi.dtype.str, oi.dtype.str)

    def test_index_properties_dtype_str(self):
        # deprecated
        pass

    def test_index_properties_inferred_type(self):
        pi = pd.Index([1, 2, 3])
        oi = orca.Index([1, 2, 3])
        # TODO：NOT IMPLEMENTED
        # self.assertEqual(pi.inferred_type, oi.inferred_type)

        pi = pd.Index([1.0, 2, 3])
        oi = orca.Index([1.0, 2, 3])
        # self.assertEqual(pi.inferred_type, oi.inferred_type)

        pi = pd.Index(['a', 'b'], dtype=str)
        oi = orca.Index(['a', 'b'], dtype=str)
        # self.assertEqual(pi.inferred_type, oi.inferred_type)

        pi = pd.Index([pd.to_datetime('1/1/2018'), pd.to_datetime('2/1/2018')])
        oi = orca.Index([pd.to_datetime('1/1/2018'), pd.to_datetime('2/1/2018')])
        # self.assertEqual(pi.inferred_type, oi.inferred_type)

    def test_index_properties_is_all_dates(self):
        pi = pd.Index([1, 2, 3])
        oi = orca.Index([1, 2, 3])
        self.assertEqual(pi.is_all_dates, oi.is_all_dates)

        pi = pd.Index([1.0, 2, 3])
        oi = pd.Index([1.0, 2, 3])
        self.assertEqual(pi.is_all_dates, oi.is_all_dates)

        pi = pd.Index(['a', 'b'], dtype=str)
        oi = pd.Index(['a', 'b'], dtype=str)
        self.assertEqual(pi.is_all_dates, oi.is_all_dates)

        pi = pd.Index([pd.to_datetime('1/1/2018'), pd.to_datetime('2/1/2018')])
        oi = pd.Index([pd.to_datetime('1/1/2018'), pd.to_datetime('2/1/2018')])
        self.assertEqual(pi.is_all_dates, oi.is_all_dates)

    def test_index_properties_shape(self):
        pi = pd.Index([1, 2, 3])
        oi = orca.Index([1, 2, 3])
        self.assertEqual(pi.shape, oi.shape)

    def test_index_properties_name(self):
        pi = pd.Index([1, 2, 3], name=['a', 'b', 'c'])
        oi = orca.Index([1, 2, 3], name=['a', 'b', 'c'])
        self.assertEqual(pi.name, oi.name)

    def test_index_properties_names(self):
        pi = pd.Index([1, 2, 3], name=['a', 'b', 'c'])
        oi = orca.Index([1, 2, 3], name=['a', 'b', 'c'])
        # TODO：NOT IMPLEMENTED
        # self.assertEqual(pi.names, oi.names)

    def test_index_properties_nbytes(self):
        pi = pd.Index([1, 2, 3], name=['a', 'b', 'c'])
        oi = orca.Index([1, 2, 3], name=['a', 'b', 'c'])
        self.assertEqual(pi.nbytes, oi.nbytes)

    def test_index_properties_ndim(self):
        pi = pd.Index([1, 2, 3], name=['a', 'b', 'c'])
        oi = orca.Index([1, 2, 3], name=['a', 'b', 'c'])
        self.assertEqual(pi.ndim, oi.ndim)

    def test_index_properties_size(self):
        pi = pd.Index([1, 2, 3], name=['a', 'b', 'c'])
        oi = orca.Index([1, 2, 3], name=['a', 'b', 'c'])
        self.assertEqual(pi.size, oi.size)

    def test_index_properties_empty(self):
        pi = pd.Index([1, 2, 3], name=['a', 'b', 'c'])
        oi = orca.Index([1, 2, 3], name=['a', 'b', 'c'])
        # TODO：NOT IMPLEMENTED
        # self.assertEqual(pi.empty, oi.empty)

    def test_index_properties_strides(self):
        # deprecated
        pass

    def test_index_properties_itemsize(self):
        # deprecated
        pass

    def test_index_properties_base(self):
        # deprecated
        pass

    def test_index_properties_T(self):
        pi = pd.Index([1, 2, 3], name='a')
        oi = orca.Index([1, 2, 3], name='a')
        assert_index_equal(pi.T, oi.T.to_pandas())

    def test_index_properties_memory_usage(self):
        pi = pd.Index([1, 2, 3], name=['a', 'b', 'c'])
        oi = orca.Index([1, 2, 3], name=['a', 'b', 'c'])
        # TODO：NOT IMPLEMENTED
        # self.assertEqual(pi.memory_usage, oi.memory_usage)


if __name__ == '__main__':
    unittest.main()
