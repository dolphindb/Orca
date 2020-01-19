import unittest
import orca
import os.path as path
from setup.settings import *
from pandas.util.testing import *


class Csv:
    pdf_csv = None
    odf_csv = None


class SeriesTest(unittest.TestCase):
    def setUp(self):
        self.PRECISION = 5

    @classmethod
    def setUpClass(cls):
        # configure data directory
        DATA_DIR = path.abspath(path.join(__file__, "../setup/data"))
        fileName = 'USPricesSample.csv'
        data = os.path.join(DATA_DIR, fileName)
        data = data.replace('\\', '/')

        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

        Csv.pdf_csv = pd.read_csv(data)
        Csv.odf_csv = orca.read_csv(data)

    @property
    def ps(self):
        return pd.Series([1, 2, 3, 4, 5, 6, 7], name='x')

    @property
    def os(self):
        return orca.Series([1, 2, 3, 4, 5, 6, 7], name='x')

    @property
    def psa(self):
        return pd.Series([10, 1, 19, np.nan], index=['a', 'b', 'c', 'd'])

    @property
    def psb(self):
        return pd.Series([-1, np.nan, 1, np.nan], index=['a', 'b', 'd', 'e'])

    def test_series_computations_descriptive_stats_all(self):
        for ps in [pd.Series([True, True], name='x'),
                   pd.Series([True, False], name='x'),
                   pd.Series([0, 1], name='x'),
                   pd.Series([1, 2, 3], name='x'),
                   pd.Series([], name='x'),
                   pd.Series([np.nan], name='x')]:
            os = orca.Series(ps)
            self.assertEqual(os.all(), ps.all())

        ps = pd.Series([1, 2, 3, 4], name='x')
        os = orca.Series(ps)

        self.assertEqual((os % 2 == 0).all(), (ps % 2 == 0).all())

    def test_series_computations_descriptive_stats_any(self):
        for ps in [pd.Series([False, False], name='x'),
                   pd.Series([True, False], name='x'),
                   pd.Series([0, 1], name='x'),
                   pd.Series([1, 2, 3], name='x'),
                   pd.Series([], name='x'),
                   pd.Series([np.nan], name='x')]:
            os = orca.Series(ps)
            self.assertEqual(os.any(), ps.any())

        ps = pd.Series([1, 2, 3, 4], name='x')
        os = orca.Series(ps)

        self.assertEqual((os % 2 == 0).any(), (ps % 2 == 0).any())

    def test_series_computations_descriptive_stats_clip(self):
        ps = pd.Series([0, 2, 4])
        os = orca.DataFrame(ps)
        # TODO：different data structure
        # Assert no lower or upper
        # assert_series_equal(os.clip().to_pandas(), ps.clip())
        # # Assert lower only
        # assert_series_equal(os.clip(1).to_pandas(), ps.clip(1))
        # # Assert upper only
        # assert_series_equal(os.clip(upper=3).to_pandas(), ps.clip(upper=3))
        # # Assert lower and upper
        # assert_series_equal(os.clip(1, 3).to_pandas(), ps.clip(1, 3))
        #
        # # Assert behavior on string values
        # str_os = orca.Series(['a', 'b', 'c'])
        # assert_series_equal(str_os.clip(1, 3).to_pandas(), str_os)

    def test_series_computations_descriptive_stats_cum(self):
        ps = pd.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        os = orca.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])

        assert_series_equal(ps.cumsum(), os.cumsum().to_pandas())
        assert_series_equal(ps.cummax(), os.cummax().to_pandas())
        assert_series_equal(ps.cummin(), os.cummin().to_pandas())
        assert_series_equal(ps.cumprod(), os.cumprod().to_pandas())

    def test_series_computations_descriptive_stats_describe(self):
        ps = pd.Series([1, 2, 3])
        os = orca.Series(ps)
        assert_series_equal(os.describe(), ps.describe())

        ps = pd.Series(['a', 'a', 'b', 'c'])
        os = orca.Series(ps)
        assert_series_equal(os.describe(), ps.describe())

        ps = pd.Series([np.datetime64("2000-01-01"), np.datetime64("2010-01-01"),np.datetime64("2010-01-01")])
        os = orca.Series(ps)
        assert_series_equal(os.describe(), ps.describe())

    def test_series_computations_descriptive_stats_is_unique(self):
        # We can't use pandas' is_unique for comparison. pandas 0.23 ignores None
        pser = pd.Series([1, 2, 2, None, None])
        oser = orca.Series(pser)
        self.assertEqual(False, oser.is_unique)

        pser = pd.Series([1, None, None])
        oser = orca.Series(pser)
        self.assertEqual(False, oser.is_unique)

        pser = pd.Series([1])
        oser = orca.Series(pser)
        self.assertEqual(pser.is_unique, oser.is_unique)

        pser = pd.Series([1, 1, 1])
        oser = orca.Series(pser)
        self.assertEqual(pser.is_unique, oser.is_unique)

    def test_series_computations_descriptive_stats_mean(self):
        ps = pd.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        os = orca.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        self.assertEqual(ps.mean(), os.mean())

        pidx = pd.MultiIndex.from_arrays([['warm', 'warm', 'cold', 'cold'], ['dog', 'falcon', 'fish', 'spider']],
                                         names=['blooded', 'animal'])
        ps = pd.Series([4, 2, 0, 8], name='legs', index=pidx)
        oidx = orca.MultiIndex.from_arrays([['warm', 'warm', 'cold', 'cold'], ['dog', 'falcon', 'fish', 'spider']],
                                           names=['blooded', 'animal'])
        os = orca.Series([4, 2, 0, 8], name='legs', index=oidx)

        self.assertEqual(ps.mean(), os.mean())
        # TODO:Series.mean() should provide more parameters
        # self.assertEqual(ps.mean(level='blooded'), os.mean(level='blooded'))
        # self.assertEqual(ps.mean(level=0), os.mean(level=0))

    def test_series_computations_descriptive_stats_nlargest(self):
        sample_lst = [1, 2, 3, 4, np.nan, 6]
        ps = pd.Series(sample_lst, name='x')
        os = orca.Series(sample_lst, name='x')
        # TODO：NOT IMPLEMENTED
        # self.assertEqual(os.nlargest(n=3), ps.nlargest(n=3))
        # self.assertEqual(os.nlargest(), ps.nlargest())

    def test_series_computations_descriptive_stats_nsmallest(self):
        sample_lst = [1, 2, 3, 4, np.nan, 6]
        ps = pd.Series(sample_lst, name='x')
        os = orca.Series(sample_lst, name='x')
        # TODO：NOT IMPLEMENTED
        # self.assertEqual(os.nsmallest(n=3), ps.nsmallest(n=3))
        # self.assertEqual(os.nsmallest(), ps.nsmallest())

    def test_series_computations_descriptive_stats_nunique(self):
        ps = pd.Series([1, 2, 1, np.nan])
        os = orca.DataFrame(ps)

        # Assert NaNs are dropped by default
        nunique_result = os.nunique()
        self.assertEqual(nunique_result, 2)
        self.assertEqual(nunique_result, ps.nunique())

        # TODO：NOT IMPLEMENTED
        # # Assert including NaN values
        # nunique_result = os.nunique(dropna=False)
        # self.assertEqual(nunique_result, 3)
        # self.assertEqual(nunique_result, ps.nunique(dropna=False))
        #
        # # Assert approximate counts
        # self.assertEqual(orca.Series(range(100)).nunique(approx=True), 103)
        # self.assertEqual(orca.Series(range(100)).nunique(approx=True, rsd=0.01), 100)

    def test_series_computations_descriptive_stats_quantile(self):
        self.assertAlmostEqual(orca.Series([24., 21., 25., 33., 26.]).quantile(q=0.57),
                               pd.Series([24., 21., 25., 33., 26.]).quantile(q=0.57), self.PRECISION)
        assert_series_equal(orca.Series([24., 21., 25., 33., 26.]).quantile(q=[0.57, 0.5, 0.23, 0.94]).to_pandas(),
                               pd.Series([24., 21., 25., 33., 26.]).quantile(q=[0.57, 0.5, 0.23, 0.94]))

    def test_series_computations_descriptive_stats_rank(self):
        pser = pd.Series([1, 2, 3, 1], name='x')
        oser = orca.DataFrame(pser)
        pser.rank()
        oser.rank()
        # TODO：DIFFERENT MEHOD
        # self.assertEqual(repr(pser.rank()),
        #                  repr(oser.rank().sort_index()))
        # self.assertEqual(repr(pser.rank()),
        #                  repr(oser.rank().sort_index()))
        # self.assertEqual(repr(pser.rank(ascending=False)),
        #                  repr(oser.rank(ascending=False).sort_index()))
        # self.assertEqual(repr(pser.rank(method='min')),
        #                  repr(oser.rank(method='min').sort_index()))
        # self.assertEqual(repr(pser.rank(method='max')),
        #                  repr(oser.rank(method='max').sort_index()))
        # self.assertEqual(repr(pser.rank(method='first')),
        #                  repr(oser.rank(method='first').sort_index()))
        # self.assertEqual(repr(pser.rank(method='dense')),
        #                  repr(oser.rank(method='dense').sort_index()))
        #
        # msg = "method must be one of 'average', 'min', 'max', 'first', 'dense'"
        # with self.assertRaisesRegex(ValueError, msg):
        #     oser.rank(method='nothing')

    def test_series_computations_mad_sem_skew_kurt_kurtosis(self):
        ps = pd.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        os = orca.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        assert_series_equal(ps.pct_change(), os.pct_change().to_pandas())
        self.assertEqual(ps.mad(), os.mad())
        self.assertEqual(ps.sem(), os.sem())
        self.assertAlmostEqual(ps.skew(), os.skew(), self.PRECISION)
        self.assertAlmostEqual(ps.kurt(), os.kurt(), self.PRECISION)
        # self.assertAlmostEqual(ps.kurtosis(), os.kurtosis(), self.PRECISION)

    def test_series_computations_descriptive_stats_std(self):
        ps = pd.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        os = orca.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        self.assertEqual(ps.var(), os.var())

        pidx = pd.MultiIndex.from_arrays([['warm', 'warm', 'cold', 'cold'], ['dog', 'falcon', 'fish', 'spider']],
                                         names=['blooded', 'animal'])
        ps = pd.Series([4, 2, 0, 8], name='legs', index=pidx)
        oidx = orca.MultiIndex.from_arrays([['warm', 'warm', 'cold', 'cold'], ['dog', 'falcon', 'fish', 'spider']],
                                           names=['blooded', 'animal'])
        os = orca.Series([4, 2, 0, 8], name='legs', index=oidx)
        self.assertEqual(ps.std(), os.std())
        # TODO:Series.std() should provide more parameters
        # self.assertEqual(ps.std(level='blooded'), os.std(level='blooded'))
        # self.assertEqual(ps.std(level=0), os.std(level=0))

    def test_series_computations_descriptive_stats_sum(self):
        ps = pd.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        os = orca.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        self.assertEqual(ps.sum(), os.sum())

        pidx = pd.MultiIndex.from_arrays([['warm', 'warm', 'cold', 'cold'], ['dog', 'falcon', 'fish', 'spider']],
                                         names=['blooded', 'animal'])
        ps = pd.Series([4, 2, 0, 8], name='legs', index=pidx)
        oidx = orca.MultiIndex.from_arrays([['warm', 'warm', 'cold', 'cold'], ['dog', 'falcon', 'fish', 'spider']],
                                           names=['blooded', 'animal'])
        os = orca.Series([4, 2, 0, 8], name='legs', index=oidx)

        self.assertEqual(ps.sum(), os.sum())
        # TODO:Series.sum() should provide more parameters
        # self.assertEqual(ps.sum(min_count=1), os.sum(min_count=1))
        # self.assertEqual(ps.sum(level='blooded'), os.sum(level='blooded'))
        # self.assertEqual(ps.sum(level=0), os.sum(level=0))

    def test_series_computations_descriptive_stats_unique(self):
        pser = pd.Series([1, 2, 2, None, None])
        oser = orca.Series(pser)
        self.assertEqual(repr(oser.unique()), repr(pser.unique()))

        ps = pd.Series([2, 1, 3, 3], name='A')
        os = orca.Series(ps)
        self.assertEqual(repr(os.unique()), repr(ps.unique()))

        ps = pd.Series([pd.Timestamp('2016-01-01') for _ in range(3)])
        os = orca.Series(ps)
        self.assertEqual(repr(os.unique()), repr(ps.unique()))

        # TODO: orca暂不支持的dtype：指定时区的datetime64和category
        # ps = pd.Series([pd.Timestamp('2016-01-01', tz='US/Eastern') for _ in range(3)])
        # os = orca.Series(ps)
        # self.assertEqual(repr(os.unique()), repr(ps.unique()))
        #
        # ps = pd.Series(pd.Categorical(list('baabc')))
        # os = orca.Series(ps)
        # self.assertEqual(repr(os.unique()), repr(ps.unique()))
        #
        # ps = pd.Series(pd.Categorical(list('baabc'), categories=list('abc'), ordered=True))
        # os = orca.Series(ps)
        # self.assertEqual(repr(os.unique()), repr(ps.unique()))

    def test_series_computations_descriptive_stats_var(self):
        ps = pd.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        os = orca.Series([10, 1, 19, -5], index=['a', 'b', 'c', 'd'])
        self.assertEqual(ps.var(), os.var())

        pidx = pd.MultiIndex.from_arrays([['warm', 'warm', 'cold', 'cold'], ['dog', 'falcon', 'fish', 'spider']],
                                         names=['blooded', 'animal'])
        ps = pd.Series([4, 2, 0, 8], name='legs', index=pidx)

        oidx = orca.MultiIndex.from_arrays([['warm', 'warm', 'cold', 'cold'], ['dog', 'falcon', 'fish', 'spider']],
                                           names=['blooded', 'animal'])
        os = orca.Series([4, 2, 0, 8], name='legs', index=oidx)

        self.assertEqual(ps.var(), os.var())
        # TODO:Series.var() should provide more parameters
        # self.assertEqual(ps.var(level='blooded'), os.var(level='blooded'))
        # self.assertEqual(ps.var(level=0), os.var(level=0))

    def test_series_computations_value_counts(self):
        ps = pd.Series([10, 1, 19, -5, -5, -5, np.nan])
        os = orca.Series(ps)
        # TODO: different sort rules
        # assert_series_equal(ps.value_counts(), os.value_counts().to_pandas())
