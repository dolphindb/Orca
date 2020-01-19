import unittest
import orca
import os.path as path
from setup.settings import *
from pandas.util.testing import *


class Csv:
    pdf_csv = None
    odf_csv = None


class SeriesTakeTest(unittest.TestCase):
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

    def test_series_reindexing_selection_label_mainpulation_first(self):
        ps = pd.Series([1, 2, 3, 4], index=pd.date_range('2018-04-09', periods=4, freq='2D'))
        os = orca.Series(ps)
        # TODO: orca error
        # assert_series_equal(os.first('3D').to_pandas(), ps.first('3D'))

    def test_series_reindexing_selection_label_mainpulation_last(self):
        ps = pd.Series([1, 2, 3, 4], index=pd.date_range('2018-04-09', periods=4, freq='2D'))
        os = orca.Series(ps)
        # TODO: orca error
        # assert_series_equal(os.last('3D').to_pandas(), ps.last('3D'))

    def test_series_reindexing_selection_label_mainpulation_reset_index(self):
        ps = pd.Series([1, 2, 3, 4], name='foo', index=pd.Index(['a', 'b', 'c', 'd'], name='idx'))
        os = orca.Series(ps)
        assert_frame_equal(os.reset_index().to_pandas(), ps.reset_index())
        # TODO: orca error got an unexpected keyword argument 'name'
        # assert_frame_equal(os.reset_index(name='values').to_pandas(), ps.reset_index(name='values'))
        # TODO: orca output dataframe, pandas output series
        # assert_series_equal(os.reset_index(drop=True).to_pandas(), ps.reset_index(drop=True))
        # TODO: orca Cannot reset_index inplace on a Series to create a DataFrame
        # assert_series_equal(os.reset_index(inplace=True, drop=True).to_pandas(), ps.reset_index(inplace=True, drop=True))
        arrays = [np.array(['bar', 'bar', 'baz', 'baz']),
                  np.array(['one', 'two', 'one', 'two'])]
        ps = pd.Series(range(4), name='foo', index=pd.MultiIndex.from_arrays(arrays, names=['a', 'b']))
        os = orca.Series(ps)
        assert_frame_equal(os.reset_index(level='a').to_pandas(), ps.reset_index(level='a'))

    def test_series_reindexing_selection_label_mainpulation_reset_mask(self):
        pass

    def test_series_reindexing_selection_label_mainpulation_take(self):
        n = np.array([0, 1, 4])
        assert_series_equal(self.os.take(n).to_pandas(), self.ps.take(n))
        # TODO: iloc bug
        # assert_series_equal(self.os.take([]).to_pandas(), self.ps.take([]))
        assert_series_equal(self.os.take([-1, -2], axis=0).to_pandas(), self.ps.take([-1, -2], axis=0))
        osa = orca.Series([10, 1, 19, np.nan], index=['a', 'b', 'c', 'd'])
        psa = pd.Series([10, 1, 19, np.nan], index=['a', 'b', 'c', 'd'])
        assert_series_equal(osa.take([3]).to_pandas(), psa.take([3]))

    def test_series_reindexing_selection_label_manipulation_idxmax(self):
        pser = pd.Series(data=[1, 4, 5], index=['A', 'B', 'C'])
        oser = orca.Series(pser)

        self.assertEqual(oser.idxmax(), pser.idxmax())
        self.assertEqual(oser.idxmax(skipna=False), pser.idxmax(skipna=False))

        index = pd.MultiIndex.from_arrays([
            ['a', 'a', 'b', 'b'], ['c', 'd', 'e', 'f']], names=('first', 'second'))
        pser = pd.Series(data=[1, 2, 4, 5], index=index)
        oser = orca.Series(pser)

        # TODO: multiIndex.idxmax()
        # self.assertEqual(oser.idxmax(), pser.idxmax())
        # self.assertEqual(oser.idxmax(skipna=False), pser.idxmax(skipna=False))
        #
        # oser = orca.Series([])
        # with self.assertRaisesRegex(ValueError, "an empty sequence"):
        #     oser.idxmax()

    def test_series_reindexing_selection_label_manipulation_idxmin(self):
        pser = pd.Series(data=[1, 4, 5], index=['A', 'B', 'C'])
        oser = orca.Series(pser)

        self.assertEqual(oser.idxmin(), pser.idxmin())
        # self.assertEqual(oser.idxmin(skipna=False), pser.idxmin(skipna=False))
        #
        # index = pd.MultiIndex.from_arrays([
        #     ['a', 'a', 'b', 'b'], ['c', 'd', 'e', 'f']], names=('first', 'second'))
        # pser = pd.Series(data=[1, 2, 4, 5], index=index)
        # oser = orca.Series(pser)

        # TODO: multiIndex.idxmin()
        # self.assertEqual(oser.idxmin(), pser.idxmin())
        # self.assertEqual(oser.idxmin(skipna=False), pser.idxmin(skipna=False))
        #
        # oser = orca.Series([])
        # with self.assertRaisesRegex(ValueError, "an empty sequence"):
        #     oser.idxmin()

    def test_series_reindexing_selection_label_manipulation_duplicated(self):
        s = pd.Series(['lama', 'cow', 'lama', 'beetle', 'lama', 'hippo'], name='animal')
        ds = orca.Series(s)
        self.assertEqual(repr(ds.duplicated().to_pandas()), repr(s.duplicated()))

        self.assertEqual(repr(ds.duplicated(keep='first').to_pandas()), repr(s.duplicated(keep='first')))

        self.assertEqual(repr(ds.duplicated(keep='last').to_pandas()), repr(s.duplicated(keep='last')))

        self.assertEqual(repr(ds.duplicated(keep=False).to_pandas()), repr(s.duplicated(keep=False)))

    def test_series_reindexing_selection_label_manipulation_drop_duplicates(self):
        s = pd.Series(['lama', 'cow', 'lama', 'beetle', 'lama', 'hippo'], name='animal')
        ds = orca.Series(s)
        assert_series_equal(ds.drop_duplicates().to_pandas(), s.drop_duplicates())
        assert_series_equal(ds.drop_duplicates(keep='first').to_pandas(), s.drop_duplicates(keep='first'))
        assert_series_equal(ds.drop_duplicates(keep='last').to_pandas(), s.drop_duplicates(keep='last'))
        assert_series_equal(ds.drop_duplicates(keep=False).to_pandas(), s.drop_duplicates(keep=False))
        # TODO: series.drop_duplicates 不支持inplace参数
        # assert_series_equal(ds.drop_duplicates(inplace=True).to_pandas(), s.drop_duplicates(inplace=True), check_names=False)
        # s = pd.Series(['lama', 'cow', 'lama', 'beetle', 'lama', 'hippo'], name='animal')
        # ds = orca.Series(s)
        # assert_series_equal(ds.drop_duplicates(keep='last', inplace=True).to_pandas(), s.drop_duplicates(keep='last', inplace=True))
        # s = pd.Series(['lama', 'cow', 'lama', 'beetle', 'lama', 'hippo'], name='animal')
        # ds = orca.Series(s)
        # assert_series_equal(ds.drop_duplicates(keep=False, inplace=True).to_pandas(), s.drop_duplicates(keep=False, inplace=True))

    def test_series_reindexing_selection_label_manipulation_head_tail(self):
        ps = self.ps
        os = self.os
        assert_series_equal(ps, os.to_pandas())

        # head
        assert_series_equal(ps.head(), os.head().to_pandas())
        assert_series_equal(ps.head(10), os.head(10).to_pandas())
        assert_series_equal(ps.head(7), os.head(7).to_pandas())
        assert_series_equal(ps.head(5), os.head(5).to_pandas())
        assert_series_equal(ps.head(3), os.head(3).to_pandas())
        # TODO: orca.Series.head(0)
        # assert_series_equal(ps.head(0), os.head(0).to_pandas())
        assert_series_equal(ps.head(-3), os.head(-3).to_pandas())
        assert_series_equal(ps[ps > 3].head(3), os[os > 3].head(3).to_pandas())
        # TODO: orca.ArithExpression.tail(-3)
        # assert_series_equal((ps+1).head(-3), (os+1).head(-3).to_pandas())

        # tail
        assert_series_equal(ps.tail(), os.tail().to_pandas())
        assert_series_equal(ps.tail(10), os.tail(10).to_pandas())
        assert_series_equal(ps.tail(7), os.tail(7).to_pandas())
        assert_series_equal(ps.tail(5), os.tail(5).to_pandas())
        assert_series_equal(ps.tail(3), os.tail(3).to_pandas())
        # TODO: orca.Series.tail(0)
        # assert_series_equal(ps.tail(0), os.tail(0).to_pandas())
        assert_series_equal(ps.tail(-3), os.tail(-3).to_pandas())
        assert_series_equal(ps[ps > 3].tail(3), os[os > 3].tail(3).to_pandas())
        # TODO: orca.ArithExpression.tail(-3)
        # assert_series_equal((ps+1).tail(-3), (os+1).tail(-3).to_pandas())

    def test_series_reindexing_selection_label_manipulation_rename(self):
        ps = pd.Series([1, 2, 3, 4, 5, 6, 7], name='x')
        os = orca.Series(ps)

        ps.name = 'renamed'
        os.name = 'renamed'
        self.assertEqual(os.name, 'renamed')
        assert_series_equal(ps, os.to_pandas())

        pidx = ps.index
        oidx = os.index
        pidx.name = 'renamed'
        oidx.name = 'renamed'
        self.assertEqual(oidx.name, 'renamed')
        assert_index_equal(pidx, oidx.to_pandas())

        # TODO: orca.Series.rename('rename')
        # assert_series_equal(ps.rename('y'), os.rename('y').to_pandas())
        # self.assertEqual(os.name, 'renamed')  # no mutation
        # assert_series_equal(ps.rename(), os.rename().to_pandas())
        # os.rename('z', inplace=True)
        # ps.rename('z', inplace=True)
        # self.assertEqual(os.name, 'z')
        # assert_series_equal(ps, os.to_pandas())

    def test_series_reindexing_selection_label_manipulation_rename_method(self):
        # Series name
        ps = pd.Series([1, 2, 3, 4, 5, 6, 7], name='x')
        os = orca.Series(ps)
        assert_series_equal(ps.rename("a"), os.rename("a").to_pandas())

    def test_series_reindexing_selection_label_manipulation_isin(self):
        ps = pd.Series(['lama', 'cow', 'lama', 'beetle', 'lama', 'hippo'], name='animal')
        os = orca.Series(ps)
        assert_series_equal(os.isin(['cow', 'lama']).to_pandas(), ps.isin(['cow', 'lama']))
        assert_series_equal(os.isin({'cow'}).to_pandas(), ps.isin({'cow'}))
        # series name
        # assert_series_equal(os.isin(orca.Series(['cow', 'lama'])).to_pandas(), ps.isin(pd.Series(['cow', 'lama'])))