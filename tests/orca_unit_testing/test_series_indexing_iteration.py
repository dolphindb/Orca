from setup.settings import *
import unittest
import orca
import os.path as path
from pandas.util.testing import *


class SeriesIndexingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # configure data directory
        DATA_DIR = path.abspath(path.join(__file__, "../setup/data"))
        fileName = 'USPricesSample.csv'
        data = os.path.join(DATA_DIR, fileName)
        data = data.replace('\\', '/')

        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    @property
    def ps_numerical(self):
        return pd.Series([1, 2, 3, 4, 5, 6, 7], name='x')

    @property
    def os_numerical(self):
        return orca.Series(self.ps_numerical)

    @property
    def ps_literal(self):
        return pd.Series(['lama', 'cow', 'lama', 'beetle', 'lama', 'hippo', 'beetle'], name='animal')

    @property
    def os_literal(self):
        return orca.Series(self.ps_literal)

    @property
    def ps_date(self):
        return pd.Series(pd.date_range("20190101", "20190107", freq="d"))

    @property
    def os_date(self):
        return orca.Series(self.ps_date)

    @property
    def pidx_literal(self):
        return pd.Index(['a', 'a', 'c', 's', 's', 'b', 'd'])

    @property
    def oidx_literal(self):
        return orca.Index(self.pidx_literal)

    def test_series_indexing_array(self):
        ps = self.ps_literal
        os = self.ps_literal
        assert_numpy_array_equal(ps.__array__(), os.__array__(), check_dtype=False)

    def test_series_indexing_at_get(self):
        ps = self.ps_literal
        os = self.os_literal
        self.assertEqual(os.at[0], ps.at[0])

        ps = self.ps_numerical
        os = self.os_numerical
        self.assertEqual(os.at[0], ps.at[0])

        ps = self.ps_date
        os = self.os_date
        self.assertEqual(os.at[0], ps.at[0])

    def test_series_indexing_at_set(self):
        ps = self.ps_literal
        os = self.os_literal

        os.at[0] = "a"
        ps.at[0] = "a"
        assert_series_equal(os.to_pandas(), ps)

        ps = self.ps_numerical
        os = self.os_numerical
        os.at[0] = 12
        ps.at[0] = 12
        assert_series_equal(os.to_pandas(), ps)

        # TODO: NOT IMPLEMENTED
        # ps = self.ps_date
        # os = self.os_date
        # os.at[0] = orca.Timestamp("20200101")
        # ps.at[0] = pd.Timestamp("20200101")
        # assert_series_equal(os.to_pandas(), ps)

    def test_series_indexing_iat_get(self):
        ps = self.ps_literal
        os = self.ps_literal
        self.assertEqual(os.iat[0], ps.iat[0])

        ps = self.ps_numerical
        os = self.os_numerical
        self.assertEqual(os.iat[0], ps.iat[0])

        ps = self.ps_date
        os = self.os_date
        self.assertEqual(os.iat[0], ps.iat[0])

    def test_series_indexing_iat_set(self):
        ps = self.ps_literal
        os = self.os_literal

        os.iat[0] = "a"
        ps.iat[0] = "a"
        assert_series_equal(os.to_pandas(), ps)

        ps = self.ps_numerical
        os = self.os_numerical
        os.iat[0] = 12
        ps.iat[0] = 12
        assert_series_equal(os.to_pandas(), ps)

        # TODO: NOT IMPLEMENTED
        # ps = self.ps_date
        # os = self.os_date
        # os.iat[0] = orca.Timestamp("20200101")
        # ps.iat[0] = pd.Timestamp("20200101")
        # assert_series_equal(os.to_pandas(), ps)

    def test_series_indexing_iter(self):
        s = pd.Series(range(5, 10))
        ors = orca.Series(s)
        self.assertEqual(repr(list(s.__iter__())), repr(list(ors.__iter__())))

        # iter with name
        s = pd.Series(['lama', 'cow', 'lama', 'beetle', 'lama', 'hippo'], name='animal')
        ors = orca.Series(s)
        self.assertEqual(repr(list(s.__iter__())), repr(list(ors.__iter__())))

        dates = pd.date_range('20180310', periods=6)
        s = pd.Series(dates)
        ors = orca.Series(s)
        # Time type not equal
        # self.assertEqual(repr(list(s.__iter__())), repr(list(ors.__iter__())))

    def test_series_indexing_items(self):
        s = pd.Series(range(5, 10))
        a = s.items()
        ors = orca.Series(s)
        b = ors.items()
        self.assertEqual(repr(list(a)), repr(list(b)))

    def test_series_indexing_where_cond_bool(self):
        s = pd.Series(range(5))
        orcas = orca.Series(s)
        # other is nan
        assert_series_equal(s.where(s > 0), orcas.where(orcas > 0).to_pandas())
        # other is number
        assert_series_equal(s.where(s > 0, 10), orcas.where(orcas > 0, 10).to_pandas())
        # other is series
        assert_series_equal(s.where(s > 0, s), orcas.where(orcas > 0, orcas).to_pandas())

    def test_series_indexing_where(self):
        s = pd.Series(range(5))
        ors = orca.Series(s)
        assert_equal(s.where(s > 2), ors.where(ors > 2).to_pandas())
        assert_equal(s.where(s > 2, -s), ors.where(ors > 2, -ors).to_pandas())
        assert_equal(s.where(s > 2, 7), ors.where(ors > 2, 7).to_pandas())

    def test_series_indexing_where_cond_series(self):
        s = pd.Series(range(5))
        cond = pd.Series([True, False, True, True, True])
        ors = orca.Series(s)
        ors_cond = orca.Series(cond)
        assert_equal(s.where(cond), ors.where(ors_cond).to_pandas())

    def test_series_indexing_loc_get(self):
        ps = pd.Series(list('abcde'), index=[0, 3, 2, 5, 4])
        os = orca.Series(list('abcde'), index=[0, 3, 2, 5, 4])
        assert_series_equal(ps, os.to_pandas())
        assert_series_equal(os.loc[3:5].to_pandas(), ps.loc[3:5])

        ps = pd.Series([1, 2, 3, 4, 5, 6, 7], name='x')
        os = orca.Series(ps)
        self.assertEqual(os.loc[0], ps.loc[0])

    def test_series_indexing_loc_set(self):
        ps = pd.Series(list('abcde'), index=[0, 3, 2, 5, 4])
        os = orca.Series(list('abcde'), index=[0, 3, 2, 5, 4])
        assert_series_equal(os.to_pandas(), ps)
        ps.loc[3:5] = "5"
        os.loc[3:5] = "5"
        assert_series_equal(os.to_pandas(), ps)
        ps = pd.Series(np.repeat(1, 5), index=[0, 3, 2, 5, 4])
        os = orca.Series(np.repeat(1, 5), index=[0, 3, 2, 5, 4])
        ps.loc[3:5] = 2
        os.loc[3:5] = 2
        assert_series_equal(os.to_pandas(), ps)
        ps = pd.Series(list('abcde'), index=["a", "a", "a", "b", "b"])
        os = orca.Series(list('abcde'), index=["a", "a", "a", "b", "b"])
        assert_series_equal(os.to_pandas(), ps)
        ps.loc["a"] = "5"
        os.loc["a"] = "5"
        assert_series_equal(os.to_pandas(), ps)
        # TODO：orca不支持：当index中的取值是重复的
        # ps = pd.Series(np.repeat(1, 5), index=["a", "a", "b", "b", "c"])
        # os = orca.Series(np.repeat(1, 5), index=["a", "a", "b", "b", "c"])
        # ps.loc["a":"b"] = 2
        # os.loc["a":"b"] = 2
        # assert_series_equal(os.to_pandas(), ps)

    def test_series_indexing_iloc_get(self):
        v = [1, 2, 3, 4, 5]
        ps = pd.Series(v, index=list(range(0, 10, 2)))
        os = orca.Series(v, index=list(range(0, 10, 2)))
        assert_series_equal(os.iloc[:3].to_pandas(), ps.iloc[:3])
        # TODO: iloc bug
        # assert_series_equal(os.iloc[[]].to_pandas(), ps.iloc[[]])

    def test_series_indexing_iloc_set(self):
        v1 = [1, 2, 3, 4, 5]
        v2 = ["a", "b", "c", "d", "e"]

        ps = pd.Series(v1, index=list(range(0, 10, 2)))
        os = orca.Series(v1, index=list(range(0, 10, 2)))
        ps.iloc[:3] = 4
        os.iloc[:3] = 4
        assert_series_equal(os.to_pandas(), ps)

        ps = pd.Series(v2, index=list(range(0, 10, 2)))
        os = orca.Series(v2, index=list(range(0, 10, 2)))
        ps.iloc[:3] = "s"
        os.iloc[:3] = "s"
        assert_series_equal(os.to_pandas(), ps)

        ps = pd.Series(v1, index=["a", "a", "b", "b", "c"])
        os = orca.Series(v1, index=["a", "a", "b", "b", "c"])
        ps.iloc[:3] = 4
        os.iloc[:3] = 4
        assert_series_equal(os.to_pandas(), ps)

        ps = pd.Series(v2, index=["a", "a", "b", "b", "c"])
        os = orca.Series(v2, index=["a", "a", "b", "b", "c"])
        ps.iloc[:3] = "s"
        os.iloc[:3] = "s"
        assert_series_equal(os.to_pandas(), ps)

    def test_series_indexing_sort_index(self):
        ps = pd.Series(list('abcde'), index=[0, 3, 2, 5, 4])
        os = orca.Series(list('abcde'), index=[0, 3, 2, 5, 4])
        assert_series_equal(ps.sort_index(), os.sort_index().to_pandas())
        assert_series_equal(ps.sort_index().loc[0:3], os.sort_index().loc[0:3].to_pandas())
        # TODO: orca不支持 当loc访问的slice的下界在index中不存在
        # assert_series_equal(ps.sort_index().loc[1:3], os.sort_index().loc[1:3].to_pandas())


if __name__ == '__main__':
    unittest.main()
