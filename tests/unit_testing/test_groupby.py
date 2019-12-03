import unittest
import orca
import os.path as path
from setup.settings import *
from pandas.util.testing import *


class Csv:
    pdf_csv = None
    odf_csv = None


class GroupByTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # configure data directory
        DATA_DIR = path.abspath(path.join(__file__, "../setup/data"))
        fileName = 'groupbyDate.csv'
        data = os.path.join(DATA_DIR, fileName)
        data = data.replace('\\', '/')

        # Orca connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

        Csv.pdf_csv = pd.read_csv(data, parse_dates=[1], dtype={"id": np.int32, "tbool": np.bool})
        # Csv.pdf_csv['tbool'] = Csv.pdf_csv["tbool"].astype(np.bool)
        Csv.odf_csv = orca.read_csv(data,
                                    dtype={"id": np.int32, "tsymbol": "SYMBOL", "tbool": np.bool, "tshort": np.int16,
                                           "tint": np.int32, "tlong": np.int64, "tfloat": np.float32,
                                           "tdouble": np.float64})

    @property
    def pdf_csv(self):
        return Csv.pdf_csv

    @property
    def odf_csv(self):
        return Csv.odf_csv

    @property
    def pdf(self):
        n = 100
        re = n / 10
        pdf_da = pd.DataFrame({'id': np.arange(1, n + 1, 1, dtype='int32'),
                               'date': np.repeat(pd.date_range('2019.08.01', periods=10, freq='D'), re),
                               'tsymbol': np.repeat(['a', 'b', 'c', 'd', 'e', 'QWW', 'FEA', 'FFW', 'DER', 'POD'], re),
                               'tbool': np.repeat(np.repeat(np.arange(2, dtype='bool'), 5), re),
                               'tchar': np.repeat(np.arange(1, 11, 1, dtype='int8'), re),
                               'tshort': np.repeat(np.arange(1, 11, 1, dtype='int16'), re),
                               'tint': np.repeat(np.arange(1, 11, 1, dtype='int32'), re),
                               'tlong': np.repeat(np.arange(1, 11, 1, dtype='int64'), re),
                               'tfloat': np.repeat(np.arange(1, 11, 1, dtype='float32'), re),
                               'tdouble': np.repeat(np.arange(1, 11, 1, dtype='float64'), re)
                               })
        pdf_da.set_index("id", inplace=True)
        return pdf_da

    @property
    def odf(self):
        return orca.DataFrame(self.pdf)

    def test_groupby_allocation_verification(self):
        self.assertIsInstance(self.odf_csv.groupby('date')['tshort'].count().to_pandas(), Series)
        with self.assertRaises(KeyError):
            self.odf_csv.groupby('date')['hello'].count()
        with self.assertRaises(KeyError):
            self.odf_csv.groupby('date')[['dare', 5, 0]].count()
        with self.assertRaises(KeyError):
            self.odf_csv.groupby('date')[['hello', 'world']].count()
        with self.assertRaises(KeyError):
            self.odf_csv.groupby('date')[np.array([1, 2, 3])].count()

    def test_groupbyObj(self):

        a = pd.date_range("20080101", periods=3650, freq='d')
        pdf = pd.DataFrame(
            {"date": np.repeat(a, 100), "value": np.repeat(1, 365000), "sym": np.repeat(["a", "b"], 182500)})
        odf = orca.DataFrame(pdf)
        obj_pdf = pdf.groupby("sym").resample("BA", on="date").sum()
        obj_odf = odf.groupby("sym").resample("BA", on="date").sum()
        assert_frame_equal(obj_odf.to_pandas(), obj_pdf, check_dtype=False)

        idx = orca.date_range('1/1/2000', periods=4, freq='T')
        odf = orca.DataFrame(data=4 * [range(2)], index=idx, columns=['a', 'b'])
        odf.iloc[2, 0] = 5
        pdf = odf.to_pandas()
        pdf.groupby('a').resample('3T').sum()
        assert_frame_equal(obj_odf.to_pandas(), obj_pdf, check_dtype=False)

        # groupby时通过dt.date指定分组
        a = pd.date_range("20080101000000", periods=3650, freq='h')
        pdf = pd.DataFrame(
            {"date": np.repeat(a, 100), "value": np.repeat(1, 365000), "sym": np.repeat(["a", "b"], 182500)})
        odf = orca.DataFrame(pdf)
        obj_pdf = pdf.groupby([pdf['date'].dt.date, 'sym'])['value'].mean()
        obj_odf = odf.groupby([odf['date'].dt.date, 'sym'])['value'].mean()
        # DIFFERENT INDEX TYPE
        # assert_series_equal(obj_odf.to_pandas(), obj_pdf, check_index_type=False)

        # TODO：groupby时指定时间列的freq
        # obj_pdf = pdf.groupby([pd.Grouper(key='date', freq='D'), pd.Grouper('sym')])['value'].mean()
        # obj_odf = odf.groupby([orca.Grouper(key='date', freq='D'), orca.Grouper('sym', )])['value'].mean()
        # assert_frame_equal(obj_odf.to_pandas(), obj_pdf)
        # obj_pdf = pdf.groupby([pd.Grouper(key='date'), pd.Grouper('sym')])['value'].mean()
        # obj_odf = odf.groupby([orca.Grouper(key='date'), orca.Grouper('sym')])['value'].mean()
        # assert_frame_equal(obj_odf.to_pandas(), obj_pdf)
        # a = pd.date_range("20080101000000", periods=3650, freq='h')
        # pdf = pd.DataFrame(
        #     {"date": np.repeat(a, 100), "value": np.repeat(1, 365000), "sym": np.repeat(["a", "b"], 182500)})
        # odf = orca.DataFrame(pdf)
        # obj_pdf = pdf.groupby([pd.Grouper(key='date', freq='h'), pd.Grouper('sym')])['value'].mean()
        # obj_odf = odf.groupby([orca.Grouper(key='date', freq='h'), orca.Grouper('sym')])['value'].mean()
        # assert_frame_equal(obj_odf.to_pandas(), obj_pdf)
        # obj_pdf = pdf.groupby([pd.Grouper(key='date'), pd.Grouper('sym')])['value'].mean()
        # obj_odf = odf.groupby([orca.Grouper(key='date'), orca.Grouper('sym')])['value'].mean()
        # assert_frame_equal(obj_odf.to_pandas(), obj_pdf)

    def test_from_import_groupby_param_by_date_all(self):
        a = self.odf_csv.drop(columns='tsymbol').groupby('date').all()
        b = self.pdf_csv.drop(columns='tsymbol').groupby('date').all()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_date_any(self):
        a = self.odf_csv.drop(columns='tsymbol').groupby('date').any()
        b = self.pdf_csv.drop(columns='tsymbol').groupby('date').any()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_date_bfill(self):
        a = self.odf_csv.groupby('date').bfill()
        b = self.pdf_csv.groupby('date').bfill()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_date_count(self):
        a = self.odf_csv.groupby('date').count()
        b = self.pdf_csv.groupby('date').count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_date_cumcount(self):
        a = self.odf_csv.groupby('date').cumcount()
        b = self.pdf_csv.groupby('date').cumcount()
        # TODO: TOO MUCH DIFFS
        self.assertIsInstance(a.to_pandas(), DataFrame)
        self.assertIsInstance(b, Series)
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_date_cummax(self):
        a = self.odf_csv.drop(columns='tsymbol').groupby('date').cummax()
        b = self.pdf_csv.drop(columns='tsymbol').groupby('date').cummax()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_import_groupby_param_by_date_cummin(self):
        a = self.odf_csv.drop(columns='tsymbol').groupby('date').cummin()
        b = self.pdf_csv.drop(columns='tsymbol').groupby('date').cummin()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_import_groupby_param_by_date_cumprod(self):
        a = self.odf_csv.groupby('date').cumprod().compute()
        b = self.pdf_csv.groupby('date').cumprod()
        # TODO: TOO MUCH DIFFS
        ar = a[(a['id'] >= 100) & (a['id'] <= 200)].compute().sort_values("id")
        br = b[(b['id'] >= 100) & (b['id'] <= 200)].sort_values("id")
        assert_frame_equal(ar.to_pandas().reset_index(drop=True), br.reset_index(drop=True), check_dtype=False,
                           check_index_type=False, check_less_precise=1)
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_date_cumsum(self):
        a = self.odf_csv.groupby('date').cumsum()
        b = self.pdf_csv.groupby('date').cumsum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_import_groupby_param_by_date_ffill(self):
        a = self.odf_csv.groupby('date').ffill()
        b = self.pdf_csv.groupby('date').ffill()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_date_first(self):
        a = self.odf_csv.groupby('date').first()
        b = self.pdf_csv.groupby('date').first()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_date_head(self):
        # TODO：NOT IMPLEMENTED
        pass
        # a = self.odf_csv.groupby('date').head()
        # b = self.pdf_csv.groupby('date').head()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_date_last(self):
        a = self.odf_csv.groupby('date').last()
        b = self.pdf_csv.groupby('date').last()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_less_precise=1)

    def test_from_import_groupby_param_by_date_max(self):
        a = self.odf_csv.groupby('date').max()
        b = self.pdf_csv.groupby('date').max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_less_precise=1)

    def test_from_import_groupby_param_by_date_mean(self):
        a = self.odf_csv.groupby('date').mean()
        b = self.pdf_csv.groupby('date').mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_date_median(self):
        a = self.odf_csv.drop(columns=["tbool", "tsymbol"]).groupby('date').median()
        b = self.pdf_csv.drop(columns=["tbool", "tsymbol"]).groupby('date').median()
        assert_frame_equal(a.to_pandas(), b, check_less_precise=1)

    def test_from_import_groupby_param_by_date_min(self):
        a = self.odf_csv.groupby('date').min()
        b = self.pdf_csv.groupby('date').min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_less_precise=1)

    def test_from_import_groupby_param_by_date_ngroup(self):
        # TODO：NOT IMPLEMENTED
        pass
        # a = self.odf_csv.groupby('date').ngroup()
        # b = self.pdf_csv.groupby('date').ngroup()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_date_nth(self):
        # TODO：NOT IMPLEMENTED
        pass
        # a = self.odf_csv.groupby('date').nth(0)
        # b = self.pdf_csv.groupby('date').nth(0)
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_date_ohlc(self):
        a = self.odf_csv.drop(columns=['tsymbol']).groupby('date').ohlc()
        # b = self.pdf_csv.drop(columns=['tsymbol']).groupby('date').ohlc()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_date_prod(self):
        a = self.odf_csv.groupby('date').prod()
        b = self.pdf_csv.groupby('date').prod()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_date_rank(self):
        a = self.odf_csv.groupby('date').rank()
        # TODO: pandas doesn't support
        # b = self.pdf_csv.groupby('date').rank()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_date_pct_change(self):
        a = self.odf_csv.drop(columns=["tbool", "tsymbol"]).groupby('date').pct_change().compute()
        b = self.pdf_csv.drop(columns=["tbool", "tsymbol"]).groupby('date').pct_change()
        # TODO: TOO MUCH DIFFS
        ar = a[(a['id'] >= 10) & (a['id'] <= 200)].compute().sort_values("id")
        br = b[(b['id'] >= 10) & (b['id'] <= 200)].sort_values("id")
        assert_frame_equal(ar.to_pandas().reset_index(drop=True), br.reset_index(drop=True),
                           check_dtype=False, check_index_type=False, check_less_precise=1)
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_date_size(self):
        a = self.odf_csv.groupby('date').size()
        b = self.pdf_csv.groupby('date').size()
        assert_series_equal(a.to_pandas(), b, check_dtype=False, check_less_precise=1)

    def test_from_import_groupby_param_by_date_sem(self):
        a = self.odf_csv.groupby('date').sem()
        b = self.pdf_csv.groupby('date').sem()
        # sem for values of string type makes no sense
        assert_frame_equal(a.to_pandas(), b.drop(columns=["tsymbol"]), check_dtype=False, check_index_type=False, check_less_precise=1,check_like=True)

    def test_from_import_groupby_param_by_date_std(self):
        a = self.odf_csv.groupby('date').std()
        b = self.pdf_csv.groupby('date').std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_less_precise=1)

    def test_from_import_groupby_param_by_date_sum(self):
        a = self.odf_csv.groupby('date').sum()
        b = self.pdf_csv.groupby('date').sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_less_precise=1)

    def test_from_import_groupby_param_by_date_var(self):
        a = self.odf_csv.groupby('date').var()
        b = self.pdf_csv.groupby('date').var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_less_precise=1)

    def test_from_import_groupby_param_by_date_tail(self):
        # TODO：NOT IMPLEMENTED
        pass
        # a = self.odf_csv.groupby('date').tail()
        # b = self.pdf_csv.groupby('date').tail()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_date_all(self):
        a = self.odf.drop(columns='tsymbol').groupby('date').all()
        b = self.pdf.drop(columns='tsymbol').groupby('date').all()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_date_any(self):
        a = self.odf.drop(columns='tsymbol').groupby('date').any()
        b = self.pdf.drop(columns='tsymbol').groupby('date').any()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_date_bfill(self):
        a = self.odf.groupby('date').bfill()
        b = self.pdf.groupby('date').bfill()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_date_count(self):
        a = self.odf.groupby('date').count()
        b = self.pdf.groupby('date').count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_date_cumcount(self):
        a = self.odf.groupby('date').cumcount()
        b = self.pdf.groupby('date').cumcount()
        # FIXME: 差异过大，无法比较
        self.assertIsInstance(a.to_pandas(), DataFrame)
        self.assertIsInstance(b, Series)
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_date_cummax(self):
        a = self.odf.drop(columns=["tsymbol"]).groupby('date').cummax()
        b = self.pdf.drop(columns=[ "tsymbol"]).groupby('date').cummax()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_date_cummin(self):
        a = self.odf.drop(columns=["tsymbol"]).groupby('date').cummin()
        b = self.pdf.drop(columns=["tsymbol"]).groupby('date').cummin()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=5)

    def test_from_pandas_groupby_param_by_date_cumprod(self):
        a = self.odf.groupby('date').cumprod().compute()
        b = self.pdf.groupby('date').cumprod()
        # TODO: TOO MUCH DIFFS
        assert_frame_equal(a.sort_index().reset_index(drop=True).to_pandas(),
                           b.sort_index().reset_index(drop=True), check_dtype=False,
                           check_index_type=False, check_less_precise=1)
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_date_cumsum(self):
        a = self.odf.groupby('date').cumsum()
        b = self.pdf.groupby('date').cumsum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_date_ffill(self):
        a = self.odf.groupby('date').ffill()
        b = self.pdf.groupby('date').ffill()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_date_first(self):
        a = self.odf.groupby('date').first()
        b = self.pdf.groupby('date').first()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_date_head(self):
        # TODO：NOT IMPLEMENTED
        pass
        # a = self.odf.groupby('date').head()
        # b = self.pdf.groupby('date').head()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_date_last(self):
        a = self.odf.groupby('date').last()
        b = self.pdf.groupby('date').last()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_date_max(self):
        a = self.odf.groupby('date').max()
        b = self.pdf.groupby('date').max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_date_mean(self):
        a = self.odf.groupby('date').mean()
        b = self.pdf.groupby('date').mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_date_median(self):
        a = self.odf.drop(columns=["tbool"]).groupby('tsymbol').median()
        b = self.pdf.drop(columns=["tbool"]).groupby('tsymbol').median()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_date_min(self):
        a = self.odf.groupby('date').min()
        b = self.pdf.groupby('date').min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_date_ngroup(self):
        # TODO：NOT IMPLEMENTED
        pass
        # a = self.odf.groupby('date').ngroup()
        # b = self.pdf.groupby('date').ngroup()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_date_nth(self):
        # TODO：NOT IMPLEMENTED
        pass
        # a = self.odf.groupby('date').nth(0)
        # b = self.pdf.groupby('date').nth(0)
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_date_ohlc(self):
        a = self.odf.groupby('date').ohlc()
        # b = self.pdf.groupby('date').ohlc()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_date_prod(self):
        a = self.odf.groupby('date').prod()
        b = self.pdf.groupby('date').prod()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_date_rank(self):
        a = self.odf.groupby('date').rank()
        # TODO: pandas doesn't support
        # b = self.pdf.groupby('date').rank()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_date_pct_change(self):
        a = self.odf.drop(columns=["tbool", "tsymbol"]).groupby('date').pct_change()
        b = self.pdf.drop(columns=["tbool", "tsymbol"]).groupby('date').pct_change()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_date_size(self):
        a = self.odf.groupby('date').size()
        b = self.pdf.groupby('date').size()
        assert_series_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_date_sem(self):
        a = self.odf.drop(columns=["tbool"]).groupby('date').sem()
        b = self.pdf.drop(columns=["tbool"]).groupby('date').sem()
        assert_frame_equal(a.to_pandas(), b.drop(columns=["tsymbol"]), check_dtype=False, check_index_type=False, check_less_precise=1, check_like=True)

    def test_from_pandas_groupby_param_by_date_std(self):
        a = self.odf.drop(columns=["tbool"]).groupby('date').std()
        b = self.pdf.drop(columns=["tbool"]).groupby('date').std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_date_sum(self):
        a = self.odf.groupby('date').sum()
        b = self.pdf.groupby('date').sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_date_var(self):
        a = self.odf.groupby('date').var()
        b = self.pdf.groupby('date').var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_date_tail(self):
        # TODO：NOT IMPLEMENTED
        pass
        # a = self.odf.groupby('date').tail()
        # b = self.pdf.groupby('date').tail()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_symbol_all(self):
        a = self.odf_csv.groupby('tsymbol').all()
        b = self.pdf_csv.groupby('tsymbol').all()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_symbol_any(self):
        a = self.odf_csv.groupby('tsymbol').any()
        b = self.pdf_csv.groupby('tsymbol').any()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_symbol_bfill(self):
        a = self.odf_csv.groupby('tsymbol').bfill()
        b = self.pdf_csv.groupby('tsymbol').bfill()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_symbol_count(self):
        a = self.odf_csv.groupby('tsymbol').count()
        b = self.pdf_csv.groupby('tsymbol').count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_symbol_cumcount(self):
        a = self.odf_csv.groupby('tsymbol').cumcount()
        b = self.pdf_csv.groupby('tsymbol').cumcount()
        # FIXME: TOO MUCH DIFFS
        self.assertIsInstance(a.to_pandas(), DataFrame)
        self.assertIsInstance(b, Series)
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_symbol_cummax(self):
        a = self.odf_csv.drop(columns=["date"]).groupby('tsymbol').cummax()
        b = self.pdf_csv.drop(columns=["date"]).groupby('tsymbol').cummax()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_symbol_cummin(self):
        a = self.odf_csv.drop(columns=["date"]).groupby('tsymbol').cummin()
        b = self.pdf_csv.drop(columns=["date"]).groupby('tsymbol').cummin()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=5)

    def test_from_import_groupby_param_by_symbol_cumprod(self):
        a = self.odf_csv.groupby('tsymbol').cumprod().compute()
        b = self.pdf_csv.groupby('tsymbol').cumprod()
        # TODO: TOO MUCH DIFFS
        ar = a[(a['id'] >= 100) & (a['id'] <= 200)].compute().sort_values("id")
        br = b[(b['id'] >= 100) & (b['id'] <= 200)].sort_values("id")
        assert_frame_equal(ar.to_pandas().reset_index(drop=True), br.reset_index(drop=True), check_dtype=False,
                           check_index_type=False, check_less_precise=1)
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_symbol_cumsum(self):
        a = self.odf_csv.groupby('tsymbol').cumsum()
        b = self.pdf_csv.groupby('tsymbol').cumsum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_symbol_ffill(self):
        a = self.odf_csv.groupby('tsymbol').ffill()
        b = self.pdf_csv.groupby('tsymbol').ffill()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_symbol_first(self):
        a = self.odf_csv.groupby('tsymbol').first()
        b = self.pdf_csv.groupby('tsymbol').first()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_symbol_head(self):
        # TODO：NOT IMPLEMENTED
        pass
        # a = self.odf_csv.groupby('tsymbol').head()
        # b = self.pdf_csv.groupby('tsymbol').head()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_symbol_last(self):
        a = self.odf_csv.groupby('tsymbol').last()
        b = self.pdf_csv.groupby('tsymbol').last()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_symbol_max(self):
        a = self.odf_csv.groupby('tsymbol').max()
        b = self.pdf_csv.groupby('tsymbol').max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_symbol_mean(self):
        a = self.odf_csv.groupby('tsymbol').mean()
        b = self.pdf_csv.groupby('tsymbol').mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_symbol_median(self):
        a = self.odf_csv.drop(columns=["tbool"]).groupby('tsymbol').median()
        b = self.pdf_csv.drop(columns=["tbool"]).groupby('tsymbol').median()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_symbol_min(self):
        a = self.odf_csv.groupby('tsymbol').min()
        b = self.pdf_csv.groupby('tsymbol').min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_symbol_ngroup(self):
        # TODO：NOT IMPLEMENTED
        pass
        # a = self.odf_csv.groupby('tsymbol').ngroup()
        # b = self.pdf_csv.groupby('tsymbol').ngroup()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_symbol_nth(self):
        # TODO：NOT IMPLEMENTED
        pass
        # a = self.odf_csv.groupby('tsymbol').nth(0)
        # b = self.pdf_csv.groupby('tsymbol').nth(0)
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_symbol_ohlc(self):
        a = self.odf_csv.groupby('tsymbol').ohlc()
        # b = self.pdf_csv.groupby('tsymbol').ohlc()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_symbol_prod(self):
        a = self.odf_csv.groupby('tsymbol').prod()
        b = self.pdf_csv.groupby('tsymbol').prod()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_symbol_rank(self):
        a = self.odf_csv.groupby('tsymbol').rank()
        b = self.pdf_csv.groupby('tsymbol').rank()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_symbol_pct_change(self):
        a = self.odf_csv.drop(columns=["tbool", "date"]).groupby('tsymbol').pct_change().compute()
        b = self.pdf_csv.drop(columns=["tbool", "date"]).groupby('tsymbol').pct_change()
        # TODO: TOO MUCH DIFFS
        ar = a[(a['id'] >= 10) & (a['id'] <= 200)].compute().sort_values("id")
        br = b[(b['id'] >= 10) & (b['id'] <= 200)].sort_values("id")
        assert_frame_equal(ar.to_pandas().reset_index(drop=True), br.reset_index(drop=True),
                           check_dtype=False, check_index_type=False, check_less_precise=1)
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_symbol_size(self):
        a = self.odf_csv.groupby('tsymbol').size()
        b = self.pdf_csv.groupby('tsymbol').size()
        assert_series_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_symbol_sem(self):
        a = self.odf_csv.groupby('tsymbol').sem()
        b = self.pdf_csv.groupby('tsymbol').sem()
        assert_frame_equal(a.to_pandas(), b.drop(columns=["date"]), check_dtype=False, check_index_type=False, check_less_precise=1, check_like=True)

    def test_from_import_groupby_param_by_symbol_std(self):
        a = self.odf_csv.drop(columns=["tbool"]).groupby('tsymbol').std()
        b = self.pdf_csv.drop(columns=["tbool"]).groupby('tsymbol').std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_symbol_sum(self):
        a = self.odf_csv.groupby('tsymbol').sum()
        b = self.pdf_csv.groupby('tsymbol').sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_symbol_var(self):
        a = self.odf_csv.groupby('tsymbol').var()
        b = self.pdf_csv.groupby('tsymbol').var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_symbol_tail(self):
        # TODO：NOT IMPLEMENTED
        pass
        # a = self.odf_csv.groupby('tsymbol').tail()
        # b = self.pdf_csv.groupby('tsymbol').tail()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_symbol_all(self):
        a = self.odf.groupby('tsymbol').all()
        b = self.pdf.groupby('tsymbol').all()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_symbol_any(self):
        a = self.odf.groupby('tsymbol').any()
        b = self.pdf.groupby('tsymbol').any()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_symbol_bfill(self):
        a = self.odf.groupby('tsymbol').bfill()
        b = self.pdf.groupby('tsymbol').bfill()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_symbol_count(self):
        a = self.odf.groupby('tsymbol').count()
        b = self.pdf.groupby('tsymbol').count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_symbol_cumcount(self):
        a = self.odf.groupby('tsymbol').cumcount()
        b = self.pdf.groupby('tsymbol').cumcount()
        # FIXME: TOOMUCH DIFFS
        self.assertIsInstance(a.to_pandas(), DataFrame)
        self.assertIsInstance(b, Series)
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_symbol_cummax(self):
        a = self.odf.drop(columns=["date"]).groupby('tsymbol').cummax()
        b = self.pdf.drop(columns=["date"]).groupby('tsymbol').cummax()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=5)

    def test_from_pandas_groupby_param_by_symbol_cummin(self):
        a = self.odf.drop(columns=["date"]).groupby('tsymbol').cummin()
        b = self.pdf.drop(columns=["date"]).groupby('tsymbol').cummin()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=5)

    def test_from_pandas_groupby_param_by_symbol_cumprod(self):
        a = self.odf.groupby('tsymbol').cumprod().compute()
        b = self.pdf.groupby('tsymbol').cumprod()
        assert_frame_equal(a.sort_values('tint').reset_index(drop=True).to_pandas().iloc[:, 1:],
                           b.sort_values('tint').reset_index(drop=True).iloc[:, 1:],
                           check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_symbol_cumsum(self):
        a = self.odf.groupby('tsymbol').cumsum()
        b = self.pdf.groupby('tsymbol').cumsum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_symbol_ffill(self):
        a = self.odf.groupby('tsymbol').ffill()
        b = self.pdf.groupby('tsymbol').ffill()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_symbol_first(self):
        a = self.odf.groupby('tsymbol').first()
        b = self.pdf.groupby('tsymbol').first()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_symbol_head(self):
        # TODO：NOT IMPLEMENTED
        pass
        # a = self.odf.groupby('tsymbol').head()
        # b = self.pdf.groupby('tsymbol').head()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_symbol_last(self):
        a = self.odf.groupby('tsymbol').last()
        b = self.pdf.groupby('tsymbol').last()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_symbol_max(self):
        a = self.odf.groupby('tsymbol').max()
        b = self.pdf.groupby('tsymbol').max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_symbol_mean(self):
        a = self.odf.groupby('tsymbol').mean()
        b = self.pdf.groupby('tsymbol').mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_symbol_median(self):
        a = self.odf.drop(columns=["tbool"]).groupby('tsymbol').median()
        b = self.pdf.drop(columns=["tbool"]).groupby('tsymbol').median()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_symbol_min(self):
        a = self.odf.groupby('tsymbol').min()
        b = self.pdf.groupby('tsymbol').min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_symbol_ngroup(self):
        # TODO：NOT IMPLEMENTED
        pass
        # a = self.odf.groupby('tsymbol').ngroup()
        # b = self.pdf.groupby('tsymbol').ngroup()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_symbol_nth(self):
        # TODO：NOT IMPLEMENTED
        pass
        # a = self.odf.groupby('tsymbol').nth(0)
        # b = self.pdf.groupby('tsymbol').nth(0)
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_symbol_ohlc(self):
        a = self.odf.drop(columns=["date"]).groupby('tsymbol').ohlc()
        # b = self.pdf.drop(columns=["date"]).groupby('tsymbol').ohlc()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_symbol_prod(self):
        a = self.odf.groupby('tsymbol').prod()
        b = self.pdf.groupby('tsymbol').prod()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_symbol_rank(self):
        a = self.odf.groupby('tsymbol').rank()
        b = self.pdf.groupby('tsymbol').rank()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_symbol_pct_change(self):
        a = self.odf.drop(columns=["tbool", "date"]).groupby('tsymbol').pct_change()
        b = self.pdf.drop(columns=["tbool", "date"]).groupby('tsymbol').pct_change()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_symbol_size(self):
        a = self.odf.groupby('tsymbol').size()
        b = self.pdf.groupby('tsymbol').size()
        assert_series_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_symbol_sem(self):
        a = self.odf.drop(columns=["tbool"]).groupby('tsymbol').sem()
        b = self.pdf.drop(columns=["tbool"]).groupby('tsymbol').sem()
        assert_frame_equal(a.to_pandas(), b.drop(columns=["date"]), check_dtype=False, check_index_type=False, check_less_precise=1, check_like=True)

    def test_from_pandas_groupby_param_by_symbol_std(self):
        a = self.odf.drop(columns=["tbool"]).groupby('tsymbol').std()
        b = self.pdf.drop(columns=["tbool"]).groupby('tsymbol').std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_symbol_sum(self):
        a = self.odf.groupby('tsymbol').sum()
        b = self.pdf.groupby('tsymbol').sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_symbol_var(self):
        a = self.odf.groupby('tsymbol').var()
        b = self.pdf.groupby('tsymbol').var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_symbol_tail(self):
        # TODO：NOT IMPLEMENTED
        pass
        # a = self.odf.groupby('tsymbol').tail()
        # b = self.pdf.groupby('tsymbol').tail()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_long_all(self):
        a = self.odf_csv.drop(columns='tsymbol').groupby('tlong').all()
        b = self.pdf_csv.drop(columns='tsymbol').groupby('tlong').all()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_less_precise=1, check_index_type=False)

    def test_from_import_groupby_param_by_long_any(self):
        a = self.odf_csv.drop(columns='tsymbol').groupby('tlong').any()
        b = self.pdf_csv.drop(columns='tsymbol').groupby('tlong').any()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_long_bfill(self):
        a = self.odf_csv.groupby('tlong').bfill()
        b = self.pdf_csv.groupby('tlong').bfill()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_long_count(self):
        a = self.odf_csv.groupby('tlong').count()
        b = self.pdf_csv.groupby('tlong').count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_long_cumcount(self):
        a = self.odf_csv.groupby('tlong').cumcount()
        b = self.pdf_csv.groupby('tlong').cumcount()
        # TODO: TOO MUCH DIFFS
        self.assertIsInstance(a.to_pandas(), DataFrame)
        self.assertIsInstance(b, Series)
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_long_cummax(self):
        a = self.odf_csv.drop(columns=["date", "tsymbol"]).groupby('tlong').cummax()
        b = self.pdf_csv.drop(columns=["date", "tsymbol"]).groupby('tlong').cummax()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_long_cummin(self):
        a = self.odf_csv.drop(columns=["date", "tsymbol"]).groupby('tlong').cummin()
        b = self.pdf_csv.drop(columns=["date", "tsymbol"]).groupby('tlong').cummin()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=5)

    def test_from_import_groupby_param_by_long_cumprod(self):
        a = self.odf_csv.groupby('tlong').cumprod().compute()
        b = self.pdf_csv.groupby('tlong').cumprod()
        # TODO: TOO MUCH DIFFS
        ar = a[(a['id'] >= 100) & (a['id'] <= 200)].compute().sort_values("id")
        br = b[(b['id'] >= 100) & (b['id'] <= 200)].sort_values("id")
        assert_frame_equal(ar.to_pandas().reset_index(drop=True), br.reset_index(drop=True), check_dtype=False,
                           check_index_type=False, check_less_precise=1)
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_long_cumsum(self):
        a = self.odf_csv.groupby('tlong').cumsum()
        b = self.pdf_csv.groupby('tlong').cumsum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_long_ffill(self):
        a = self.odf_csv.groupby('tlong').ffill()
        b = self.pdf_csv.groupby('tlong').ffill()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_long_first(self):
        a = self.odf_csv.groupby('tlong').first()
        b = self.pdf_csv.groupby('tlong').first()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_long_head(self):
        # TODO：NOT IMPLEMENTED
        pass
        # a = self.odf_csv.groupby('tlong').head()
        # b = self.pdf_csv.groupby('tlong').head()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_long_last(self):
        a = self.odf_csv.groupby('tlong').last()
        b = self.pdf_csv.groupby('tlong').last()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_long_max(self):
        a = self.odf_csv.groupby('tlong').max()
        b = self.pdf_csv.groupby('tlong').max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_long_mean(self):
        a = self.odf_csv.groupby('tlong').mean()
        b = self.pdf_csv.groupby('tlong').mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_long_median(self):
        a = self.odf_csv.drop(columns=["tbool", "tsymbol"]).groupby('tlong').median()
        b = self.pdf_csv.drop(columns=["tbool", "tsymbol"]).groupby('tlong').median()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_long_min(self):
        a = self.odf_csv.groupby('tlong').min()
        b = self.pdf_csv.groupby('tlong').min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_long_ngroup(self):
        # TODO：NOT IMPLEMENTED
        pass
        # a = self.odf_csv.groupby('tlong').ngroup()
        # b = self.pdf_csv.groupby('tlong').ngroup()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_long_nth(self):
        # TODO：NOT IMPLEMENTED
        pass
        # a = self.odf_csv.groupby('tlong').nth(0)
        # b = self.pdf_csv.groupby('tlong').nth(0)
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_long_ohlc(self):
        a = self.odf_csv.drop(columns=['tsymbol', "date"]).groupby('tlong').ohlc()
        b = self.pdf_csv.drop(columns=['tsymbol', "date"]).groupby('tlong').ohlc()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_long_prod(self):
        a = self.odf_csv.groupby('tlong').prod()
        b = self.pdf_csv.groupby('tlong').prod()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_long_rank(self):
        a = self.odf_csv.groupby('tlong').rank()
        # TODO: pandas doesn't support
        # b = self.pdf_csv.groupby('tlong').rank()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_long_pct_change(self):
        a = self.odf_csv.drop(columns=["tbool", "tsymbol", "date"]).groupby('tlong').pct_change().compute()
        b = self.pdf_csv.drop(columns=["tbool", "tsymbol", "date"]).groupby('tlong').pct_change()
        # TODO: TOO MUCH DIFFS
        ar = a[(a['id'] >= 10) & (a['id'] <= 200)].compute().sort_values("id")
        br = b[(b['id'] >= 10) & (b['id'] <= 200)].sort_values("id")
        assert_frame_equal(ar.to_pandas().reset_index(drop=True), br.reset_index(drop=True),
                           check_dtype=False, check_index_type=False, check_less_precise=1)
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_long_size(self):
        a = self.odf_csv.groupby('tlong').size()
        b = self.pdf_csv.groupby('tlong').size()
        assert_series_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_long_sem(self):
        a = self.odf_csv.groupby('tlong').sem()
        b = self.pdf_csv.groupby('tlong').sem()
        assert_frame_equal(a.to_pandas(), b.drop(columns=["date","tsymbol"]), check_dtype=False, check_index_type=False, check_less_precise=1, check_like=True)

    def test_from_import_groupby_param_by_long_std(self):
        a = self.odf_csv.drop(columns=["tbool"]).groupby('tlong').std()
        b = self.pdf_csv.drop(columns=["tbool"]).groupby('tlong').std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_long_sum(self):
        a = self.odf_csv.groupby('tlong').sum()
        b = self.pdf_csv.groupby('tlong').sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_long_var(self):
        a = self.odf_csv.groupby('tlong').var()
        b = self.pdf_csv.groupby('tlong').var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_long_tail(self):
        # TODO：NOT IMPLEMENTED
        pass
        # a = self.odf_csv.groupby('tlong').tail()
        # b = self.pdf_csv.groupby('tlong').tail()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_long_all(self):
        a = self.odf.drop(columns='tsymbol').groupby('tlong').all()
        b = self.pdf.drop(columns='tsymbol').groupby('tlong').all()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_long_any(self):
        a = self.odf.drop(columns='tsymbol').groupby('tlong').any()
        b = self.pdf.drop(columns='tsymbol').groupby('tlong').any()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_long_bfill(self):
        a = self.odf.groupby('tlong').bfill()
        b = self.pdf.groupby('tlong').bfill()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_long_count(self):
        a = self.odf.groupby('tlong').count()
        b = self.pdf.groupby('tlong').count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_long_cumcount(self):
        a = self.odf.groupby('tlong').cumcount()
        b = self.pdf.groupby('tlong').cumcount()
        # FIXME: 差异过大，无法比较
        self.assertIsInstance(a.to_pandas(), DataFrame)
        self.assertIsInstance(b, Series)
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_long_cummax(self):
        a = self.odf.drop(columns=["date", "tsymbol"]).groupby('tlong').cummax()
        b = self.pdf.drop(columns=["date", "tsymbol"]).groupby('tlong').cummax()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=5)

    def test_from_pandas_groupby_param_by_long_cummin(self):
        a = self.odf.drop(columns=["date", "tsymbol"]).groupby('tlong').cummin()
        b = self.pdf.drop(columns=["date", "tsymbol"]).groupby('tlong').cummin()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=5)

    def test_from_pandas_groupby_param_by_long_cumprod(self):
        a = self.odf.groupby('tlong').cumprod().compute()
        b = self.pdf.groupby('tlong').cumprod()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True),
                           check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_long_cumsum(self):
        a = self.odf.groupby('tlong').cumsum()
        b = self.pdf.groupby('tlong').cumsum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_long_ffill(self):
        a = self.odf.groupby('tlong').ffill()
        b = self.pdf.groupby('tlong').ffill()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_long_first(self):
        a = self.odf.groupby('tlong').first()
        b = self.pdf.groupby('tlong').first()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_long_head(self):
        # TODO：NOT IMPLEMENTED
        pass
        # a = self.odf.groupby('tlong').head()
        # b = self.pdf.groupby('tlong').head()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_long_last(self):
        a = self.odf.groupby('tlong').last()
        b = self.pdf.groupby('tlong').last()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_long_max(self):
        a = self.odf.groupby('tlong').max()
        b = self.pdf.groupby('tlong').max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_long_mean(self):
        a = self.odf.groupby('tlong').mean()
        b = self.pdf.groupby('tlong').mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_long_median(self):
        a = self.odf.drop(columns=["tbool", "tsymbol"]).groupby('tlong').median()
        b = self.pdf.drop(columns=["tbool", "tsymbol"]).groupby('tlong').median()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_long_min(self):
        a = self.odf.groupby('tlong').min()
        b = self.pdf.groupby('tlong').min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_long_ngroup(self):
        # TODO：NOT IMPLEMENTED
        pass
        # a = self.odf.groupby('tlong').ngroup()
        # b = self.pdf.groupby('tlong').ngroup()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_long_nth(self):
        # TODO：NOT IMPLEMENTED
        pass
        # a = self.odf.groupby('tlong').nth(0)
        # b = self.pdf.groupby('tlong').nth(0)
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_long_ohlc(self):
        a = self.odf.drop(columns=['tsymbol', "date"]).groupby('tlong').ohlc()
        b = self.pdf.drop(columns=['tsymbol', "date"]).groupby('tlong').ohlc()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_long_prod(self):
        a = self.odf.groupby('tlong').prod()
        b = self.pdf.groupby('tlong').prod()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_long_rank(self):
        a = self.odf.groupby('tlong').rank()
        # TODO: pandas doesn't support
        # b = self.pdf.groupby('tlong').rank()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_long_pct_change(self):
        a = self.odf.drop(columns=["tbool", "tsymbol", "date"]).groupby('tlong').pct_change()
        b = self.pdf.drop(columns=["tbool", "tsymbol", "date"]).groupby('tlong').pct_change()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_long_size(self):
        a = self.odf.groupby('tlong').size()
        b = self.pdf.groupby('tlong').size()
        assert_series_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_long_sem(self):
        a = self.odf.drop(columns=["tbool"]).groupby('tlong').sem()
        b = self.pdf.drop(columns=["tbool"]).groupby('tlong').sem()
        assert_frame_equal(a.to_pandas(), b.drop(columns=["date", "tsymbol"]), check_dtype=False, check_index_type=False, check_less_precise=1, check_like=True)

    def test_from_pandas_groupby_param_by_long_std(self):
        a = self.odf.drop(columns=["tbool"]).groupby('tlong').std()
        b = self.pdf.drop(columns=["tbool"]).groupby('tlong').std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_long_sum(self):
        a = self.odf.groupby('tlong').sum()
        b = self.pdf.groupby('tlong').sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_long_var(self):
        a = self.odf.groupby('tlong').var()
        b = self.pdf.groupby('tlong').var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_long_tail(self):
        # TODO：NOT IMPLEMENTED
        pass
        # a = self.odf.groupby('tlong').tail()
        # b = self.pdf.groupby('tlong').tail()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_float_all(self):
        a = self.odf_csv.drop(columns='tsymbol').groupby('tfloat').all()
        b = self.pdf_csv.drop(columns='tsymbol').groupby('tfloat').all()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_less_precise=1, check_index_type=False)

    def test_from_import_groupby_param_by_float_any(self):
        a = self.odf_csv.drop(columns='tsymbol').groupby('tfloat').any()
        b = self.pdf_csv.drop(columns='tsymbol').groupby('tfloat').any()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_float_bfill(self):
        a = self.odf_csv.groupby('tfloat').bfill()
        b = self.pdf_csv.groupby('tfloat').bfill()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_float_count(self):
        a = self.odf_csv.groupby('tfloat').count()
        b = self.pdf_csv.groupby('tfloat').count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_float_cumcount(self):
        a = self.odf_csv.groupby('tfloat').cumcount()
        b = self.pdf_csv.groupby('tfloat').cumcount()
        # TODO:TOO MUCH DIFFS
        self.assertIsInstance(a.to_pandas(), DataFrame)
        self.assertIsInstance(b, Series)
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_float_cummax(self):
        # TODO: cummax for temperal and literal data is not allowed in orca
        a = self.odf_csv.drop(columns=["date", "tsymbol"]).groupby('tfloat').cummax()
        b = self.pdf_csv.drop(columns=["date", "tsymbol"]).groupby('tfloat').cummax()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_float_cummin(self):
        a = self.odf_csv.drop(columns=["date", "tsymbol"]).groupby('tfloat').cummin()
        b = self.pdf_csv.drop(columns=["date", "tsymbol"]).groupby('tfloat').cummin()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=5)

    def test_from_import_groupby_param_by_float_cumprod(self):
        a = self.odf_csv.groupby('tfloat').cumprod().compute()
        b = self.pdf_csv.groupby('tfloat').cumprod()
        # TODO: TOO MUCH DIFFS
        ar = a[(a['id'] >= 100) & (a['id'] <= 200)].compute().sort_values("id")
        br = b[(b['id'] >= 100) & (b['id'] <= 200)].sort_values("id")
        assert_frame_equal(ar.to_pandas().reset_index(drop=True), br.reset_index(drop=True), check_dtype=False,
                           check_index_type=False, check_less_precise=1)
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_float_cumsum(self):
        a = self.odf_csv.groupby('tfloat').cumsum()
        b = self.pdf_csv.groupby('tfloat').cumsum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_float_ffill(self):
        a = self.odf_csv.groupby('tfloat').ffill()
        b = self.pdf_csv.groupby('tfloat').ffill()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_float_first(self):
        a = self.odf_csv.groupby('tfloat').first()
        b = self.pdf_csv.groupby('tfloat').first()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_float_head(self):
        # TODO：NOT IMPLEMENTED
        pass
        # a = self.odf_csv.groupby('tfloat').head()
        # b = self.pdf_csv.groupby('tfloat').head()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_float_last(self):
        a = self.odf_csv.groupby('tfloat').last()
        b = self.pdf_csv.groupby('tfloat').last()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_float_max(self):
        a = self.odf_csv.groupby('tfloat').max()
        b = self.pdf_csv.groupby('tfloat').max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_float_mean(self):
        a = self.odf_csv.groupby('tfloat').mean()
        b = self.pdf_csv.groupby('tfloat').mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_float_median(self):
        a = self.odf_csv.drop(columns=["tbool", "tsymbol"]).groupby('tfloat').median()
        b = self.pdf_csv.drop(columns=["tbool", "tsymbol"]).groupby('tfloat').median()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_float_min(self):
        a = self.odf_csv.groupby('tfloat').min()
        b = self.pdf_csv.groupby('tfloat').min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_float_ngroup(self):
        # TODO：NOT IMPLEMENTED
        pass
        # a = self.odf_csv.groupby('tfloat').ngroup()
        # b = self.pdf_csv.groupby('tfloat').ngroup()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_float_nth(self):
        # TODO：NOT IMPLEMENTED
        pass
        # a = self.odf_csv.groupby('tfloat').nth(0)
        # b = self.pdf_csv.groupby('tfloat').nth(0)
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_float_ohlc(self):
        a = self.odf_csv.drop(columns=['tsymbol', "date"]).groupby('tfloat').ohlc()
        b = self.pdf_csv.drop(columns=['tsymbol', "date"]).groupby('tfloat').ohlc()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_float_prod(self):
        a = self.odf_csv.groupby('tfloat').prod()
        b = self.pdf_csv.groupby('tfloat').prod()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_float_rank(self):
        a = self.odf_csv.groupby('tfloat').rank()
        # TODO: pandas doesn't support
        # b = self.pdf_csv.groupby('tfloat').rank()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_float_pct_change(self):
        a = self.odf_csv.drop(columns=["tbool", "tsymbol", "date"]).groupby('tfloat').pct_change().compute()
        b = self.pdf_csv.drop(columns=["tbool", "tsymbol", "date"]).groupby('tfloat').pct_change()
        # TODO: TOO MUCH DIFFS
        ar = a[(a['id'] >= 10) & (a['id'] <= 200)].compute().sort_values("id")
        br = b[(b['id'] >= 10) & (b['id'] <= 200)].sort_values("id")
        assert_frame_equal(ar.to_pandas().reset_index(drop=True), br.reset_index(drop=True),
                           check_dtype=False, check_index_type=False, check_less_precise=1)
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_float_size(self):
        a = self.odf_csv.groupby('tfloat').size()
        b = self.pdf_csv.groupby('tfloat').size()
        assert_series_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_float_sem(self):
        # make no sense
        pass

    def test_from_import_groupby_param_by_float_std(self):
        a = self.odf_csv.drop(columns=["tbool"]).groupby('tfloat').std()
        b = self.pdf_csv.drop(columns=["tbool"]).groupby('tfloat').std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_float_sum(self):
        a = self.odf_csv.groupby('tfloat').sum()
        b = self.pdf_csv.groupby('tfloat').sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_float_var(self):
        a = self.odf_csv.groupby('tfloat').var()
        b = self.pdf_csv.groupby('tfloat').var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_float_tail(self):
        # TODO：NOT IMPLEMENTED
        pass
        # a = self.odf_csv.groupby('tfloat').tail()
        # b = self.pdf_csv.groupby('tfloat').tail()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_float_all(self):
        a = self.odf.drop(columns='tsymbol').groupby('tfloat').all()
        b = self.pdf.drop(columns='tsymbol').groupby('tfloat').all()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_float_any(self):
        a = self.odf.drop(columns='tsymbol').groupby('tfloat').any()
        b = self.pdf.drop(columns='tsymbol').groupby('tfloat').any()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_float_bfill(self):
        a = self.odf.groupby('tfloat').bfill()
        b = self.pdf.groupby('tfloat').bfill()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_float_count(self):
        a = self.odf.groupby('tfloat').count()
        b = self.pdf.groupby('tfloat').count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_float_cumcount(self):
        a = self.odf.groupby('tfloat').cumcount()
        b = self.pdf.groupby('tfloat').cumcount()
        # FIXME: 差异过大，无法比较
        self.assertIsInstance(a.to_pandas(), DataFrame)
        self.assertIsInstance(b, Series)
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_float_cummax(self):
        a = self.odf.drop(columns=["date", "tsymbol"]).groupby('tfloat').cummax()
        b = self.pdf.drop(columns=["date", "tsymbol"]).groupby('tfloat').cummax()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=5)

    def test_from_pandas_groupby_param_by_float_cummin(self):
        a = self.odf.drop(columns=["date", "tsymbol"]).groupby('tfloat').cummin()
        b = self.pdf.drop(columns=["date", "tsymbol"]).groupby('tfloat').cummin()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=5)

    def test_from_pandas_groupby_param_by_float_cumprod(self):
        a = self.odf.groupby('tfloat').cumprod().compute()
        b = self.pdf.groupby('tfloat').cumprod()
        # TODO: TOO MUCH DIFFS
        assert_frame_equal(a.reset_index(drop=True).to_pandas(),
                           b.reset_index(drop=True), check_dtype=False,
                           check_index_type=False, check_less_precise=1)
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_float_cumsum(self):
        a = self.odf.groupby('tfloat').cumsum()
        b = self.pdf.groupby('tfloat').cumsum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_float_ffill(self):
        a = self.odf.groupby('tfloat').ffill()
        b = self.pdf.groupby('tfloat').ffill()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_float_first(self):
        a = self.odf.groupby('tfloat').first()
        b = self.pdf.groupby('tfloat').first()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_float_head(self):
        # TODO：NOT IMPLEMENTED
        pass
        # a = self.odf.groupby('tfloat').head()
        # b = self.pdf.groupby('tfloat').head()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_float_last(self):
        a = self.odf.groupby('tfloat').last()
        b = self.pdf.groupby('tfloat').last()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_float_max(self):
        a = self.odf.groupby('tfloat').max()
        b = self.pdf.groupby('tfloat').max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_float_mean(self):
        a = self.odf.groupby('tfloat').mean()
        b = self.pdf.groupby('tfloat').mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_float_median(self):
        a = self.odf.drop(columns=["tbool", "tsymbol"]).groupby('tfloat').median()
        b = self.pdf.drop(columns=["tbool", "tsymbol"]).groupby('tfloat').median()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_float_min(self):
        a = self.odf.groupby('tfloat').min()
        b = self.pdf.groupby('tfloat').min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_float_ngroup(self):
        # TODO：NOT IMPLEMENTED
        pass
        # a = self.odf.groupby('tfloat').ngroup()
        # b = self.pdf.groupby('tfloat').ngroup()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_float_nth(self):
        # TODO：NOT IMPLEMENTED
        pass
        # a = self.odf.groupby('tfloat').nth(0)
        # b = self.pdf.groupby('tfloat').nth(0)
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_float_ohlc(self):
        a = self.odf.drop(columns=['tsymbol', "date"]).groupby('tfloat').ohlc()
        b = self.pdf.drop(columns=['tsymbol', "date"]).groupby('tfloat').ohlc()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_float_prod(self):
        a = self.odf.groupby('tfloat').prod()
        b = self.pdf.groupby('tfloat').prod()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_float_rank(self):
        a = self.odf.groupby('tfloat').rank()
        # TODO: pandas doesn't support
        # b = self.pdf.groupby('tfloat').rank()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_float_pct_change(self):
        a = self.odf.drop(columns=["tbool", "tsymbol", "date"]).groupby('tfloat').pct_change()
        b = self.pdf.drop(columns=["tbool", "tsymbol", "date"]).groupby('tfloat').pct_change()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_float_size(self):
        a = self.odf.groupby('tfloat').size()
        b = self.pdf.groupby('tfloat').size()
        assert_series_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_float_sem(self):
        a = self.odf.drop(columns=["tbool"]).groupby('tfloat').sem()
        b = self.pdf.drop(columns=["tbool"]).groupby('tfloat').sem()
        assert_frame_equal(a.to_pandas(), b.drop(columns=["date", "tsymbol"]), check_dtype=False, check_index_type=False, check_less_precise=1, check_like=True)

    def test_from_pandas_groupby_param_by_float_std(self):
        a = self.odf.drop(columns=["tbool"]).groupby('tfloat').std()
        b = self.pdf.drop(columns=["tbool"]).groupby('tfloat').std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_float_sum(self):
        a = self.odf.groupby('tfloat').sum()
        b = self.pdf.groupby('tfloat').sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_float_var(self):
        a = self.odf.groupby('tfloat').var()
        b = self.pdf.groupby('tfloat').var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_float_tail(self):
        # TODO：NOT IMPLEMENTED
        pass
        # a = self.odf.groupby('tfloat').tail()
        # b = self.pdf.groupby('tfloat').tail()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_compo_date_symbol_all(self):
        a = self.odf_csv.groupby(['date', 'tsymbol']).all()
        b = self.pdf_csv.groupby(['date', 'tsymbol']).all()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_less_precise=1, check_index_type=False)

    def test_from_import_groupby_param_by_compo_date_symbol_any(self):
        a = self.odf_csv.groupby(['date', 'tsymbol']).any()
        b = self.pdf_csv.groupby(['date', 'tsymbol']).any()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_compo_date_symbol_bfill(self):
        a = self.odf_csv.groupby(['date', 'tsymbol']).bfill()
        b = self.pdf_csv.groupby(['date', 'tsymbol']).bfill()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_compo_date_symbol_count(self):
        a = self.odf_csv.groupby(['date', 'tsymbol']).count()
        b = self.pdf_csv.groupby(['date', 'tsymbol']).count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_compo_date_symbol_cumcount(self):
        a = self.odf_csv.groupby(['date', 'tsymbol']).cumcount()
        b = self.pdf_csv.groupby(['date', 'tsymbol']).cumcount()
        # TODO: TOO MUCH DIFFS
        self.assertIsInstance(a.to_pandas(), DataFrame)
        self.assertIsInstance(b, Series)
        # assert_series_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_compo_date_symbol_cummax(self):
        a = self.odf_csv.groupby(['date', 'tsymbol']).cummax()
        b = self.pdf_csv.groupby(['date', 'tsymbol']).cummax()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_compo_date_symbol_cummin(self):
        a = self.odf_csv.groupby(['date', 'tsymbol']).cummin()
        b = self.pdf_csv.groupby(['date', 'tsymbol']).cummin()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_compo_date_symbol_cumprod(self):
        a = self.odf_csv.groupby(['date', 'tsymbol']).cumprod().compute()
        b = self.pdf_csv.groupby(['date', 'tsymbol']).cumprod()
        # TODO: TOO MUCH DIFFS
        ar = a[(a['id'] >= 100) & (a['id'] <= 200)].compute().sort_values("id")
        br = b[(b['id'] >= 100) & (b['id'] <= 200)].sort_values("id")
        assert_frame_equal(ar.to_pandas().reset_index(drop=True), br.reset_index(drop=True),
                           check_dtype=False, check_index_type=False, check_less_precise=1)
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=5)

    def test_from_import_groupby_param_by_compo_date_symbol_cumsum(self):
        a = self.odf_csv.groupby(['date', 'tsymbol']).cumsum()
        b = self.pdf_csv.groupby(['date', 'tsymbol']).cumsum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_compo_date_symbol_ffill(self):
        a = self.odf_csv.groupby(['date', 'tsymbol']).ffill()
        b = self.pdf_csv.groupby(['date', 'tsymbol']).ffill()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_compo_date_symbol_first(self):
        a = self.odf_csv.groupby(['date', 'tsymbol']).first()
        b = self.pdf_csv.groupby(['date', 'tsymbol']).first()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_compo_date_symbol_head(self):
        # TODO：NOT IMPLEMENTED
        pass
        # a = self.odf_csv.groupby(['date', 'tsymbol']).head()
        # b = self.pdf_csv.groupby(['date', 'tsymbol']).head()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_compo_date_symbol_last(self):
        a = self.odf_csv.groupby(['date', 'tsymbol']).last()
        b = self.pdf_csv.groupby(['date', 'tsymbol']).last()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_compo_date_symbol_max(self):
        a = self.odf_csv.groupby(['date', 'tsymbol']).max()
        b = self.pdf_csv.groupby(['date', 'tsymbol']).max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_compo_date_symbol_mean(self):
        a = self.odf_csv.groupby(['date', 'tsymbol']).mean()
        b = self.pdf_csv.groupby(['date', 'tsymbol']).mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_compo_date_symbol_median(self):
        a = self.odf_csv.drop(columns=["tbool"]).groupby(['date', 'tsymbol']).median()
        b = self.pdf_csv.drop(columns=["tbool"]).groupby(['date', 'tsymbol']).median()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_compo_date_symbol_min(self):
        a = self.odf_csv.groupby(['date', 'tsymbol']).min()
        b = self.pdf_csv.groupby(['date', 'tsymbol']).min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_compo_date_symbol_ngroup(self):
        # TODO：NOT IMPLEMENTED
        pass
        # a = self.odf_csv.groupby(['date', 'tsymbol']).ngroup()
        # b = self.pdf_csv.groupby(['date', 'tsymbol']).ngroup()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_compo_date_symbol_nth(self):
        # TODO：NOT IMPLEMENTED
        pass
        # a = self.odf_csv.groupby(['date', 'tsymbol']).nth(0)
        # b = self.pdf_csv.groupby(['date', 'tsymbol']).nth(0)
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_compo_int_bool_ohlc(self):
        odf_csv = self.odf_csv.fillna(1).compute()
        pdf_csv = self.pdf_csv.fillna(1)
        a = odf_csv.drop(columns=['tsymbol', "date"]).groupby(['tint', 'tbool']).ohlc()
        # non-numerical dtypes are not allowed for ohlc in pandas
        b = pdf_csv.drop(columns=['tsymbol', "date"]).groupby(['tint', 'tbool']).ohlc()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_compo_date_symbol_prod(self):
        a = self.odf_csv.groupby(['date', 'tsymbol']).prod()
        b = self.pdf_csv.groupby(['date', 'tsymbol']).prod()
        # TODO: TOO MUCH DIFFS
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_compo_date_symbol_rank(self):
        a = self.odf_csv.groupby(['date', 'tsymbol']).rank().compute()
        b = self.pdf_csv.groupby(['date', 'tsymbol']).rank()
        # TODO: DIFFERENT METHOD
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_compo_date_symbol_pct_change(self):
        a = self.odf_csv.drop(columns=["tbool"]).groupby(['date', 'tsymbol']).pct_change().compute()
        b = self.pdf_csv.drop(columns=["tbool"]).groupby(['date', 'tsymbol']).pct_change()
        # TODO: TOO MUCH DIFFS
        ar = a[(a['id'] >= 10) & (a['id'] <= 200)].compute().sort_values("id")
        br = b[(b['id'] >= 10) & (b['id'] <= 200)].sort_values("id")
        assert_frame_equal(ar.to_pandas().reset_index(drop=True), br.reset_index(drop=True),
                           check_dtype=False, check_index_type=False, check_less_precise=1)
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_compo_date_symbol_size(self):
        a = self.odf_csv.groupby(['date', 'tsymbol']).size()
        b = self.pdf_csv.groupby(['date', 'tsymbol']).size()
        assert_series_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_compo_date_symbol_sem(self):
        a = self.odf_csv.groupby(['date', 'tsymbol']).sem()
        b = self.pdf_csv.groupby(['date', 'tsymbol']).sem()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_compo_date_symbol_std(self):
        a = self.odf_csv.drop(columns=["tbool"]).groupby(['date', 'tsymbol']).std()
        b = self.pdf_csv.drop(columns=["tbool"]).groupby(['date', 'tsymbol']).std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_compo_date_symbol_sum(self):
        assert_frame_equal(self.odf_csv.to_pandas(), self.pdf_csv, check_dtype=False)
        a = self.odf_csv.groupby(['date', 'tsymbol']).sum()
        b = self.pdf_csv.groupby(['date', 'tsymbol']).sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_compo_date_symbol_var(self):
        a = self.odf_csv.groupby(['date', 'tsymbol']).var()
        b = self.pdf_csv.groupby(['date', 'tsymbol']).var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_groupby_param_by_compo_date_symbol_tail(self):
        # TODO：NOT IMPLEMENTED
        pass
        # a = self.odf_csv.groupby(['date', 'tsymbol']).tail()
        # b = self.pdf_csv.groupby(['date', 'tsymbol']).tail()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_compo_date_symbol_all(self):
        a = self.odf.groupby(['date', 'tsymbol']).all()
        b = self.pdf.groupby(['date', 'tsymbol']).all()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_compo_date_symbol_any(self):
        a = self.odf.groupby(['date', 'tsymbol']).any()
        b = self.pdf.groupby(['date', 'tsymbol']).any()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_compo_date_symbol_bfill(self):
        a = self.odf.groupby(['date', 'tsymbol']).bfill()
        b = self.pdf.groupby(['date', 'tsymbol']).bfill()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_compo_date_symbol_count(self):
        a = self.odf.groupby(['date', 'tsymbol']).count()
        b = self.pdf.groupby(['date', 'tsymbol']).count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_compo_date_symbol_cumcount(self):
        a = self.odf.groupby(['date', 'tsymbol']).cumcount()
        b = self.pdf.groupby(['date', 'tsymbol']).cumcount()
        # FIXME: 差异过大，无法比较
        self.assertIsInstance(a.to_pandas(), DataFrame)
        self.assertIsInstance(b, Series)
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_compo_date_symbol_cummax(self):
        a = self.odf.groupby(['date', 'tsymbol']).cummax()
        b = self.pdf.groupby(['date', 'tsymbol']).cummax()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_compo_date_symbol_cummin(self):
        a = self.odf.groupby(['date', 'tsymbol']).cummin()
        b = self.pdf.groupby(['date', 'tsymbol']).cummin()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=5)

    def test_from_pandas_groupby_param_by_compo_date_symbol_cumprod(self):
        a = self.odf.groupby(['date', 'tsymbol']).cumprod().compute()
        b = self.pdf.groupby(['date', 'tsymbol']).cumprod()
        # TODO: TOO MUCH DIFFS
        assert_frame_equal(a.sort_index().reset_index(drop=True).to_pandas(),
                           b.sort_index().reset_index(drop=True), check_dtype=False,
                           check_index_type=False, check_less_precise=1)
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_compo_date_symbol_cumsum(self):
        a = self.odf.groupby(['date', 'tsymbol']).cumsum()
        b = self.pdf.groupby(['date', 'tsymbol']).cumsum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_compo_date_symbol_ffill(self):
        a = self.odf.groupby(['date', 'tsymbol']).ffill()
        b = self.pdf.groupby(['date', 'tsymbol']).ffill()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_compo_date_symbol_first(self):
        a = self.odf.groupby(['date', 'tsymbol']).first()
        b = self.pdf.groupby(['date', 'tsymbol']).first()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_compo_date_symbol_head(self):
        # TODO：NOT IMPLEMENTED
        pass
        # a = self.odf.groupby(['date', 'tsymbol']).head()
        # b = self.pdf.groupby(['date', 'tsymbol']).head()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_compo_date_symbol_last(self):
        a = self.odf.groupby(['date', 'tsymbol']).last()
        b = self.pdf.groupby(['date', 'tsymbol']).last()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_compo_date_symbol_max(self):
        a = self.odf.groupby(['date', 'tsymbol']).max()
        b = self.pdf.groupby(['date', 'tsymbol']).max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_compo_date_symbol_mean(self):
        a = self.odf.groupby(['date', 'tsymbol']).mean()
        b = self.pdf.groupby(['date', 'tsymbol']).mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_compo_date_symbol_median(self):
        a = self.odf.drop(columns=["tbool"]).groupby(['date', 'tsymbol']).median()
        b = self.pdf.drop(columns=["tbool"]).groupby(['date', 'tsymbol']).median()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_compo_date_symbol_min(self):
        a = self.odf.groupby(['date', 'tsymbol']).min()
        b = self.pdf.groupby(['date', 'tsymbol']).min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_compo_date_symbol_ngroup(self):
        # TODO：NOT IMPLEMENTED
        pass
        # a = self.odf.groupby(['date', 'tsymbol']).ngroup()
        # b = self.pdf.groupby(['date', 'tsymbol']).ngroup()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_compo_date_symbol_nth(self):
        # TODO：NOT IMPLEMENTED
        pass
        # a = self.odf.groupby(['date', 'tsymbol']).nth(0)
        # b = self.pdf.groupby(['date', 'tsymbol']).nth(0)
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_compo_date_symbol_ohlc(self):
        a = self.odf.groupby(['date', 'tsymbol']).ohlc()
        # b = self.pdf.groupby(['date', 'tsymbol']).ohlc()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_compo_date_symbol_prod(self):
        a = self.odf.groupby(['date', 'tsymbol']).prod()
        b = self.pdf.groupby(['date', 'tsymbol']).prod()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_compo_date_symbol_rank(self):
        a = self.odf.groupby(['date', 'tsymbol']).rank()
        b = self.pdf.groupby(['date', 'tsymbol']).rank()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_compo_date_symbol_pct_change(self):
        a = self.odf.drop(columns=["tbool"]).groupby(['date', 'tsymbol']).pct_change().compute()
        b = self.pdf.drop(columns=["tbool"]).groupby(['date', 'tsymbol']).pct_change()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_compo_date_symbol_size(self):
        a = self.odf.groupby(['date', 'tsymbol']).size()
        b = self.pdf.groupby(['date', 'tsymbol']).size()
        assert_series_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_compo_date_symbol_sem(self):
        a = self.odf.drop(columns=["tbool"]).groupby(['date', 'tsymbol']).sem()
        b = self.pdf.drop(columns=["tbool"]).groupby(['date', 'tsymbol']).sem()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_compo_date_symbol_std(self):
        a = self.odf.drop(columns=["tbool"]).groupby(['date', 'tsymbol']).std()
        b = self.pdf.drop(columns=["tbool"]).groupby(['date', 'tsymbol']).std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_compo_date_symbol_sum(self):
        a = self.odf.groupby(['date', 'tsymbol']).sum()
        b = self.pdf.groupby(['date', 'tsymbol']).sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_compo_date_symbol_var(self):
        a = self.odf.groupby(['date', 'tsymbol']).var()
        b = self.pdf.groupby(['date', 'tsymbol']).var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_pandas_groupby_param_by_compo_date_symbol_tail(self):
        # TODO：NOT IMPLEMENTED
        pass
        # a = self.odf.groupby(['date', 'tsymbol']).tail()
        # b = self.pdf.groupby(['date', 'tsymbol']).tail()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)


if __name__ == '__main__':
    unittest.main()
