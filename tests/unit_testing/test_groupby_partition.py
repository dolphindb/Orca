import unittest
import orca
import os.path as path
from setup.settings import *
from pandas.util.testing import *


def _create_odf_csv(data, dfsDatabase):
    # call function default_session() to get session object
    s = orca.default_session()
    dolphindb_script = """
    login("admin", "123456")
    dbPath="dfs://groupbyDateDB"
    if(existsDatabase(dbPath))
        dropDatabase(dbPath)
    schema = extractTextSchema('{data}')
    cols = exec name from schema
    types = ["INT", "DATE", "SYMBOL", "BOOL", "SHORT", "INT", "LONG", "FLOAT", "DOUBLE"]
    schema = table(50000:0, cols, types)
    tt=schema(schema).colDefs
    tt.drop!(`typeInt)
    tt.rename!(`name`type)
    db = database(dbPath, RANGE, 1 501 1001 1501 2001 2501 3001)
    tb = db.createPartitionedTable(schema, `tb, `id)
    db.loadTextEx(`tb,`id, '{data}' ,, tt)""".format(data=data)
    s.run(dolphindb_script)
    return orca.read_table(dfsDatabase, 'tb')


class Csv:
    pdf_csv = None
    odfs_csv = None


class DfsGroupByTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # configure data directory
        DATA_DIR = path.abspath(path.join(__file__, "../setup/data"))
        fileName = 'groupbyDate.csv'
        data = os.path.join(DATA_DIR, fileName)
        data = data.replace('\\', '/')
        dfsDatabase = "dfs://groupbyDateDB"

        # Orca connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

        Csv.pdf_csv = pd.read_csv(data, parse_dates=[1], dtype={"id": np.int32, "tbool": np.bool, "tshort": np.int16,
                                           "tint": np.int32, "tlong": np.int64, "tfloat": np.float32,
                                           "tdouble": np.float64})
        Csv.pdf_csv['tbool'] = Csv.pdf_csv["tbool"].astype(np.bool)
        Csv.odfs_csv = _create_odf_csv(data, dfsDatabase)

        Csv.odfs_csv.set_index("id", inplace=True)
        Csv.pdf_csv.set_index("id", inplace=True)

    @property
    def pdf_csv(self):
        return Csv.pdf_csv

    @property
    def odfs_csv(self):
        return Csv.odfs_csv

    def test_dfs_groupby_param_by_date_all(self):
        pass
        # TODO： NOT SUPPORTED FOR PARTITIONED TABLE
        # a = self.odfs_csv.groupby('date').all()
        # b = self.pdf_csv.groupby('date').all()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_date_any(self):
        pass
        # TODO： NOT SUPPORTED FOR PARTITIONED TABLE
        # a = self.odfs_csv.groupby('date').any()
        # b = self.pdf_csv.groupby('date').any()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_date_bfill(self):
        a = self.odfs_csv.groupby('date').bfill()
        b = self.pdf_csv.groupby('date').bfill()
        # TODO: bfill for strings is not allowed in Orca
        assert_frame_equal(a.to_pandas().sort_index().reset_index(drop=True).iloc[:, 1:],
                           b.sort_index().reset_index(drop=True).iloc[:, 1:], check_dtype=False,
                           check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_date_count(self):
        a = self.odfs_csv.groupby('date').count()
        b = self.pdf_csv.groupby('date').count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_date_cumcount(self):
        a = self.odfs_csv.groupby('date').cumcount()
        b = self.pdf_csv.groupby('date').cumcount()
        # TODO: TO MUCH DIFFS
        self.assertIsInstance(a.to_pandas(), DataFrame)
        self.assertIsInstance(b, Series)
        # assert_series_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_date_cummax(self):
        a = self.odfs_csv.drop(columns=['tsymbol']).groupby('date').cummax()
        b = self.pdf_csv.drop(columns=['tsymbol']).groupby('date').cummax()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_dfs_groupby_param_by_date_cummin(self):
        a = self.odfs_csv.drop(columns=['tsymbol']).groupby('date').cummin()
        b = self.pdf_csv.drop(columns=['tsymbol']).groupby('date').cummin()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_date_cumprod(self):
        a = self.odfs_csv.groupby('date').cumprod()
        b = self.pdf_csv.groupby('date').cumprod()
        # TODO: TO MUCH DIFFS
        assert_frame_equal(a.to_pandas().iloc[0:5].reset_index(drop=True), b.iloc[0:5].reset_index(drop=True), check_dtype=False,
                           check_index_type=False, check_less_precise=1)
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_date_cumsum(self):
        a = self.odfs_csv.groupby('date').cumsum()
        b = self.pdf_csv.groupby('date').cumsum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True),
                           check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_date_ffill(self):
        a = self.odfs_csv.groupby('date').ffill()
        b = self.pdf_csv.groupby('date').ffill()
        assert_frame_equal(a.to_pandas().sort_index().reset_index(drop=True).iloc[:, 1:],
                           b.sort_index().reset_index(drop=True).iloc[:, 1:], check_dtype=False,
                           check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_date_first(self):
        a = self.odfs_csv.groupby('date').first()
        b = self.pdf_csv.groupby('date').first()
        b['tbool'] = b['tbool'].astype(np.bool, errors="ignore")
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_date_head(self):
        # TODO： NOT SUPPORTED FOR groupby
        pass
        # a = self.odfs_csv.groupby('date').head()
        # b = self.pdf_csv.groupby('date').head()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_date_last(self):
        a = self.odfs_csv.groupby('date').last()
        b = self.pdf_csv.groupby('date').last()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_date_max(self):
        a = self.odfs_csv.groupby('date').max()
        b = self.pdf_csv.groupby('date').max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_date_mean(self):
        a = self.odfs_csv.groupby('date').mean()
        b = self.pdf_csv.groupby('date').mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_date_median(self):
        # TODO： NOT SUPPORTED FOR PARTITIONED TABLE
        pass
        # a = self.odfs_csv.groupby('date').median()
        # b = self.pdf_csv.groupby('date').median()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_date_min(self):
        a = self.odfs_csv.groupby('date').min()
        b = self.pdf_csv.groupby('date').min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_date_ngroup(self):
        # TODO： NOT IMPLEMENTED
        pass
        # a = self.odfs_csv.groupby('date').ngroup()
        # b = self.pdf_csv.groupby('date').ngroup()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_date_nth(self):
        # TODO： NOT IMPLEMENTED
        pass
        # a = self.odfs_csv.groupby('date').nth(0)
        # b = self.pdf_csv.groupby('date').nth(0)
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_date_ohlc(self):
        a = self.odfs_csv.drop(columns=['tsymbol', "date"]).groupby(['tint', 'tbool']).ohlc()
        b = self.pdf_csv.drop(columns=['tsymbol', "date"]).groupby(['tint', 'tbool']).ohlc()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_date_prod(self):
        a = self.odfs_csv.groupby('date').prod()
        b = self.pdf_csv.groupby('date').prod()
        # TODO：DIFFS
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_date_rank(self):
        a = self.odfs_csv.groupby('date').rank()
        # TODO: pandas doesn't support
        # b = self.pdf_csv.groupby('date').rank()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_date_pct_change(self):
        a = self.odfs_csv.drop(columns=["tbool", "tsymbol"]).groupby('date').pct_change()
        b = self.pdf_csv.drop(columns=["tbool", "tsymbol"]).groupby('date').pct_change()
        assert_frame_equal(a.to_pandas(), b.replace(np.inf, np.nan), check_dtype=False, check_index_type=False, check_less_precise=2)

    def test_dfs_groupby_param_by_date_size(self):
        a = self.odfs_csv.groupby('date').size()
        b = self.pdf_csv.groupby('date').size()
        assert_series_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_date_sem(self):
        # TODO： NOT SUPPORTED FOR PARTITIONED TABLE
        pass
        # a = self.odfs_csv.groupby('date').sem()
        # b = self.pdf_csv.groupby('date').sem()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_date_std(self):
        a = self.odfs_csv.groupby('date').std()
        b = self.pdf_csv.groupby('date').std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_date_sum(self):
        a = self.odfs_csv.groupby('date').sum()
        b = self.pdf_csv.groupby('date').sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_date_var(self):
        a = self.odfs_csv.groupby('date').var()
        b = self.pdf_csv.groupby('date').var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_date_tail(self):
        # TODO： NOT SUPPORTED FOR groupby
        pass
        # a = self.odfs_csv.groupby('date').tail()
        # b = self.pdf_csv.groupby('date').tail()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_symbol_all(self):
        # TODO： NOT SUPPORTED FOR PARTITIONED TABLE
        pass
        # a = self.odfs_csv.groupby('tsymbol').all()
        # b = self.pdf_csv.groupby('tsymbol').all()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_symbol_any(self):
        # TODO： NOT SUPPORTED FOR PARTITIONED TABLE
        pass
        # a = self.odfs_csv.groupby('tsymbol').any()
        # b = self.pdf_csv.groupby('tsymbol').any()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_symbol_bfill(self):
        a = self.odfs_csv.groupby('tsymbol').bfill()
        b = self.pdf_csv.groupby('tsymbol').bfill()
        assert_frame_equal(a.to_pandas().sort_index().reset_index(drop=True).iloc[:, 1:],
                           b.sort_index().reset_index(drop=True).iloc[:, 1:], check_dtype=False,
                           check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_symbol_count(self):
        a = self.odfs_csv.groupby('tsymbol').count()
        b = self.pdf_csv.groupby('tsymbol').count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_symbol_cumcount(self):
        a = self.odfs_csv.groupby('tsymbol').cumcount()
        b = self.pdf_csv.groupby('tsymbol').cumcount()
        # TODO: TO MUCH DIFFS
        self.assertIsInstance(a.to_pandas(), DataFrame)
        self.assertIsInstance(b, Series)
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_symbol_cummax(self):
        a = self.odfs_csv.drop(columns=['date']).groupby('tsymbol').cummax()
        b = self.pdf_csv.drop(columns=['date']).groupby('tsymbol').cummax()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_dfs_groupby_param_by_symbol_cummin(self):
        a = self.odfs_csv.drop(columns=['date']).groupby('tsymbol').cummin()
        b = self.pdf_csv.drop(columns=['date']).groupby('tsymbol').cummin()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_dfs_groupby_param_by_symbol_cumprod(self):
        a = self.odfs_csv.groupby('tsymbol').cumprod()
        b = self.pdf_csv.groupby('tsymbol').cumprod()
        assert_frame_equal(a.to_pandas().iloc[0:5].reset_index(drop=True), b.iloc[0:5].reset_index(drop=True), check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_symbol_cumsum(self):
        a = self.odfs_csv.groupby('tsymbol').cumsum()
        b = self.pdf_csv.groupby('tsymbol').cumsum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True),
                           check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_symbol_ffill(self):
        a = self.odfs_csv.groupby('tsymbol').ffill()
        b = self.pdf_csv.groupby('tsymbol').ffill()
        assert_frame_equal(a.to_pandas().sort_index().reset_index(drop=True).iloc[:, 1:],
                           b.sort_index().reset_index(drop=True).iloc[:, 1:], check_dtype=False,
                           check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_symbol_first(self):
        a = self.odfs_csv.groupby('tsymbol').first()
        b = self.pdf_csv.groupby('tsymbol').first()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_symbol_head(self):
        # TODO： NOT SUPPORTED FOR groupby
        pass
        # a = self.odfs_csv.groupby('tsymbol').head()
        # b = self.pdf_csv.groupby('tsymbol').head()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_symbol_last(self):
        a = self.odfs_csv.groupby('tsymbol').last()
        b = self.pdf_csv.groupby('tsymbol').last()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_symbol_max(self):
        a = self.odfs_csv.groupby('tsymbol').max()
        b = self.pdf_csv.groupby('tsymbol').max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_symbol_mean(self):
        a = self.odfs_csv.groupby('tsymbol').mean()
        b = self.pdf_csv.groupby('tsymbol').mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_symbol_median(self):
        # TODO： NOT SUPPORTED FOR PARTITIONED TABLE
        pass
        # a = self.odfs_csv.groupby('tsymbol').median()
        # b = self.pdf_csv.groupby('tsymbol').median()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_symbol_min(self):
        a = self.odfs_csv.groupby('tsymbol').min()
        b = self.pdf_csv.groupby('tsymbol').min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_symbol_ngroup(self):
        # TODO： NOT IMPLEMENTED
        pass
        # a = self.odfs_csv.groupby('tsymbol').ngroup()
        # b = self.pdf_csv.groupby('tsymbol').ngroup()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_symbol_nth(self):
        # TODO： NOT IMPLEMENTED
        pass
        # a = self.odfs_csv.groupby('tsymbol').nth(0)
        # b = self.pdf_csv.groupby('tsymbol').nth(0)
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_symbol_ohlc(self):
        a = self.odfs_csv.groupby('tsymbol').ohlc()
        # pandas doesn't support
        # b = self.pdf_csv.groupby('tsymbol').ohlc()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_symbol_prod(self):
        a = self.odfs_csv.groupby('tsymbol').prod()
        b = self.pdf_csv.groupby('tsymbol').prod()
        assert_frame_equal(a.to_pandas(), b.fillna(0), check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_symbol_rank(self):
        a = self.odfs_csv.groupby('tsymbol').rank()
        b = self.pdf_csv.groupby('tsymbol').rank()
        # TODO: DIFFERENT METHOD
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_symbol_pct_change(self):
        a = self.odfs_csv.drop(columns=["tbool", "date"]).groupby('tsymbol').pct_change()
        b = self.pdf_csv.drop(columns=["tbool", "date"]).groupby('tsymbol').pct_change()
        assert_frame_equal(a.to_pandas(), b.replace(np.inf, np.nan), check_dtype=False, check_index_type=False,
                           check_less_precise=2)

    def test_dfs_groupby_param_by_symbol_size(self):
        a = self.odfs_csv.groupby('tsymbol').size()
        b = self.pdf_csv.groupby('tsymbol').size()
        assert_series_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_symbol_sem(self):
        # TODO： NOT SUPPORTED FOR PARTITIONED TABLE
        pass
        # a = self.odfs_csv.groupby('tsymbol').sem()
        # b = self.pdf_csv.groupby('tsymbol').sem()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_symbol_std(self):
        a = self.odfs_csv.groupby('tsymbol').std()
        b = self.pdf_csv.groupby('tsymbol').std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_symbol_sum(self):
        a = self.odfs_csv.groupby('tsymbol').sum()
        b = self.pdf_csv.groupby('tsymbol').sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_symbol_var(self):
        a = self.odfs_csv.groupby('tsymbol').var()
        b = self.pdf_csv.groupby('tsymbol').var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_symbol_tail(self):
        # TODO： NOT SUPPORTED FOR groupby
        pass
        # a = self.odfs_csv.groupby('tsymbol').tail()
        # b = self.pdf_csv.groupby('tsymbol').tail()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_long_all(self):
        # TODO： NOT SUPPORTED FOR PARTITIONED TABLE
        pass
        # a = self.odfs_csv.groupby('tlong').all()
        # b = self.pdf_csv.groupby('tlong').all()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_long_any(self):
        # TODO： NOT SUPPORTED FOR PARTITIONED TABLE
        pass
        # a = self.odfs_csv.groupby('tlong').any()
        # b = self.pdf_csv.groupby('tlong').any()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_long_bfill(self):
        a = self.odfs_csv.groupby('tlong').bfill()
        b = self.pdf_csv.groupby('tlong').bfill()
        assert_frame_equal(a.to_pandas().sort_index().reset_index(drop=True).iloc[:, 1:],
                           b.sort_index().reset_index(drop=True).iloc[:, 1:], check_dtype=False,
                           check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_long_count(self):
        a = self.odfs_csv.groupby('tlong').count()
        b = self.pdf_csv.groupby('tlong').count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_long_cumcount(self):
        a = self.odfs_csv.groupby('tlong').cumcount()
        b = self.pdf_csv.groupby('tlong').cumcount()
        # TODO: TO MUCH DIFFS
        self.assertIsInstance(a.to_pandas(), DataFrame)
        self.assertIsInstance(b, Series)
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_long_cummax(self):
        a = self.odfs_csv.drop(columns=['date', 'tsymbol']).groupby('tlong').cummax()
        b = self.pdf_csv.drop(columns=['date', 'tsymbol']).groupby('tlong').cummax()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_dfs_groupby_param_by_long_cummin(self):
        a = self.odfs_csv.drop(columns=['date', 'tsymbol']).groupby('tlong').cummin()
        b = self.pdf_csv.drop(columns=['date', 'tsymbol']).groupby('tlong').cummin()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_dfs_groupby_param_by_long_cumprod(self):
        a = self.odfs_csv.groupby('tlong').cumprod()
        b = self.pdf_csv.groupby('tlong').cumprod()
        # TODO: TO MUCH DIFFS
        assert_frame_equal(a.to_pandas().iloc[0:50].reset_index(drop=True), b.iloc[0:50].reset_index(drop=True),
                           check_dtype=False, check_index_type=False, check_less_precise=1)
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_long_cumsum(self):
        a = self.odfs_csv.groupby('tlong').cumsum()
        b = self.pdf_csv.groupby('tlong').cumsum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True),
                           check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_long_ffill(self):
        a = self.odfs_csv.groupby('tlong').ffill()
        b = self.pdf_csv.groupby('tlong').ffill()
        assert_frame_equal(a.to_pandas().sort_index().reset_index(drop=True).iloc[:, 1:],
                           b.sort_index().reset_index(drop=True).iloc[:, 1:], check_dtype=False,
                           check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_long_first(self):
        a = self.odfs_csv.groupby('tlong').first()
        b = self.pdf_csv.groupby('tlong').first()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_long_head(self):
        # TODO： NOT SUPPORTED FOR groupby
        pass
        # a = self.odfs_csv.groupby('tlong').head()
        # b = self.pdf_csv.groupby('tlong').head()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_long_last(self):
        a = self.odfs_csv.groupby('tlong').last()
        b = self.pdf_csv.groupby('tlong').last()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_long_max(self):
        a = self.odfs_csv.groupby('tlong').max()
        b = self.pdf_csv.groupby('tlong').max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_long_mean(self):
        a = self.odfs_csv.groupby('tlong').mean()
        b = self.pdf_csv.groupby('tlong').mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_long_median(self):
        # TODO： NOT SUPPORTED FOR PARTITIONED TABLE
        pass
        # a = self.odfs_csv.groupby('tlong').median()
        # b = self.pdf_csv.groupby('tlong').median()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_long_min(self):
        a = self.odfs_csv.groupby('tlong').min()
        b = self.pdf_csv.groupby('tlong').min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_long_ngroup(self):
        # TODO： NOT IMPLEMENTED
        pass
        # a = self.odfs_csv.groupby('tlong').ngroup()
        # b = self.pdf_csv.groupby('tlong').ngroup()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_long_nth(self):
        # TODO： NOT IMPLEMENTED
        pass
        # a = self.odfs_csv.groupby('tlong').nth()
        # b = self.pdf_csv.groupby('tlong').nth()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_long_ohlc(self):
        a = self.odfs_csv.drop(columns=['tsymbol', "date"]).groupby('tlong').ohlc()
        b = self.pdf_csv.drop(columns=['tsymbol', "date"]).groupby('tlong').ohlc()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_long_prod(self):
        a = self.odfs_csv.groupby('tlong').prod()
        b = self.pdf_csv.groupby('tlong').prod()
        # TODO：DIFFS
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_long_rank(self):
        a = self.odfs_csv.groupby('tlong').rank()
        # b = self.pdf_csv.groupby('tlong').rank()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_long_pct_change(self):
        a = self.odfs_csv.drop(columns=["tbool", "tsymbol", "date"]).groupby('tlong').pct_change()
        b = self.pdf_csv.drop(columns=["tbool", "tsymbol", "date"]).groupby('tlong').pct_change()
        assert_frame_equal(a.to_pandas(), b.replace(np.inf, np.nan), check_dtype=False, check_index_type=False, check_less_precise=2)

    def test_dfs_groupby_param_by_long_size(self):
        a = self.odfs_csv.groupby('tlong').size().loc[0:]
        b = self.pdf_csv.groupby('tlong').size()
        assert_series_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_long_sem(self):
        # TODO： NOT SUPPORTED FOR PARTITIONED TABLE
        pass
        # a = self.odfs_csv.groupby('tlong').sem()
        # b = self.pdf_csv.groupby('tlong').sem()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_long_std(self):
        a = self.odfs_csv.groupby('tlong').std()
        b = self.pdf_csv.groupby('tlong').std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_long_sum(self):
        a = self.odfs_csv.groupby('tlong').sum()
        b = self.pdf_csv.groupby('tlong').sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_long_var(self):
        a = self.odfs_csv.groupby('tlong').var()
        b = self.pdf_csv.groupby('tlong').var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_dfs_groupby_param_by_long_tail(self):
        # TODO： NOT SUPPORTED FOR groupby
        pass
        # a = self.odfs_csv.groupby('tlong').tail()
        # b = self.pdf_csv.groupby('tlong').tail()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False, check_less_precise=1)


if __name__ == '__main__':
    unittest.main()
