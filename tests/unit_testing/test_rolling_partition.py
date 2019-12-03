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
    dbPath="dfs://onlyNumericalColumnsDB"
    if(existsDatabase(dbPath))
        dropDatabase(dbPath)
    schema = extractTextSchema('{data}')
    cols = exec name from schema
    types = ["INT", "BOOL", "CHAR", "SHORT", "INT", "LONG", "FLOAT", "DOUBLE"]
    schema = table(50000:0, cols, types)
    db = database(dbPath, RANGE, 1 501 1001 1501 2001 2501 3001)
    tb = db.createPartitionedTable(schema, `tb, `id)
    tt=schema(schema).colDefs
    tt.drop!(`typeInt`comment)
    tt.rename!(`name`type)
    update tt set type="INT" where name="tchar"
    tdata=loadText('{data}' ,, tt)
    tb.append!(tdata)""".format(data=data)
    s.run(dolphindb_script)
    return orca.read_table(dfsDatabase, 'tb')


class Csv:
    pdf_csv = None
    odfs_csv = None


class DfsRollingTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # configure data directory
        DATA_DIR = path.abspath(path.join(__file__, "../setup/data"))
        fileName = 'onlyNumericalColumns.csv'
        data = os.path.join(DATA_DIR, fileName)
        data = data.replace('\\', '/')
        dfsDatabase = "dfs://onlyNumericalColumnsDB"

        # Orca connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")
        Csv.pdf_csv = pd.read_csv(data, dtype={"tbool": np.bool, "tchar": np.int8, "tshort": np.int16, "tint": np.int32,
                                               "tlong": np.int64, "tfloat": np.float32, "tdouble": np.float64})
        Csv.odfs_csv = _create_odf_csv(data, dfsDatabase)

    @property
    def pdf_csv(self):
        return Csv.pdf_csv

    @property
    def odfs_csv(self):
        return Csv.odfs_csv

    def test_dfs_rolling_param_window_sum(self):
        a = self.odfs_csv.rolling(window=5).sum().to_pandas()
        b = self.pdf_csv.rolling(window=5).sum()
        assert_frame_equal(a, b)

    def test_dfs_rolling_param_window_count(self):
        a = self.odfs_csv.rolling(window=5).count().to_pandas()
        b = self.pdf_csv.rolling(window=5).count()
        assert_frame_equal(a, b, check_dtype=False)

    def test_dfs_rolling_param_window_mean(self):
        a = self.odfs_csv.rolling(window=5).mean().to_pandas()
        b = self.pdf_csv.rolling(window=5).mean()
        assert_frame_equal(a, b)

    def test_dfs_rolling_param_window_max(self):
        assert_frame_equal(self.odfs_csv.to_pandas(), self.pdf_csv, check_dtype=False)
        a = self.odfs_csv.rolling(window=5).max().to_pandas()
        b = self.pdf_csv.rolling(window=5).max()
        assert_frame_equal(a, b, check_dtype=False)

    def test_dfs_rolling_param_window_min(self):
        a = self.odfs_csv.rolling(window=5).min().to_pandas()
        b = self.pdf_csv.rolling(window=5).min()
        assert_frame_equal(a, b, check_dtype=False)

    def test_dfs_rolling_param_window_std(self):
        a = self.odfs_csv.rolling(window=5).std().to_pandas()
        b = self.pdf_csv.rolling(window=5).std()
        assert_frame_equal(a, b)

    def test_dfs_rolling_param_window_var(self):
        a = self.odfs_csv.rolling(window=5).var().to_pandas()
        b = self.pdf_csv.rolling(window=5).var()
        assert_frame_equal(a, b)


if __name__ == '__main__':
    unittest.main()
