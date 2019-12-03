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
                dbPath="dfs://allTypesOfColumnsDB"
                if(existsDatabase(dbPath))
                    dropDatabase(dbPath)
                schema = extractTextSchema('{data}')
                cols = exec name from schema
                types = ["INT", "DATE", "MONTH", "TIME", "MINUTE", "SECOND", "DATETIME", "TIMESTAMP", "NANOTIME", "NANOTIMESTAMP", "STRING", "SYMBOL", "BOOL", "INT", "SHORT", "INT", "LONG", "DOUBLE", "DOUBLE"]
                schema = table(50000:0, cols, types)
                tt=schema(schema).colDefs
                tt.drop!(`typeInt)
                tt.rename!(`name`type)
                db = database(dbPath, RANGE, 1 501 1001 1501 2001 2501 3001)
                tb1 = db.createPartitionedTable(schema, `tb1, `id)
                db.loadTextEx(`tb1,`id, '{data}' ,, tt)""".format(data=data)
    s.run(dolphindb_script)
    return orca.read_table(dfsDatabase, 'tb1')


class Csv:
    pdf_csv = None
    odfs_csv = None


class DfsResampleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # configure data directory
        DATA_DIR = path.abspath(path.join(__file__, "../setup/data"))
        fileName = 'allTypesOfColumns.csv'
        data = os.path.join(DATA_DIR, fileName)
        data = data.replace('\\', '/')
        dfsDatabase = "dfs://allTypesOfColumnsDB"

        # Orca connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

        Csv.pdf_csv = pd.read_csv(data, parse_dates=[1, 2, 3, 4, 5, 6, 7, 8, 9])
        Csv.odfs_csv = _create_odf_csv(data, dfsDatabase)

    @property
    def pdf_csv(self):
        return Csv.pdf_csv

    @property
    def odfs_csv(self):
        return Csv.odfs_csv

    def test_dfs_resample_param_rule_year_param_on_date_count(self):
        a = self.odfs_csv.resample("y", on="date").count()
        b = self.pdf_csv.resample("y", on="date").count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        self.odfs_csv.resample("3y", on="date").count()
        self.pdf_csv.resample("3y", on="date").count()

        with self.assertRaises(TypeError):
            self.odfs_csv.resample("y").count()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("y").count()

        a = self.odfs_csv.resample("y", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = self.pdf_csv.resample("y", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odfs_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("y", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdfi.resample("y", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odfs_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("y")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdfi.resample("y")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_dfs_resample_param_rule_year_param_on_date_max(self):
        a = self.odfs_csv.resample("y", on="date").max()
        b = self.pdf_csv.resample("y", on="date").max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True).iloc[:, 0:2], b.reset_index(drop=True).iloc[:, 0:2],
                           check_dtype=False)
        assert_frame_equal(a.to_pandas().reset_index(drop=True).iloc[:, 6:8], b.reset_index(drop=True).iloc[:, 6:8],
                           check_dtype=False)
        assert_frame_equal(a.to_pandas().reset_index(drop=True).iloc[:, 9:19], b.reset_index(drop=True).iloc[:, 9:19],
                           check_dtype=False)

        with self.assertRaises(TypeError):
            self.odfs_csv.resample("y").max()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("y").max()

        a = self.odfs_csv.resample("y", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = self.pdf_csv.resample("y", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odfs_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("y", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdfi.resample("y", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odfs_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("y")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdfi.resample("y")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_dfs_resample_param_rule_year_param_on_date_mean(self):
        a = self.odfs_csv.resample("y", on="date").mean()
        b = self.pdf_csv.resample("y", on="date").mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        with self.assertRaises(TypeError):
            self.odfs_csv.resample("y").mean()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("y").mean()

        a = self.odfs_csv.resample("y", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = self.pdf_csv.resample("y", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odfs_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("y", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdfi.resample("y", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odfs_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("y")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdfi.resample("y")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_dfs_resample_param_rule_year_param_on_date_min(self):
        a = self.odfs_csv.resample("y", on="date").min()
        b = self.pdf_csv.resample("y", on="date").min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True).iloc[:, 0:2], b.reset_index(drop=True).iloc[:, 0:2],
                           check_dtype=False)
        assert_frame_equal(a.to_pandas().reset_index(drop=True).iloc[:, 6:8], b.reset_index(drop=True).iloc[:, 6:8],
                           check_dtype=False)
        assert_frame_equal(a.to_pandas().reset_index(drop=True).iloc[:, 9:19], b.reset_index(drop=True).iloc[:, 9:19],
                           check_dtype=False)

        with self.assertRaises(TypeError):
            self.odfs_csv.resample("y").min()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("y").min()

        a = self.odfs_csv.resample("y", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = self.pdf_csv.resample("y", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odfs_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("y", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdfi.resample("y", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odfs_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("y")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdfi.resample("y")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_dfs_resample_param_rule_year_param_on_date_std(self):
        a = self.odfs_csv.resample("y", on="date").std()
        b = self.pdf_csv.resample("y", on="date").std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        with self.assertRaises(TypeError):
            self.odfs_csv.resample("y").std()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("y").std()

        a = self.odfs_csv.resample("y", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = self.pdf_csv.resample("y", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odfs_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("y", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdfi.resample("y", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odfs_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("y")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdfi.resample("y")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_dfs_resample_param_rule_year_param_on_date_sum(self):
        a = self.odfs_csv.resample("y", on="date").sum()
        b = self.pdf_csv.resample("y", on="date").sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        with self.assertRaises(TypeError):
            self.odfs_csv.resample("y").sum()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("y").sum()

        a = self.odfs_csv.resample("y", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = self.pdf_csv.resample("y", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odfs_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("y", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdfi.resample("y", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odfs_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("y")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdfi.resample("y")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_dfs_resample_param_rule_year_param_on_date_var(self):
        a = self.odfs_csv.resample("y", on="date").var()
        b = self.pdf_csv.resample("y", on="date").var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        with self.assertRaises(TypeError):
            self.odfs_csv.resample("y").var()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("y").var()

        a = self.odfs_csv.resample("y", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = self.pdf_csv.resample("y", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odfs_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("y", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdfi.resample("y", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odfs_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("y")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdfi.resample("y")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_dfs_resample_param_rule_month_param_on_date_count(self):
        a = self.odfs_csv.resample("m", on="date").count()
        b = self.pdf_csv.resample("m", on="date").count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        self.odfs_csv.resample("3m", on="date").count()
        self.pdf_csv.resample("3m", on="date").count()

        with self.assertRaises(TypeError):
            self.odfs_csv.resample("m").count()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("m").count()

        a = self.odfs_csv.resample("m", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = self.pdf_csv.resample("m", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odfs_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("m", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdfi.resample("m", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odfs_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("m")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdfi.resample("m")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_dfs_resample_param_rule_month_param_on_date_max(self):
        a = self.odfs_csv.resample("m", on="date").max()
        b = self.pdf_csv.resample("m", on="date").max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True).iloc[:, 0:2], b.reset_index(drop=True).iloc[:, 0:2],
                           check_dtype=False)
        assert_frame_equal(a.to_pandas().reset_index(drop=True).iloc[:, 6:8], b.reset_index(drop=True).iloc[:, 6:8],
                           check_dtype=False)
        assert_frame_equal(a.to_pandas().reset_index(drop=True).iloc[:, 9:19], b.reset_index(drop=True).iloc[:, 9:19],
                           check_dtype=False)

        with self.assertRaises(TypeError):
            self.odfs_csv.resample("m").max()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("m").max()

        a = self.odfs_csv.resample("m", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = self.pdf_csv.resample("m", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odfs_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("m", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdfi.resample("m", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odfs_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("m")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdfi.resample("m")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_dfs_resample_param_rule_month_param_on_date_mean(self):
        a = self.odfs_csv.resample("m", on="date").mean()
        b = self.pdf_csv.resample("m", on="date").mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        with self.assertRaises(TypeError):
            self.odfs_csv.resample("m").mean()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("m").mean()

        a = self.odfs_csv.resample("m", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = self.pdf_csv.resample("m", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odfs_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("m", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdfi.resample("m", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odfs_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("m")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdfi.resample("m")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_dfs_resample_param_rule_month_param_on_date_min(self):
        a = self.odfs_csv.resample("m", on="date").min()
        b = self.pdf_csv.resample("m", on="date").min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True).iloc[:, 0:2], b.reset_index(drop=True).iloc[:, 0:2],
                           check_dtype=False)
        assert_frame_equal(a.to_pandas().reset_index(drop=True).iloc[:, 6:8], b.reset_index(drop=True).iloc[:, 6:8],
                           check_dtype=False)
        assert_frame_equal(a.to_pandas().reset_index(drop=True).iloc[:, 9:19], b.reset_index(drop=True).iloc[:, 9:19],
                           check_dtype=False)

        with self.assertRaises(TypeError):
            self.odfs_csv.resample("m").min()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("m").min()

        a = self.odfs_csv.resample("m", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = self.pdf_csv.resample("m", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odfs_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("m", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdfi.resample("m", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odfs_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("m")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdfi.resample("m")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_dfs_resample_param_rule_month_param_on_date_std(self):
        a = self.odfs_csv.resample("m", on="date").std()
        b = self.pdf_csv.resample("m", on="date").std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        with self.assertRaises(TypeError):
            self.odfs_csv.resample("m").std()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("m").std()

        a = self.odfs_csv.resample("m", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = self.pdf_csv.resample("m", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odfs_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("m", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdfi.resample("m", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odfs_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("m")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdfi.resample("m")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_dfs_resample_param_rule_month_param_on_date_sum(self):
        a = self.odfs_csv.resample("m", on="date").sum()
        b = self.pdf_csv.resample("m", on="date").sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        with self.assertRaises(TypeError):
            self.odfs_csv.resample("m").sum()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("m").sum()

        a = self.odfs_csv.resample("m", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = self.pdf_csv.resample("m", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odfs_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("m", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdfi.resample("m", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odfs_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("m")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdfi.resample("m")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_dfs_resample_param_rule_month_param_on_date_var(self):
        a = self.odfs_csv.resample("m", on="date").var()
        b = self.pdf_csv.resample("m", on="date").var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        with self.assertRaises(TypeError):
            self.odfs_csv.resample("m").var()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("m").var()

        a = self.odfs_csv.resample("m", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = self.pdf_csv.resample("m", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odfs_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("m", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdfi.resample("m", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odfs_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("m")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdfi.resample("m")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_dfs_resample_param_rule_day_param_on_date_count(self):
        a = self.odfs_csv.resample("d", on="date").count()
        b = self.pdf_csv.resample("d", on="date").count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        self.odfs_csv.resample("3d", on="date").count()
        self.pdf_csv.resample("3d", on="date").count()

        with self.assertRaises(TypeError):
            self.odfs_csv.resample("d").count()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("d").count()

        a = self.odfs_csv.resample("d", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = self.pdf_csv.resample("d", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("d", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdfi.resample("d", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("d")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdfi.resample("d")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_dfs_resample_param_rule_day_param_on_date_max(self):
        a = self.odfs_csv.resample("d", on="date").max()
        b = self.pdf_csv.resample("d", on="date").max()
        assert_frame_equal(a.to_pandas().iloc[:, 0:2], b.iloc[:, 0:2], check_dtype=False)
        assert_frame_equal(a.to_pandas().iloc[:, 6:8], b.iloc[:, 6:8], check_dtype=False)
        assert_frame_equal(a.to_pandas().iloc[:, 9:19], b.iloc[:, 9:19], check_dtype=False)

        with self.assertRaises(TypeError):
            self.odfs_csv.resample("d").max()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("d").max()

        a = self.odfs_csv.resample("d", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = self.pdf_csv.resample("d", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("d", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdfi.resample("d", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("d")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdfi.resample("d")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_dfs_resample_param_rule_day_param_on_date_mean(self):
        a = self.odfs_csv.resample("d", on="date").mean()
        b = self.pdf_csv.resample("d", on="date").mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odfs_csv.resample("d").mean()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("d").mean()

        a = self.odfs_csv.resample("d", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = self.pdf_csv.resample("d", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("d", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdfi.resample("d", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("d")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdfi.resample("d")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_dfs_resample_param_rule_day_param_on_date_min(self):
        a = self.odfs_csv.resample("d", on="date").min()
        b = self.pdf_csv.resample("d", on="date").min()
        assert_frame_equal(a.to_pandas().iloc[:, 0:2], b.iloc[:, 0:2], check_dtype=False)
        assert_frame_equal(a.to_pandas().iloc[:, 6:8], b.iloc[:, 6:8], check_dtype=False)
        assert_frame_equal(a.to_pandas().iloc[:, 9:19], b.iloc[:, 9:19], check_dtype=False)

        with self.assertRaises(TypeError):
            self.odfs_csv.resample("d").min()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("d").min()

        a = self.odfs_csv.resample("d", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = self.pdf_csv.resample("d", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("d", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdfi.resample("d", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("d")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdfi.resample("d")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_dfs_resample_param_rule_day_param_on_date_std(self):
        a = self.odfs_csv.resample("d", on="date").std()
        b = self.pdf_csv.resample("d", on="date").std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odfs_csv.resample("d").std()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("d").std()

        a = self.odfs_csv.resample("d", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = self.pdf_csv.resample("d", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("d", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdfi.resample("d", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("d")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdfi.resample("d")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_dfs_resample_param_rule_day_param_on_date_sum(self):
        a = self.odfs_csv.resample("d", on="date").sum()
        b = self.pdf_csv.resample("d", on="date").sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odfs_csv.resample("d").sum()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("d").sum()

        a = self.odfs_csv.resample("d", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = self.pdf_csv.resample("d", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("d", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdfi.resample("d", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("d")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdfi.resample("d")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_dfs_resample_param_rule_day_param_on_date_var(self):
        a = self.odfs_csv.resample("d", on="date").var()
        b = self.pdf_csv.resample("d", on="date").var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odfs_csv.resample("d").var()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("d").var()

        a = self.odfs_csv.resample("d", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = self.pdf_csv.resample("d", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("d", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdfi.resample("d", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("d")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdfi.resample("d")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_dfs_resample_param_rule_week_param_on_date_count(self):
        a = self.odfs_csv.resample("w", on="date").count()
        b = self.pdf_csv.resample("w", on="date").count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        self.odfs_csv.resample("3w", on="date").count()
        self.pdf_csv.resample("3w", on="date").count()

        with self.assertRaises(TypeError):
            self.odfs_csv.resample("w").count()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("w").count()

        a = self.odfs_csv.resample("w", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = self.pdf_csv.resample("w", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("w", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdfi.resample("w", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("w")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdfi.resample("w")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_dfs_resample_param_rule_week_param_on_date_max(self):
        a = self.odfs_csv.resample("w", on="date").max()
        b = self.pdf_csv.resample("w", on="date").max()
        assert_frame_equal(a.to_pandas().iloc[:, 0:2], b.iloc[:, 0:2], check_dtype=False)
        assert_frame_equal(a.to_pandas().iloc[:, 6:8], b.iloc[:, 6:8], check_dtype=False)
        assert_frame_equal(a.to_pandas().iloc[:, 9:19], b.iloc[:, 9:19], check_dtype=False)

        with self.assertRaises(TypeError):
            self.odfs_csv.resample("w").max()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("w").max()

        a = self.odfs_csv.resample("w", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = self.pdf_csv.resample("w", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("w", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdfi.resample("w", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("w")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdfi.resample("w")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_dfs_resample_param_rule_week_param_on_date_mean(self):
        a = self.odfs_csv.resample("w", on="date").mean()
        b = self.pdf_csv.resample("w", on="date").mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odfs_csv.resample("w").mean()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("w").mean()

        a = self.odfs_csv.resample("w", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = self.pdf_csv.resample("w", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("w", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdfi.resample("w", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("w")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdfi.resample("w")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_dfs_resample_param_rule_week_param_on_date_min(self):
        a = self.odfs_csv.resample("w", on="date").min()
        b = self.pdf_csv.resample("w", on="date").min()
        assert_frame_equal(a.to_pandas().iloc[:, 0:2], b.iloc[:, 0:2], check_dtype=False)
        assert_frame_equal(a.to_pandas().iloc[:, 6:8], b.iloc[:, 6:8], check_dtype=False)
        assert_frame_equal(a.to_pandas().iloc[:, 9:19], b.iloc[:, 9:19], check_dtype=False)

        with self.assertRaises(TypeError):
            self.odfs_csv.resample("w").min()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("w").min()

        a = self.odfs_csv.resample("w", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = self.pdf_csv.resample("w", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("w", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdfi.resample("w", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("w")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdfi.resample("w")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_dfs_resample_param_rule_week_param_on_date_std(self):
        a = self.odfs_csv.resample("w", on="date").std()
        b = self.pdf_csv.resample("w", on="date").std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odfs_csv.resample("w").std()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("w").std()

        a = self.odfs_csv.resample("w", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = self.pdf_csv.resample("w", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("w", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdfi.resample("w", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("w")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdfi.resample("w")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_dfs_resample_param_rule_week_param_on_date_sum(self):
        a = self.odfs_csv.resample("w", on="date").sum()
        b = self.pdf_csv.resample("w", on="date").sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odfs_csv.resample("w").sum()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("w").sum()

        a = self.odfs_csv.resample("w", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = self.pdf_csv.resample("w", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("w", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdfi.resample("w", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("w")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdfi.resample("w")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_dfs_resample_param_rule_week_param_on_date_var(self):
        a = self.odfs_csv.resample("w", on="date").var()
        b = self.pdf_csv.resample("w", on="date").var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odfs_csv.resample("w").var()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("w").var()

        a = self.odfs_csv.resample("w", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = self.pdf_csv.resample("w", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("w", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdfi.resample("w", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("w")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdfi.resample("w")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_dfs_resample_param_rule_hour_param_on_timestamp_count(self):
        a = self.odfs_csv.resample("H", on="timestamp").count()
        b = self.pdf_csv.resample("H", on="timestamp").count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        self.odfs_csv.resample("3H", on="timestamp").count()
        self.pdf_csv.resample("3H", on="timestamp").count()

        with self.assertRaises(TypeError):
            self.odfs_csv.resample("H").count()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("H").count()

        a = self.odfs_csv.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = self.pdf_csv.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdfi.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('timestamp')
        pdfi = self.pdf_csv.set_index('timestamp')
        a = odfi.resample("H")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdfi.resample("H")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_dfs_resample_param_rule_hour_param_on_timestamp_max(self):
        a = self.odfs_csv.resample("H", on="timestamp").max()
        b = self.pdf_csv.resample("H", on="timestamp").max()
        assert_frame_equal(a.to_pandas().iloc[:, 0:2], b.iloc[:, 0:2], check_dtype=False)
        assert_frame_equal(a.to_pandas().iloc[:, 6:8], b.iloc[:, 6:8], check_dtype=False)
        assert_frame_equal(a.to_pandas().iloc[:, 9:19], b.iloc[:, 9:19], check_dtype=False)

        with self.assertRaises(TypeError):
            self.odfs_csv.resample("H").max()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("H").max()

        a = self.odfs_csv.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = self.pdf_csv.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdfi.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('timestamp')
        pdfi = self.pdf_csv.set_index('timestamp')
        a = odfi.resample("H")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdfi.resample("H")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_dfs_resample_param_rule_hour_param_on_timestamp_mean(self):
        a = self.odfs_csv.resample("H", on="timestamp").mean()
        b = self.pdf_csv.resample("H", on="timestamp").mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odfs_csv.resample("H").mean()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("H").mean()

        a = self.odfs_csv.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = self.pdf_csv.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdfi.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('timestamp')
        pdfi = self.pdf_csv.set_index('timestamp')
        a = odfi.resample("H")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdfi.resample("H")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_dfs_resample_param_rule_hour_param_on_timestamp_min(self):
        a = self.odfs_csv.resample("H", on="timestamp").min()
        b = self.pdf_csv.resample("H", on="timestamp").min()
        assert_frame_equal(a.to_pandas().iloc[:, 0:3], b.iloc[:, 0:3], check_dtype=False)
        assert_frame_equal(a.to_pandas().iloc[:, 6:8], b.iloc[:, 6:8], check_dtype=False)
        assert_frame_equal(a.to_pandas().iloc[:, 9:19], b.iloc[:, 9:19], check_dtype=False)

        with self.assertRaises(TypeError):
            self.odfs_csv.resample("H").min()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("H").min()

        a = self.odfs_csv.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = self.pdf_csv.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdfi.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('timestamp')
        pdfi = self.pdf_csv.set_index('timestamp')
        a = odfi.resample("H")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdfi.resample("H")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_dfs_resample_param_rule_hour_param_on_timestamp_std(self):
        a = self.odfs_csv.resample("H", on="timestamp").std()
        b = self.pdf_csv.resample("H", on="timestamp").std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odfs_csv.resample("H").std()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("H").std()

        a = self.odfs_csv.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = self.pdf_csv.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdfi.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('timestamp')
        pdfi = self.pdf_csv.set_index('timestamp')
        a = odfi.resample("H")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdfi.resample("H")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_dfs_resample_param_rule_hour_param_on_timestamp_sum(self):
        a = self.odfs_csv.resample("H", on="timestamp").sum()
        b = self.pdf_csv.resample("H", on="timestamp").sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odfs_csv.resample("H").sum()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("H").sum()

        a = self.odfs_csv.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = self.pdf_csv.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdfi.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('timestamp')
        pdfi = self.pdf_csv.set_index('timestamp')
        a = odfi.resample("H")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdfi.resample("H")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_dfs_resample_param_rule_hour_param_on_timestamp_var(self):
        a = self.odfs_csv.resample("H", on="timestamp").var()
        b = self.pdf_csv.resample("H", on="timestamp").var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odfs_csv.resample("H").var()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("H").var()

        a = self.odfs_csv.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = self.pdf_csv.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdfi.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('timestamp')
        pdfi = self.pdf_csv.set_index('timestamp')
        a = odfi.resample("H")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdfi.resample("H")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_dfs_resample_param_rule_minute_param_on_timestamp_count(self):
        a = self.odfs_csv.resample("T", on="timestamp").count()
        b = self.pdf_csv.resample("T", on="timestamp").count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        self.odfs_csv.resample("3T", on="timestamp").count()
        self.pdf_csv.resample("3T", on="timestamp").count()

        with self.assertRaises(TypeError):
            self.odfs_csv.resample("T").count()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("T").count()

        a = self.odfs_csv.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = self.pdf_csv.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdfi.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('timestamp')
        pdfi = self.pdf_csv.set_index('timestamp')
        a = odfi.resample("T")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdfi.resample("T")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_dfs_resample_param_rule_minute_param_on_timestamp_max(self):
        a = self.odfs_csv.resample("T", on="timestamp").max()
        b = self.pdf_csv.resample("T", on="timestamp").max()
        assert_frame_equal(a.to_pandas().iloc[:, 0:2], b.iloc[:, 0:2], check_dtype=False)
        assert_frame_equal(a.to_pandas().iloc[:, 6:8], b.iloc[:, 6:8], check_dtype=False)
        assert_frame_equal(a.to_pandas().iloc[:, 9:19], b.iloc[:, 9:19], check_dtype=False)

        with self.assertRaises(TypeError):
            self.odfs_csv.resample("T").max()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("T").max()

        a = self.odfs_csv.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = self.pdf_csv.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdfi.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('timestamp')
        pdfi = self.pdf_csv.set_index('timestamp')
        a = odfi.resample("T")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdfi.resample("T")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_dfs_resample_param_rule_minute_param_on_timestamp_mean(self):
        a = self.odfs_csv.resample("T", on="timestamp").mean()
        b = self.pdf_csv.resample("T", on="timestamp").mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odfs_csv.resample("T").mean()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("T").mean()

        a = self.odfs_csv.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = self.pdf_csv.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdfi.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('timestamp')
        pdfi = self.pdf_csv.set_index('timestamp')
        a = odfi.resample("T")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdfi.resample("T")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_dfs_resample_param_rule_minute_param_on_timestamp_min(self):
        a = self.odfs_csv.resample("T", on="timestamp").min()
        b = self.pdf_csv.resample("T", on="timestamp").min()
        assert_frame_equal(a.to_pandas().iloc[:, 0:3], b.iloc[:, 0:3], check_dtype=False)
        assert_frame_equal(a.to_pandas().iloc[:, 6:8], b.iloc[:, 6:8], check_dtype=False)
        assert_frame_equal(a.to_pandas().iloc[:, 9:19], b.iloc[:, 9:19], check_dtype=False)

        with self.assertRaises(TypeError):
            self.odfs_csv.resample("T").min()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("T").min()

        a = self.odfs_csv.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = self.pdf_csv.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdfi.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('timestamp')
        pdfi = self.pdf_csv.set_index('timestamp')
        a = odfi.resample("T")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdfi.resample("T")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_dfs_resample_param_rule_minute_param_on_timestamp_std(self):
        a = self.odfs_csv.resample("T", on="timestamp").std()
        b = self.pdf_csv.resample("T", on="timestamp").std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odfs_csv.resample("T").std()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("T").std()

        a = self.odfs_csv.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = self.pdf_csv.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdfi.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('timestamp')
        pdfi = self.pdf_csv.set_index('timestamp')
        a = odfi.resample("T")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdfi.resample("T")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_dfs_resample_param_rule_minute_param_on_timestamp_sum(self):
        a = self.odfs_csv.resample("T", on="timestamp").sum()
        b = self.pdf_csv.resample("T", on="timestamp").sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odfs_csv.resample("T").sum()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("T").sum()

        a = self.odfs_csv.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = self.pdf_csv.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdfi.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('timestamp')
        pdfi = self.pdf_csv.set_index('timestamp')
        a = odfi.resample("T")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdfi.resample("T")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_dfs_resample_param_rule_minute_param_on_timestamp_var(self):
        a = self.odfs_csv.resample("T", on="timestamp").var()
        b = self.pdf_csv.resample("T", on="timestamp").var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odfs_csv.resample("T").var()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("T").var()

        a = self.odfs_csv.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = self.pdf_csv.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdfi.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('timestamp')
        pdfi = self.pdf_csv.set_index('timestamp')
        a = odfi.resample("T")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdfi.resample("T")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_dfs_resample_param_rule_second_param_on_timestamp_count(self):
        a = self.odfs_csv.resample("S", on="timestamp").count()
        b = self.pdf_csv.resample("S", on="timestamp").count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        self.odfs_csv.resample("3S", on="timestamp").count()
        self.pdf_csv.resample("3S", on="timestamp").count()

        with self.assertRaises(TypeError):
            self.odfs_csv.resample("S").count()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("S").count()

        a = self.odfs_csv.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = self.pdf_csv.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdfi.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('timestamp')
        pdfi = self.pdf_csv.set_index('timestamp')
        a = odfi.resample("S")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdfi.resample("S")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_dfs_resample_param_rule_second_param_on_timestamp_max(self):
        a = self.odfs_csv.resample("S", on="timestamp").max()
        b = self.pdf_csv.resample("S", on="timestamp").max()
        assert_frame_equal(a.to_pandas().iloc[:, 0:2], b.iloc[:, 0:2], check_dtype=False)
        assert_frame_equal(a.to_pandas().iloc[:, 6:8], b.iloc[:, 6:8], check_dtype=False)
        assert_frame_equal(a.to_pandas().iloc[:, 9:19], b.iloc[:, 9:19], check_dtype=False)

        with self.assertRaises(TypeError):
            self.odfs_csv.resample("S").max()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("S").max()

        a = self.odfs_csv.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = self.pdf_csv.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdfi.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('timestamp')
        pdfi = self.pdf_csv.set_index('timestamp')
        a = odfi.resample("S")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdfi.resample("S")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_dfs_resample_param_rule_second_param_on_timestamp_mean(self):
        a = self.odfs_csv.resample("S", on="timestamp").mean()
        b = self.pdf_csv.resample("S", on="timestamp").mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odfs_csv.resample("S").mean()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("S").mean()

        a = self.odfs_csv.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = self.pdf_csv.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdfi.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('timestamp')
        pdfi = self.pdf_csv.set_index('timestamp')
        a = odfi.resample("S")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdfi.resample("S")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_dfs_resample_param_rule_second_param_on_timestamp_min(self):
        a = self.odfs_csv.resample("S", on="timestamp").min()
        b = self.pdf_csv.resample("S", on="timestamp").min()
        assert_frame_equal(a.to_pandas().iloc[:, 0:3], b.iloc[:, 0:3], check_dtype=False)
        assert_frame_equal(a.to_pandas().iloc[:, 6:8], b.iloc[:, 6:8], check_dtype=False)
        assert_frame_equal(a.to_pandas().iloc[:, 9:19], b.iloc[:, 9:19], check_dtype=False)

        with self.assertRaises(TypeError):
            self.odfs_csv.resample("S").min()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("S").min()

        a = self.odfs_csv.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = self.pdf_csv.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdfi.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('timestamp')
        pdfi = self.pdf_csv.set_index('timestamp')
        a = odfi.resample("S")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdfi.resample("S")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_dfs_resample_param_rule_second_param_on_timestamp_std(self):
        a = self.odfs_csv.resample("S", on="timestamp").std()
        b = self.pdf_csv.resample("S", on="timestamp").std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odfs_csv.resample("S").std()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("S").std()

        a = self.odfs_csv.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = self.pdf_csv.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdfi.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('timestamp')
        pdfi = self.pdf_csv.set_index('timestamp')
        a = odfi.resample("S")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdfi.resample("S")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_dfs_resample_param_rule_second_param_on_timestamp_sum(self):
        a = self.odfs_csv.resample("S", on="timestamp").sum()
        b = self.pdf_csv.resample("S", on="timestamp").sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odfs_csv.resample("S").sum()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("S").sum()

        a = self.odfs_csv.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = self.pdf_csv.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdfi.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('timestamp')
        pdfi = self.pdf_csv.set_index('timestamp')
        a = odfi.resample("S")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdfi.resample("S")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_dfs_resample_param_rule_second_param_on_timestamp_var(self):
        a = self.odfs_csv.resample("S", on="timestamp").var()
        b = self.pdf_csv.resample("S", on="timestamp").var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odfs_csv.resample("S").var()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("S").var()

        a = self.odfs_csv.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = self.pdf_csv.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdfi.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odfs_csv.set_index('timestamp')
        pdfi = self.pdf_csv.set_index('timestamp')
        a = odfi.resample("S")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdfi.resample("S")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)


if __name__ == '__main__':
    unittest.main()
