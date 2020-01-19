import unittest
import orca
import os.path as path
from setup.settings import *
from pandas.util.testing import *


def _create_odf_csv(data_left, data_right):
    # call function default_session() to get session object
    s = orca.default_session()

    dolphindb_script = """
    login("admin", "123456")
    dbPath="dfs://testjoinDB"
    if(existsDatabase(dbPath))
        dropDatabase(dbPath)
    tt_left=extractTextSchema('{datal}')
    update tt_left set type="SYMBOL" where name="TRDSTAT"
    tt_right=extractTextSchema('{datar}')
    schema_left = table(50000:0, tt_left.name, tt_left.type)
    schema_right = table(50000:0, tt_right.name, tt_right.type)
    db = database(dbPath, RANGE, 2010.01.04 2011.01.04 2012.01.04 2013.01.04 2014.01.04 2015.01.04  2016.01.04)
    tb_left = db.createPartitionedTable(schema_left, `tb_left, `date)
    db.loadTextEx(`tb_left, `date, '{datal}' ,, tt_left)
    tb_right = db.createPartitionedTable(schema_right, `tb_right, `date)
    db.loadTextEx(`tb_right, `date, '{datar}' ,, tt_right)""".format(datal=data_left, datar=data_right)
    s.run(dolphindb_script)


class Csv:
    odf_csv_left = None
    odf_csv_right = None
    odfs_csv_left = None
    odfs_csv_right = None
    pdf_csv_left = None
    pdf_csv_right = None


class DfsJoinTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # configure data directory
        DATA_DIR = path.abspath(path.join(__file__, "../setup/data"))
        left_fileName = 'test_join_left_table.csv'
        right_fileName = 'test_join_right_table.csv'
        data_left = os.path.join(DATA_DIR, left_fileName)
        data_left = data_left.replace('\\', '/')
        data_right = os.path.join(DATA_DIR, right_fileName)
        data_right = data_right.replace('\\', '/')
        dfsDatabase = "dfs://testjoinDB"

        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")
        _create_odf_csv(data_left, data_right)

        # import
        Csv.odf_csv_left = orca.read_csv(data_left, dtype={"TRDSTAT": "SYMBOL"})
        Csv.odfs_csv_left = orca.read_table(dfsDatabase, 'tb_left')
        Csv.pdf_csv_left = pd.read_csv(data_left, parse_dates=[1])

        Csv.odf_csv_right = orca.read_csv(data_right, dtype={"TRDSTAT": "SYMBOL"})
        Csv.odfs_csv_right = orca.read_table(dfsDatabase, 'tb_right')
        Csv.pdf_csv_right = pd.read_csv(data_right, parse_dates=[0])

    @property
    def odf_csv_left(self):
        return Csv.odf_csv_left

    @property
    def odf_csv_right(self):
        return Csv.odf_csv_right

    @property
    def odfs_csv_left(self):
        return Csv.odfs_csv_left

    @property
    def odfs_csv_right(self):
        return Csv.odfs_csv_right

    @property
    def pdf_csv_left(self):
        return Csv.pdf_csv_left

    @property
    def pdf_csv_right(self):
        return Csv.pdf_csv_right

    @property
    def odf_csv_left_index(self):
        return Csv.odf_csv_left.set_index("date")

    @property
    def odf_csv_right_index(self):
        return Csv.odf_csv_right.set_index("date")

    @property
    def odfs_csv_left_index(self):
        return Csv.odfs_csv_left.set_index("date")

    @property
    def odfs_csv_right_index(self):
        return Csv.odfs_csv_right.set_index("date")

    @property
    def pdf_csv_left_index(self):
        return Csv.pdf_csv_left.set_index("date")

    @property
    def pdf_csv_right_index(self):
        return Csv.pdf_csv_right.set_index("date")

    def test_join_from_dfs_param_lsuffix_paran_rsuffix(self):
        # TODO： NOT IMPLEMENTED
        pass
        # odfs_join = self.odfs_csv_left.join(self.odfs_csv_right, lsuffix='_caller', rsuffix='_other')
        # pdf_join = self.pdf_csv_left.join(self.pdf_csv_right, lsuffix='_caller', rsuffix='_other')
        # assert_frame_equal(odfs_join.to_pandas(), pdf_join, check_dtype=False)

    def test_join_from_dfs_param_on(self):
        odfsl = self.odfs_csv_left
        odfsr = self.odfs_csv_right
        odfsr.set_index("TICKER", inplace=True)
        pdfl = self.pdf_csv_left
        pdfr = self.pdf_csv_right
        pdfr.set_index("TICKER", inplace=True)
        # TODO: When joining two partitioned tables, the partitioning column(s) must be the same as or a subset of the joining column(s).
        #  Here,the default joining column is default rangeIndex but not the partition column date
        # odfs_join = odfsl.join(odfsr, on='TICKER', lsuffix='_caller', rsuffix='_other')
        # pdf_join = pdfl.join(pdfr, on='TICKER', lsuffix='_caller', rsuffix='_other')
        # assert_frame_equal(odfs_join.to_pandas(), pdf_join, check_dtype=False)

    def test_join_from_dfs_param_how(self):
        # TODO: JOIN ON 2 PARTITIONED TABLES
        pass
        # by default, how = left

        # TODO: When joining two partitioned tables, the partitioning column(s) must be the same as or a subset of the joining column(s).
        #  Here,the default joining column is default rangeIndex but not the partition column date
        # odfs_join = self.odfs_csv_left.join(self.odfs_csv_right, how="right", lsuffix='_caller', rsuffix='_other')
        # pdf_join = self.pdf_csv_left.join(self.pdf_csv_right, how="right", lsuffix='_caller', rsuffix='_other')
        # assert_frame_equal(odfs_join.to_pandas(), pdf_join, check_dtype=False)

        # # how = right
        # odfs_join = self.odfs_csv_left.join(self.odfs_csv_right_index, on="date", how="right", lsuffix='_caller', rsuffix='_other')
        # pdf_join = self.pdf_csv_left.join(self.pdf_csv_right_index, on="date", how="right", lsuffix='_caller', rsuffix='_other')
        # assert_frame_equal(odfs_join.to_pandas(), pdf_join, check_dtype=False)
        #
        # # how = inner
        # odfs_join = self.odfs_csv_left.join(self.odfs_csv_right, how="inner", lsuffix='_caller', rsuffix='_other')
        # pdf_join = self.pdf_csv_left.join(self.pdf_csv_right, how="inner", lsuffix='_caller', rsuffix='_other')
        # assert_frame_equal(odfs_join.to_pandas(), pdf_join, check_dtype=False)
        #
        # # how = outer
        # odfs_join = self.odfs_csv_left.join(self.odfs_csv_right, how="outer", lsuffix='_caller', rsuffix='_other')
        # pdf_join = self.pdf_csv_left.join(self.pdf_csv_right, how="outer", lsuffix='_caller', rsuffix='_other')
        # assert_frame_equal(odfs_join.to_pandas(), pdf_join, check_dtype=False)

    def test_join_from_dfs_param_sort(self):
        # TODO： NOT IMPLEMENTED
        pass
        # odfs_join = self.odfs_csv_left.join(self.odfs_csv_right, lsuffix='_caller', rsuffix='_other', sort=True)
        # pdf_join = self.pdf_csv_left.join(self.pdf_csv_right, lsuffix='_caller', rsuffix='_other', sort=True)
        # assert_frame_equal(odfs_join.to_pandas(), pdf_join, check_dtype=False)

    def test_join_from_dfs_index_param_lsuffix_paran_rsuffix(self):
        odfs_join = self.odfs_csv_left_index.join(self.odfs_csv_right_index, lsuffix='_caller', rsuffix='_other')
        pdf_join = self.pdf_csv_left_index.join(self.pdf_csv_right_index, lsuffix='_caller', rsuffix='_other')
        assert_frame_equal(odfs_join.to_pandas().iloc[:, 0:11], pdf_join.iloc[:, 0:11], check_dtype=False)
        assert_frame_equal(odfs_join.to_pandas().iloc[:, 12:], pdf_join.iloc[:, 12:], check_dtype=False)

    def test_join_from_dfs_index_param_on(self):
        odfs_join = self.odfs_csv_left.join(self.odfs_csv_right_index, on='date', lsuffix='_caller', rsuffix='_other')
        pdf_join = self.pdf_csv_left.join(self.pdf_csv_right_index, on='date', lsuffix='_caller', rsuffix='_other')

        assert_frame_equal(odfs_join.to_pandas().iloc[:, 0:12], pdf_join.iloc[:, 0:12], check_dtype=False)
        assert_frame_equal(odfs_join.to_pandas().iloc[:, 13:], pdf_join.iloc[:, 13:], check_dtype=False)

    def test_join_from_dfs_index_param_lsuffix_paran_how(self):
        # by default, how = left

        # how = right
        odfs_join = self.odfs_csv_left_index.join(self.odfs_csv_right_index, how="right", lsuffix='_caller', rsuffix='_other')
        pdf_join = self.pdf_csv_left_index.join(self.pdf_csv_right_index, how="right", lsuffix='_caller', rsuffix='_other')
        pdf_join.loc[:, 'TICKER_caller'].fillna("", inplace=True)
        pdf_join.loc[:, 'TRDSTAT'].fillna("", inplace=True)
        assert_frame_equal(odfs_join.sort_index().to_pandas(), pdf_join.sort_index(), check_dtype=False)

        # how = inner
        odfs_join = self.odfs_csv_left_index.join(self.odfs_csv_right_index, how="inner", lsuffix='_caller', rsuffix='_other')
        pdf_join = self.pdf_csv_left_index.join(self.pdf_csv_right_index, how="inner", lsuffix='_caller', rsuffix='_other')
        assert_frame_equal(odfs_join.to_pandas(), pdf_join, check_dtype=False)

        # how = outer
        odfs_join = self.odfs_csv_left_index.join(self.odfs_csv_right_index, how="outer", lsuffix='_caller', rsuffix='_other')
        pdf_join = self.pdf_csv_left_index.join(self.pdf_csv_right_index, how="outer", lsuffix='_caller', rsuffix='_other')
        pdf_join.loc[:, 'TICKER_caller'].fillna("", inplace=True)
        pdf_join.loc[:, 'TRDSTAT'].fillna("", inplace=True)
        # TODO:orca partition does not support orderly index
        # assert_frame_equal(odfs_join.to_pandas().reset_index(drop=True), pdf_join.reset_index(drop=True), check_dtype=False)

    def test_join_from_dfs_index_param_lsuffix_paran_sort(self):
        # TODO： NOT IMPLEMENTED
        pass
        # odfs_join = self.odfs_csv_left_index.join(self.odfs_csv_right_index, lsuffix='_caller', rsuffix='_other', sort=True)
        # pdf_join = self.pdf_csv_left_index.join(self.pdf_csv_right_index, lsuffix='_caller', rsuffix='_other', sort=True)
        # assert_frame_equal(odfs_join.to_pandas(), pdf_join, check_dtype=False)


if __name__ == '__main__':
    unittest.main()
