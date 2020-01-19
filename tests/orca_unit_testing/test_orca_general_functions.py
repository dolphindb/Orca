import unittest
import orca
import os.path as path
from setup.settings import *
from pandas.util.testing import *

WORK_DIR = WORK_DIR.replace('\\', '/')


class Csv:
    pdf_csv = None
    odf_csv = None


class DBNames:
    disk = WORK_DIR + "onDiskUnpartitionedDB"
    diskPRange = WORK_DIR + "onDiskPartitionedRangeDB"
    diskPValue = WORK_DIR + "onDiskPartitionedValueDB"
    dfsRange = "dfs://RangeDB"
    dfsValue = "dfs://ValueDB"


class TBNames:
    shared = "tshared"
    streamShared = "tStreamShared"
    diskTB = "tb1"
    dfsTB = "tb1"


def _clear(dbName):
    s = orca.default_session()
    s.run("""
    login("admin", "123456")
    dbPath='{dir}'
    if(exists(dbPath))
        dropDatabase(dbPath)
    """.format(dir=dbName))


def _create_tables(DATA_DIR):
    s = orca.default_session()

    dolphindb_script = """
    login("admin", "123456")
    names=`id`date`month`time`minute`second`datetime`timestamp`nanotime`nanotimestamp`tstring`tsymbol`tbool`tchar`tshort`tint`tlong`tfloat`tdouble
    types=`INT`DATE`DATE`TIME`SECOND`SECOND`DATETIME`TIMESTAMP`NANOTIME`NANOTIMESTAMP`STRING`SYMBOL`BOOL`CHAR`SHORT`INT`LONG`FLOAT`DOUBLE
    schema=table(names as name, types as typeString)

    // in-memory
    t=loadText('{datadir}',,schema)

    // shared
    share t as {shared}

    //stream shared
    share streamTable(100:0, names, types) as {streamShared}
    {streamShared}.append!(t)

    //disk
    dbPath='{disk}'
    if(exists(dbPath))
        dropDatabase(dbPath)
    //saveTable(dbPath, t, `{disktb})

    //disk range partition
    dbPath='{diskPR}'
    if(existsDatabase(dbPath))
        dropDatabase(dbPath)
    db=database(dbPath, RANGE, 1 21 41 61 81 101)
    tb=db.createPartitionedTable(t,`{disktb},`id)
    tb.append!(t)

    //disk value partition
    dbPath='{diskPV}'
    if(existsDatabase(dbPath))
        dropDatabase(dbPath)
    db=database(dbPath, VALUE, `A`B`C`D`E`F`G)
    tb=db.createPartitionedTable(t,`{disktb},`tstring)
    tb.append!(t)

    //dfs range partition
    dbPath='{dfsR}'
    if(existsDatabase(dbPath))
        dropDatabase(dbPath)
    db=database(dbPath, RANGE, 1 21 41 61 81 101)
    tb=db.createPartitionedTable(t,`{dfstb},`id)
    tb.append!(t)

    //dfs value partition
    dbPath='{dfsV}'
    if(existsDatabase(dbPath))
        dropDatabase(dbPath)
    db=database(dbPath, VALUE, `A`B`C`D`E`F`G)
    tb=db.createPartitionedTable(t,`{dfstb},`tstring)
    tb.append!(t)""".format(datadir=DATA_DIR, shared=TBNames.shared, streamShared=TBNames.streamShared,
                            disk=DBNames.disk, diskPR=DBNames.diskPRange, diskPV=DBNames.diskPValue,
                            dfsR=DBNames.dfsRange, dfsV=DBNames.dfsValue,
                            disktb=TBNames.diskTB, dfstb=TBNames.dfsTB)
    s.run(dolphindb_script)


class GeneralFunctionsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # configure data directory
        DATA_DIR = path.abspath(path.join(__file__, "../setup/data"))
        DATA_DIR = DATA_DIR.replace('\\', '/')
        fileName = 'testTables.csv'
        data = os.path.join(DATA_DIR, fileName)
        data = data.replace('\\', '/')

        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

        Csv.odf_csv = orca.read_csv(data, dtype={"date": 'DATE', "tstring": "STRING", "tsymbol": "SYMBOL",
                                                 "tbool": "BOOL", "tchar": np.int8, "tshort": np.int16,
                                                 "tlong": np.int64, "tfloat": np.float32})

        Csv.pdf_csv = pd.read_csv(data, parse_dates=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                                  dtype={"id": np.int32, "tbool": np.bool, "tchar": np.int8, "tshort": np.int16,
                                         "tint": np.int32, "tfloat": np.float32})
        _create_tables(data)

    @property
    def pdf_csv(self):
        return Csv.pdf_csv

    @property
    def odf_csv(self):
        return Csv.odf_csv

    @property
    def odf_disk(self):
        return orca.read_table(DBNames.disk, 'tb1')

    @property
    def odf_disk_partitioned_range(self):
        return orca.read_table(DBNames.diskPRange, 'tb1')

    @property
    def odf_disk_partitioned_value(self):
        return orca.read_table(DBNames.diskPValue, 'tb1')

    @property
    def odfs_range(self):
        return orca.read_table(DBNames.dfsRange, 'tb1')

    @property
    def odfs_value(self):
        return orca.read_table(DBNames.dfsValue, 'tb1')

    @property
    def pdf(self):
        return pd.DataFrame({
            'a': [1, 2, 3, 4, 5, 6, 7, 8, 9],
            'b': [4, 5, 6, 3, 2, 1, 0, 0, 0],
        }, index=[0, 1, 3, 5, 6, 8, 9, 9, 9])

    @property
    def odf(self):
        return orca.DataFrame(self.pdf)

    def test_orca_to_datetime(self):
        lt = ['3/11/2000', '3/12/2000', '3/13/2000'] * 100
        assert_index_equal(orca.to_datetime(lt), pd.to_datetime(lt))

        # ps = pd.Series(lt)
        # os = orca.Series(ps)
        # self.assertEqual(pd.to_datetime(ps, infer_datetime_format=True),
        #                  orca.to_datetime(os, infer_datetime_format=True))

    def test_orca_concat_series(self):
        s1 = pd.Series(['a', 'b'])
        s2 = pd.Series(['c', 'd'])
        o1 = orca.Series(['a', 'b'])
        o2 = orca.Series(['c', 'd'])
        assert_series_equal(pd.concat([s1, s2]), orca.concat([o1, o2]).to_pandas())
        assert_series_equal(pd.concat([s1, s2], ignore_index=True),
                            orca.concat([o1, o2], ignore_index=True).to_pandas())

    def test_orca_concat_dataframe(self):
        pdf1 = pd.DataFrame([['a', 1], ['b', 2]], columns=['letter', 'number'])
        pdf2 = pd.DataFrame([['c', 3], ['d', 4]], columns=['letter', 'number'])
        odf1 = orca.DataFrame([['a', 1], ['b', 2]], columns=['letter', 'number'])
        odf2 = orca.DataFrame([['c', 3], ['d', 4]], columns=['letter', 'number'])
        assert_frame_equal(pd.concat([pdf1, pdf2]), orca.concat([odf1, odf2]).to_pandas())
        # assert_frame_equal(pd.concat([pdf1, pdf1]), orca.concat([odf1, odf1]).to_pandas())

        assert_frame_equal(pd.concat([pdf1, pdf2], join="inner"), orca.concat([odf1, odf2], join="inner").to_pandas())
        assert_frame_equal(pd.concat([pdf1, pdf2], ignore_index=True),
                           orca.concat([odf1, odf2], ignore_index=True).to_pandas())

        pdf1 = pd.DataFrame([[3, 1], [6, 2]], columns=['letter', 'number'])
        odf1 = orca.DataFrame([[3, 1], [6, 2]], columns=['letter', 'number'])
        pdf3 = pd.DataFrame([[100, 3, 16], [90, 4, 7]], columns=['letter', 'number', 'animal'])
        odf3 = orca.DataFrame([[100, 3, 16], [90, 4, 7]], columns=['letter', 'number', 'animal'])
        assert_frame_equal(pd.concat([pdf1, pdf3], join="inner"), orca.concat([odf1, odf3], join="inner").to_pandas())
        assert_frame_equal(pd.concat([pdf1, pdf3], join="outer", sort=False),
                           orca.concat([odf1, odf3], join="outer", sort=False).to_pandas())
        assert_frame_equal(pd.concat([pdf1, pdf3], ignore_index=True, sort=False),
                           orca.concat([odf1, odf3], ignore_index=True, sort=False).to_pandas())

        tuples = [('cobra', 'mark i'), ('cobra', 'mark ii'), ('sidewinder', 'mark i'), ('sidewinder', 'mark ii'),
                  ('viper', 'mark ii'), ('viper', 'mark iii')]
        index = pd.MultiIndex.from_tuples(tuples)
        values = [[12, 2], [0, 4], [10, 20], [1, 4], [7, 1], [16, 36]]
        pdf = pd.DataFrame(values, columns=['max_speed', 'shield'], index=index)

        index = orca.MultiIndex.from_tuples(tuples)
        odf = orca.DataFrame(values, columns=['max_speed', 'shield'], index=index)
        assert_frame_equal(pd.concat([pdf, pdf1], ignore_index=True, sort=False),
                           orca.concat([odf, odf1], ignore_index=True, sort=False).to_pandas())

    def test_orca_read_shared_table(self):
        odf = orca.read_shared_table(TBNames.shared)
        assert_frame_equal(odf.to_pandas(), self.odf_csv.to_pandas())
        orca.default_session().run("undef(`{sh},SHARED)".format(sh=TBNames.shared))

    def test_orca_read_shared_streamtable(self):
        odf = orca.read_shared_table(TBNames.streamShared)
        assert_frame_equal(odf.to_pandas(), self.odf_csv.to_pandas())
        orca.default_session().run("undef(`{sh},SHARED)".format(sh=TBNames.streamShared))

    def test_orca_save_table_disk(self):
        orca.save_table(DBNames.disk, TBNames.diskTB, self.odf)
        odf_disk = orca.read_table(DBNames.disk, TBNames.diskTB)
        # index will be reset
        assert_frame_equal(self.pdf.reset_index(drop=True), odf_disk.to_pandas())
        _clear(DBNames.disk)

    def test_orca_save_table_disk_patition(self):
        odf = self.odf_csv
        # range
        orca.save_table(DBNames.diskPRange, TBNames.diskTB, self.odf_csv)
        x = orca.read_table(DBNames.diskPRange, TBNames.diskTB)
        assert_frame_equal(odf.to_pandas(), x.to_pandas())
        _clear(DBNames.diskPRange)
        # value
        orca.save_table(DBNames.diskPValue, TBNames.diskTB, self.odf_csv)
        x = orca.read_table(DBNames.diskPValue, TBNames.diskTB)
        assert_frame_equal(odf.to_pandas(), x.to_pandas())
        _clear(DBNames.diskPValue)

    def test_orca_save_table_dfs(self):
        odf = self.odf_csv
        # range
        orca.save_table(DBNames.dfsRange, TBNames.dfsTB, self.odf_csv)
        x = orca.read_table(DBNames.dfsRange, TBNames.dfsTB)
        assert_frame_equal(odf.append(odf).to_pandas().sort_values("id").reset_index(drop=True),
                           x.to_pandas().sort_values("id").reset_index(drop=True), check_index_type=False)
        # value
        orca.save_table(DBNames.dfsValue, TBNames.dfsTB, self.odf_csv)
        x = orca.read_table(DBNames.dfsValue, TBNames.dfsTB)
        assert_frame_equal(odf.append(odf).to_pandas().sort_values("id").reset_index(drop=True),
                           x.to_pandas().sort_values("id").reset_index(drop=True), check_index_type=False)


if __name__ == '__main__':
    unittest.main()
