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
    disk = WORK_DIR + "onDiskDB"
    diskPRange = WORK_DIR + "onDiskPartitionedRangeDB"
    diskPValue = WORK_DIR + "onDiskPartitionedValueDB"
    dfsRange = "dfs://RangeDB"
    dfsValue = "dfs://ValueDB"


class TBNames:
    shared = "tshared"
    streamShared = "tStreamShared"
    diskTB = "tb1"
    dfsTB = "tb1"


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
    if(existsDatabase(dbPath))
        dropDatabase(dbPath)
        // rmdir(dbPath, true)
    saveTable(dbPath, t, `{disktb})
    
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


class DataFrameTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # configure data directory
        DATA_DIR = path.abspath(path.join(__file__, "../setup/data"))
        fileName = 'testTables.csv'
        data = os.path.join(DATA_DIR, fileName)
        data = data.replace('\\', '/')

        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

        # odf from import
        Csv.odf_csv = orca.read_csv(data, dtype={"tbool": np.bool, "tchar": np.int8, "tshort": np.int16, "tlong": np.int64,
                                                 "tfloat": np.float32})

        # pdf from import
        Csv.pdf_csv = pd.read_csv(data, dtype={"tbool": np.bool, "tchar": np.int8, "tshort": np.int16, "tlong": np.int64,
                                                 "tfloat": np.float32})

        _create_tables(data)

    @property
    def pdf_csv(self):
        return Csv.pdf_csv

    @property
    def odf_csv(self):
        return Csv.odf_csv

    @property
    def pdf(self):
        return pd.DataFrame({
            'a': [1, 2, 3, 4, 5, 6, 7, 8, 9],
            'b': [4, 5, 6, 3, 2, 1, 0, 0, 0],
        }, index=[0, 1, 3, 5, 6, 8, 9, 9, 9])

    @property
    def odf(self):
        return orca.DataFrame(self.pdf)

    def test_dataframe_combining_joining_merging_append_in_memory_all_types(self):
        n = 10  # note that n should be a multiple of 10
        re = n / 10
        pdf1 = pd.DataFrame({'id': np.arange(1, n + 1, 1, dtype='int32'),
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
        n = 20  # note that n should be a multiple of 10
        re = n / 10
        pdf2 = pd.DataFrame({'id': np.arange(1, n + 1, 1, dtype='int32'),
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

        odf1 = orca.DataFrame(pdf1)
        odf2 = orca.DataFrame(pdf2)
        assert_frame_equal(odf1.append(odf2).to_pandas(), pdf1.append(pdf2))

    def test_dataframe_Combining_joining_merging_append_in_memory(self):
        pdf = pd.DataFrame([[1, 2], [3, 4]], columns=list('AB'))
        pdf2 = pd.DataFrame([[5, 6], [7, 8]], columns=list('AB'))

        odf = orca.DataFrame([[1, 2], [3, 4]], columns=list('AB'))
        odf2 = orca.DataFrame([[5, 6], [7, 8]], columns=list('AB'))

        assert_frame_equal(pdf.append(pdf2), odf.append(odf2).to_pandas())
        assert_frame_equal(pdf.append(pdf2, ignore_index=True), odf.append(odf2, ignore_index=True).to_pandas())
        assert_frame_equal(pdf.append(pdf2, sort=True), odf.append(odf2, sort=True).to_pandas())
        odf.append(odf2, inplace=True)
        assert_frame_equal(pdf.append(pdf2), odf.to_pandas())

    def test_dataframe_Combining_joining_merging_append_on_disk(self):
        odftemp = self.odf_csv
        odfd = orca.read_table(DBNames.disk, TBNames.diskTB)
        assert_frame_equal(odfd.to_pandas(), odftemp.to_pandas())

        odfd.append(odftemp, inplace=True)
        odf_compare = odftemp.append(odftemp)
        assert_frame_equal(odfd.to_pandas(), odf_compare.to_pandas())

        # reload didk table
        odfd_rs = orca.read_table(DBNames.disk, TBNames.diskTB)
        assert_frame_equal(odfd_rs.to_pandas(), odftemp.to_pandas())

    def test_dataframe_Combining_joining_merging_append_on_disk_partitioned(self):
        odftemp = self.odf_csv
        odfd = orca.read_table(DBNames.diskPRange, TBNames.diskTB)
        assert_frame_equal(odfd.to_pandas(), odftemp.to_pandas())

        odfd.append(odftemp, inplace=True)
        odf_compare = odftemp.append(odftemp)

        # reload didk table
        odfd_rs = orca.read_table(DBNames.diskPRange, TBNames.diskTB)
        assert_frame_equal(odfd_rs.to_pandas().sort_values("id").reset_index(drop=True), odf_compare.to_pandas().sort_values("id").reset_index(drop=True))

    def test_dataframe_Combining_joining_merging_append_dfs(self):
        odftemp = self.odf_csv
        odfs = orca.read_table(DBNames.dfsRange, TBNames.dfsTB)
        assert_frame_equal(odfs.to_pandas(), odftemp.to_pandas())

        odfs.append(odftemp, inplace=True)
        odf_compare = odftemp.append(odftemp)

        # reload dfs table
        odfs_rs = orca.read_table(DBNames.dfsRange, TBNames.dfsTB)
        assert_frame_equal(odfs_rs.to_pandas().sort_values("id").reset_index(drop=True), odf_compare.to_pandas().sort_values("id").reset_index(drop=True))

    def test_dataframe_Combining_joining_merging_append_shared_stream_table(self):
        odftemp = self.odf_csv
        stream_tb = orca.read_shared_table(TBNames.streamShared)
        assert_frame_equal(stream_tb.to_pandas(), odftemp.to_pandas())

        stream_tb.append(odftemp, inplace=True)
        odf_compare = odftemp.append(odftemp)

        # reload stream table
        stream_tb_rs = orca.read_shared_table(TBNames.streamShared)
        assert_frame_equal(stream_tb_rs.to_pandas(), odf_compare.to_pandas().reset_index(drop=True))


if __name__ == '__main__':
    unittest.main()
