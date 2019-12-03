import unittest
import orca
import os.path as path
from setup.settings import *
from pandas.util.testing import *


class Csv:
    pdf_csv = None
    odf_csv = None
    odf_disk = None


def _odf_disk_unpartitioned(data):
    s = orca.default_session()

    # import a csv file to a dfs table
    dolphindb_script = """
    login("admin", "123456")
    dbPath='{WORK_DIR}'+'testOnDiskunpartitionedDB'
    if(existsDatabase(dbPath))
        dropDatabase(dbPath)
    tt=extractTextSchema('{data}')
    update tt set type='FLOAT' where name in ['SHROUT', 'DLSTCD', 'DLPRC', 'VOL']
    update tt set type='SYMBOL' where name='TRDSTAT'
    schema = table(500:0, tt.name, tt.type)
    db = database(dbPath)
    USPrice = db.createTable(schema, `USPrices, `date)
    db.loadTextEx(`USPrices,`date, '{data}' ,, tt)""".format(WORK_DIR=WORK_DIR, data=data)
    s.run(dolphindb_script)
    return orca.read_table(WORK_DIR + 'testOnDiskunpartitionedDB', 'USPrices')


def _create_odf_disk(data):
    # call function default_session() to get session object
    s = orca.default_session()

    # import a csv file to a dfs table
    dolphindb_script = """
    login("admin", "123456")
    dbPath='{WORK_DIR}'+'testOnDiskDB'
    if(existsDatabase(dbPath))
        dropDatabase(dbPath)
    tt=extractTextSchema('{data}')
    update tt set type='FLOAT' where name in ['SHROUT', 'DLSTCD', 'DLPRC', 'VOL']
    update tt set type='SYMBOL' where name='TRDSTAT'
    schema = table(500:0, tt.name, tt.type)
    db = database(dbPath, RANGE, 2010.01.04 2011.01.04 2012.01.04 2013.01.04 2014.01.04 2015.01.04  2016.01.04)
    USPrice = db.createPartitionedTable(schema, `USPrices, `date)
    db.loadTextEx(`USPrices,`date, '{data}' ,, tt)""".format(WORK_DIR=WORK_DIR, data=data)
    s.run(dolphindb_script)
    return orca.read_table(WORK_DIR + 'testOnDiskDB', 'USPrices')


def _create_odf_dfs(data):
    # call function default_session() to get session object
    s = orca.default_session()

    # import a csv file to a dfs table
    dolphindb_script = """
    login("admin", "123456")
    dbPath="dfs://USPricesDB"
    if(existsDatabase(dbPath))
        dropDatabase(dbPath)
    tt=extractTextSchema('{data}')
    update tt set type='FLOAT' where name in ['SHROUT', 'DLSTCD', 'DLPRC', 'VOL']
    update tt set type='SYMBOL' where name='TRDSTAT'
    schema = table(500:0, tt.name, tt.type)
    db = database(dbPath, RANGE, 2010.01.04 2011.01.04 2012.01.04 2013.01.04 2014.01.04 2015.01.04  2016.01.04)
    USPrice = db.createPartitionedTable(schema, `USPrices, `date)
    db.loadTextEx(`USPrices,`date, '{data}' ,, tt)""".format(data=data)
    s.run(dolphindb_script)
    return orca.read_table("dfs://USPricesDB", 'USPrices')


class DataFrameTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # configure data directory
        DATA_DIR = path.abspath(path.join(__file__, "../setup/data"))
        fileName = 'USPricesSample.csv'
        data = os.path.join(DATA_DIR, fileName)
        data = data.replace('\\', '/')

        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

        # odf from import
        Csv.odf_csv = orca.read_csv(data, dtype={"PERMNO": np.int32, "date": 'DATE', "TRDSTAT": 'SYMBOL',
                                                 "DLSTCD": np.float32,
                                                 "DLPRC": np.float32, "VOL": np.float32, "SHROUT": np.float32})
        # pdf from import
        Csv.pdf_csv = pd.read_csv(data, parse_dates=[1],
                                  dtype={"PERMNO": np.int32, "SHRCD": np.int32, "HEXCD": np.int32, "DLSTCD": np.float32,
                                         "DLPRC": np.float32, "VOL": np.float32, "SHROUT": np.float32})

        Csv.odf_disk = _create_odf_disk(data)
        Csv.odf_dfs = _create_odf_dfs(data)
        # Csv.odf_npdisk = _odf_disk_unpartitioned(data)

    @property
    def odf_dfs(self):
        return Csv.odf_dfs

    @property
    def odf_npdisk(self):
        return Csv.odf_npdisk

    @property
    def odf_disk(self):
        return Csv.odf_disk

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

    def test_series_concat(self):
        s1 = pd.Series(['a', 'b'])
        s2 = pd.Series(['c', 'd'])
        o1 = orca.Series(['a', 'b'])
        o2 = orca.Series(['c', 'd'])
        assert_series_equal(pd.concat([s1, s2]), orca.concat([o1, o2]).to_pandas())
        assert_series_equal(pd.concat([s1, s2], ignore_index=True),
                            orca.concat([o1, o2], ignore_index=True).to_pandas())

    def test_dataframe_concat(self):
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

    def test_save_table_in_memory_disk(self):
        # y = self.odf_npdisk
        odf = orca.DataFrame({
            'a': [1, 2, 3, 4, 5, 6, 7, 8, 9],
            'b': [4, 5, 6, 3, 2, 1, 0, 0, 0],
        }, index=[0, 1, 3, 5, 6, 8, 9, 9, 9])
        orca.default_session().run(f"db = database('{WORK_DIR}imd');")
        orca.save_table(WORK_DIR + "imd", "imdb", odf)
        x = orca.read_table(WORK_DIR + "imd", "imdb")
        # print(x)
        # print(self.pdf)
        # index will be reset
        assert_frame_equal(self.pdf.reset_index(drop=True), x.to_pandas())

    def test_save_table_patition_disk(self):
        orca.default_session().run(
            f"db = database('{WORK_DIR}padisk', RANGE, 2010.01.04 2011.01.04 2012.01.04 2013.01.04 2014.01.04 2015.01.04  2016.01.04)")
        odf = self.odf_disk
        orca.save_table(WORK_DIR + "padisk", "tdisk", self.odf_disk)
        x = orca.read_table(WORK_DIR + "padisk", "tdisk")
        # print(len(x))
        assert_frame_equal(odf.to_pandas(), x.to_pandas())
        orca.save_table(WORK_DIR + "padisk", "tdisk", self.odf_disk)
        x = orca.read_table(WORK_DIR + "padisk", "tdisk")
        # print(len(x))

    def test_save_table_patition_dfs(self):
        odf = self.odf_disk
        # print(len(odf))
        orca.save_table('dfs://USPricesDB', "USPrices", odf)

        x = orca.read_table('dfs://USPricesDB', "USPrices")
        # orca.save_table('dfs://USPtestDB', "USPrices", self.odf_disk)
        # print(len(x))
        # assert_frame_equal(odf, odf.to_pandas(), x.to_pandas())

    # def test_read_csv(self):
    #    orca.default_session().run(f"n=1000000;ID=rand(100, n);dates=2017.08.07..2017.08.11;date=rand(dates, n);"
    #                               f"x=rand(10.0, n);t=table(ID, date, x); saveText(t, '{WORK_DIR}tn.txt');"
    #                               f"db = database('{WORK_DIR}rangedb', RANGE, 0 50 100)")
    #    x = orca.read_csv(db_handle=f"{WORK_DIR}rangedb",table_name="pt",
    #                      partition_columns="ID", path=f"{WORK_DIR}tn.txt")

    def test_to_csv(self):
        df = orca.DataFrame(
            {'name': ['Raphael', 'Donatello'], 'mask': ['red', 'purple'], 'weapon': ['sai', 'bo staff']})
        df.to_csv(path_or_buf=f"{WORK_DIR}tocsv.csv")
    #    x = orca.read_csv(path = f"{WORK_DIR}tocsv.csv")
    #    print(x)


if __name__ == '__main__':
    unittest.main()
