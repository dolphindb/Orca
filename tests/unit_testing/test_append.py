import unittest
import orca
import os.path as path
from setup.settings import *
from pandas.util.testing import *


class Csv:
    pdf_csv = None
    odf_csv = None
    odf_disk = None


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
    return orca.read_table(WORK_DIR+'testOnDiskDB', 'USPrices')


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

    @property
    def odf_dfs(self):
        return Csv.odf_dfs

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
        pdf = odf_disk = orca.read_table()

        # print(self.odf_csv.dtypes)
        # print(self.pdf_csv.dtypes)
        pdf = pd.DataFrame(columns=self.odf_csv._data_columns)
        odf = orca.DataFrame(columns=self.odf_csv._data_columns)
        # assert_frame_equal(self.pdf_csv, self.odf_csv.to_pandas())
        # assert_series_equal(self.pdf_csv.dtypes, self.odf_csv.dtypes)
        # assert_frame_equal(pdf.append(self.pdf_csv), odf.append(self.odf_csv).to_pandas())

    def test_dataframe_Combining_joining_merging_append_on_disk_unpa(self):
         orca.default_session().run(f"db = database('{WORK_DIR}imdx');")
         orca.save_table(WORK_DIR + "imdx", "imdb", self.odf)
         x = orca.read_table(WORK_DIR + "imdx", "imdb")
         x.append(self.odf, inplace=True)
         y = orca.read_table(WORK_DIR + "imdx", "imdb")
         # print(len(y))

    def test_dataframe_Combining_joining_merging_append_on_disk(self):
        odf = self.odf_disk
        pdf = self.pdf_csv
        # print(len(odf))
        # assert_frame_equal(pdf, odf.to_pandas())
        # print(len(odf.append(odf)))
        odf.append(odf, inplace=True)
        x = orca.read_table(WORK_DIR+'testOnDiskDB', 'USPrices')
        # print(len(x))

    def test_dataframe_Combining_joining_merging_append_dfs(self):
        odf = self.odf_dfs
        pdf = self.pdf_csv
        # assert_frame_equal(pdf, odf.to_pandas(), check_dtype=False)
        # print(len(odf))
        odf.append(odf, inplace=True)
        pdf.append(pdf)
        x = orca.read_table("dfs://USPricesDB", 'USPrices')
        # print(len(x))


if __name__ == '__main__':
    unittest.main()
