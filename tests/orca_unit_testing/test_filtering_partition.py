import unittest
import orca
import os.path as path
from setup.settings import *
from pandas.util.testing import *


def _create_odf_csv(data):
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


def _create_odf_pandas(n, pdf):
    # call function default_session() to get session object
    s = orca.default_session()

    # upload a local dataframe to a dfs table
    dolphindb_script = """
    login('admin', '123456')
    dbPath='dfs://filteringDB'
    if(existsDatabase(dbPath))
        dropDatabase(dbPath)
    db=database(dbPath, VALUE, 1..""" + str(n) + """) 
    tdata=table(1:0,`id`date`tsymbol`tbool`tchar`tshort`tint`long`tfloat`tdouble, 
    [INT,DATE,SYMBOL,BOOL,CHAR,SHORT,INT,LONG,FLOAT,DOUBLE]) 
    db.createPartitionedTable(tdata, `tb, `id) """
    s.run(dolphindb_script)
    s.run("tableInsert{loadTable('dfs://filteringDB',`tb)}", pdf)
    return orca.read_table("dfs://filteringDB", 'tb')


class Csv:
    pdf_csv = None
    odfs_csv = None
    pdf_csv_re = None
    odfs_csv_re = None


class FromDataframe:
    pdf = None
    odfs = None


class DfsFilteringTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # configure data directory
        DATA_DIR = path.abspath(path.join(__file__, "../setup/data"))
        fileName = 'USPricesSample.csv'
        data = os.path.join(DATA_DIR, fileName)
        data = data.replace('\\', '/')

        # Orca connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

        Csv.pdf_csv = pd.read_csv(data, parse_dates=[1],
                                  dtype={"PERMNO": np.int32, "SHRCD": np.int32, "HEXCD": np.int32,
                                         "DLSTCD": np.float32, "DLPRC": np.float32, "VOL": np.float32,
                                         "SHROUT": np.float32})
        Csv.pdf_csv_re = Csv.pdf_csv.set_index("PERMNO")
        Csv.pdf_csv.set_index("date", inplace=True)

        Csv.odfs_csv = _create_odf_csv(data)
        Csv.odfs_csv_re = Csv.odfs_csv.set_index("PERMNO")
        Csv.odfs_csv.set_index("date", inplace=True)

        n = 100  # note that n should be a multiple of 10
        re = n / 10
        FromDataframe.pdf = pd.DataFrame({
            "id": np.arange(1, n + 1, 1, dtype='int32'),
            'date': np.repeat(pd.date_range('2019.08.01', periods=10, freq='D'), re),
            'tsymbol': np.repeat(['a', 'b', 'c', 'd', 'e', 'QWW', 'FEA', 'FFW', 'DER', 'POD'], re),
            'tbool': np.repeat(np.repeat(np.arange(2, dtype='bool'), 5), re),
            'tchar': np.repeat(np.arange(1, 11, 1, dtype='int8'), re),
            'tshort': np.repeat(np.arange(1, 11, 1, dtype='int16'), re),
            'tint': np.repeat(np.arange(1, 11, 1, dtype='int32'), re),
            'tlong': np.repeat(np.arange(1, 11, 1, dtype='int64'), re),
            'tfloat': np.repeat(np.arange(1, 11, 1, dtype='float32'), re),
            'tdouble': np.repeat(np.arange(1, 11, 1, dtype='float64'), re)})
        FromDataframe.odfs = _create_odf_pandas(n, FromDataframe.pdf)

    # set column 'date' as index
    @property
    def pdf_csv(self):
        return Csv.pdf_csv

    @property
    def odfs_csv(self):
        return Csv.odfs_csv

    # set column 'PERMNO' as index
    @property
    def pdf_re(self):
        return Csv.pdf_csv_re

    @property
    def odfs_re(self):
        return Csv.odfs_csv_re

    # drop literal columns (temporal column 'date' has been set as index)
    @property
    def pdf_d(self):
        return self.pdf_csv.drop(columns=['TICKER', 'CUSIP', 'TRDSTAT', 'DLRET'])

    @property
    def odfs_d(self):
        return self.odfs_csv.drop(columns=['TICKER', 'CUSIP', 'TRDSTAT', 'DLRET'])

    @property
    def pdf(self):
        return FromDataframe.pdf.set_index("id")

    @property
    def odfs(self):
        return orca.DataFrame(self.pdf)

    # drop temporal, literal and logical columns
    @property
    def pdf_dr(self):
        return self.pdf.drop(columns=['date', 'tsymbol', 'tbool'])

    @property
    def odfs_dr(self):
        return self.odfs.drop(columns=['date', 'tsymbol', 'tbool'])

    def test_dfs_from_pandas_filtering_param_cond_equal(self):
        # filtering ==
        a = self.odfs[self.odfs["tsymbol"] == 'a'].to_pandas()
        b = self.pdf[self.pdf["tsymbol"] == 'a']
        assert_frame_equal(a, b, check_dtype=False)

    def test_dfs_from_pandas_filtering_param_cond_not_equal(self):
        # filtering !=
        a = self.odfs[self.odfs["tsymbol"] != 'a'].to_pandas()
        b = self.pdf[self.pdf["tsymbol"] != 'a']
        assert_frame_equal(a, b, check_dtype=False)

    def test_dfs_from_pandas_filtering_param_cond_less_than_or_equal(self):
        # filtering <=
        a = self.odfs[self.odfs["tfloat"] <= 3.0].to_pandas()
        b = self.pdf[self.pdf["tfloat"] <= 3.0]
        assert_frame_equal(a, b, check_dtype=False)

    def test_dfs_from_pandas_filtering_param_cond_greater_than_or_equal(self):
        #  filtering >=
        a = self.odfs[self.odfs["tfloat"] >= 3.0].to_pandas()
        b = self.pdf[self.pdf["tfloat"] >= 3.0]
        assert_frame_equal(a, b, check_dtype=False)

    def test_dfs_from_pandas_filtering_param_cond_less_than(self):
        #  filtering <
        a = self.odfs[self.odfs["tfloat"] < 3.0].to_pandas()
        b = self.pdf[self.pdf["tfloat"] < 3.0]
        assert_frame_equal(a, b, check_dtype=False)

    def test_dfs_from_pandas_filtering_param_cond_greater_than(self):
        #  filtering >
        a = self.odfs[self.odfs["tfloat"] > 3.0].to_pandas()
        b = self.pdf[self.pdf["tfloat"] > 3.0]
        assert_frame_equal(a, b, check_dtype=False)

    def test_dfs_from_pandas_filtering_and(self):
        #  filtering  &
        odfs = self.odfs
        pdf = self.pdf
        a = odfs[(odfs["tfloat"] < 3.0) & (odfs["tfloat"] > 1.0)].to_pandas()
        b = pdf[(pdf["tfloat"] < 3.0) & (pdf["tfloat"] > 1.0)]
        ad = odfs[(odfs["tfloat"] < 3.0), (odfs["tfloat"] > 1.0)].to_pandas()
        assert_frame_equal(a, b, check_dtype=False)
        assert_frame_equal(ad, b, check_dtype=False)

    def test_dfs_from_pandas_filtering_and_and(self):
        #  filtering & &
        odfs = self.odfs
        pdf = self.pdf
        a = odfs[(odfs["tfloat"] < 3.0) & (odfs["tfloat"] > 1.0) & (odfs["tint"] > 2)].to_pandas()
        b = pdf[(pdf["tfloat"] < 3.0) & (pdf["tfloat"] > 1.0) & (pdf["tint"] > 2)]
        ad = odfs[(odfs["tfloat"] < 3.0), (odfs["tfloat"] > 1.0), (odfs["tint"] > 2)].to_pandas()
        assert_frame_equal(a, b, check_dtype=False)
        assert_frame_equal(ad, b, check_dtype=False)

    def test_dfs_from_pandas_filtering_and_or(self):
        #  filtering & |
        odfs = self.odfs
        pdf = self.pdf
        a = odfs[(odfs["tfloat"] < 3.0) & (odfs["tfloat"] > 1.0) | (odfs["tfloat"] == 4.0)].to_pandas()
        b = pdf[(pdf["tfloat"] < 3.0) & (pdf["tfloat"] > 1.0) | (pdf["tfloat"] == 4.0)]
        ad = odfs[(odfs["tfloat"] < 3.0), (odfs["tfloat"] > 1.0) | (odfs["tfloat"] == 4.0)].to_pandas()
        bd = pdf[(pdf["tfloat"] < 3.0) & ((pdf["tfloat"] > 1.0) | (pdf["tfloat"] == 4.0))]
        assert_frame_equal(a, b, check_dtype=False)
        self.assertEqual(repr(ad), repr(bd))

    def test_dfs_from_pandas_filtering_or(self):
        #  filtering |
        odfs = self.odfs
        pdf = self.pdf
        a = odfs[(odfs["tfloat"] < 3.0) | (odfs["tfloat"] > 4.0)].to_pandas()
        b = pdf[(pdf["tfloat"] < 3.0) | (pdf["tfloat"] > 4.0)]
        assert_frame_equal(a, b, check_dtype=False)

    def test_dfs_from_pandas_filtering_or_and(self):
        #  filtering | &
        odfs = self.odfs
        pdf = self.pdf
        a = odfs[(odfs["tfloat"] == 3.0) | (odfs["tfloat"] > 1.0) & (odfs["tint"] > 2)].to_pandas()
        b = pdf[(pdf["tfloat"] == 3.0) | (pdf["tfloat"] > 1.0) & (pdf["tint"] > 2)]
        ad = odfs[(odfs["tfloat"] == 3.0) | (odfs["tfloat"] > 1.0), (odfs["tint"] > 2)].to_pandas()
        assert_frame_equal(a, b, check_dtype=False)
        assert_frame_equal(ad, b, check_dtype=False)

    def test_dfs_from_pandas_filtering_or_or(self):
        #  filtering | |
        odfs = self.odfs
        pdf = self.pdf
        a = odfs[(odfs["tfloat"] < 3.0) | (odfs["tfloat"] > 1.0) | (odfs["tfloat"] == 4.0)].to_pandas()
        b = pdf[(pdf["tfloat"] < 3.0) | (pdf["tfloat"] > 1.0) | (pdf["tfloat"] == 4.0)]
        assert_frame_equal(a, b, check_dtype=False)

    def test_dfs_from_pandas_filtering_operation_add(self):
        #  filtering +
        odfs_dr = self.odfs_dr
        pdf_dr = self.pdf_dr
        b = pdf_dr[pdf_dr["tfloat"] < 3.0] + pdf_dr[pdf_dr["tfloat"] > 1.0]
        a = (odfs_dr[odfs_dr["tfloat"] < 3.0] + odfs_dr[odfs_dr["tfloat"] > 1.0]).to_pandas()
        assert_frame_equal(a, b, check_dtype=False)

    def test_dfs_from_pandas_filtering_operation_sub(self):
        #  filtering -
        odfs_dr = self.odfs_dr
        pdf_dr = self.pdf_dr
        a = (odfs_dr[odfs_dr["tfloat"] < 3.0] - odfs_dr[odfs_dr["tfloat"] > 1.0]).to_pandas()
        b = pdf_dr[pdf_dr["tfloat"] < 3.0] - pdf_dr[pdf_dr["tfloat"] > 1.0]
        assert_frame_equal(a, b, check_dtype=False)

    def test_dfs_from_pandas_filtering_operation_mul(self):
        #  filtering *
        odfs_dr = self.odfs_dr
        pdf_dr = self.pdf_dr
        a = (odfs_dr[odfs_dr["tfloat"] < 3.0] * odfs_dr[odfs_dr["tfloat"] > 1.0]).to_pandas()
        b = pdf_dr[pdf_dr["tfloat"] < 3.0] * pdf_dr[pdf_dr["tfloat"] > 1.0]
        assert_frame_equal(a, b, check_dtype=False)

    def test_dfs_from_pandas_filtering_operation_div(self):
        #  filtering /
        odfs_dr = self.odfs_dr
        pdf_dr = self.pdf_dr
        a = (odfs_dr[odfs_dr["tfloat"] < 3.0] / odfs_dr[odfs_dr["tfloat"] > 1.0]).to_pandas()
        b = pdf_dr[pdf_dr["tfloat"] < 3.0] / pdf_dr[pdf_dr["tfloat"] > 1.0]
        assert_frame_equal(a, b, check_dtype=False)

    def test_dfs_from_pandas_filtering_operation_groupby(self):
        #  filtering groupby sum
        a = self.odfs[self.odfs["tfloat"] < 3.0].groupby("tsymbol").sum().to_pandas()
        b = self.pdf[self.pdf["tfloat"] < 3.0].groupby("tsymbol").sum()
        assert_frame_equal(a, b, check_dtype=False)

        #  filtering groupby count
        a = self.odfs[self.odfs["tfloat"] < 3.0].groupby("tsymbol").count().to_pandas()
        b = self.pdf[self.pdf["tfloat"] < 3.0].groupby("tsymbol").count()
        assert_frame_equal(a, b, check_dtype=False)

    def test_dfs_from_pandas_filtering_operation_resample(self):
        a = self.odfs[self.odfs["tfloat"] < 5.0].resample("d", on="date").sum().to_pandas()
        b = self.pdf[self.pdf["tfloat"] < 5.0].resample("d", on="date").sum()
        b['tbool'] = b['tbool'].astype(int)
        assert_frame_equal(a, b, check_dtype=False)

    def test_dfs_from_pandas_filtering_operation_rolling(self):
        a = self.odfs[self.odfs["tfloat"] < 5.0].rolling(window=2, on="date").sum().to_pandas()
        b = self.pdf[self.pdf["tfloat"] < 5.0].rolling(window=2, on="date").sum()
        assert_frame_equal(a, b, check_dtype=False)

    def test_dfs_from_import_filtering_param_cond_equal(self):
        # filtering ==
        a = self.odfs_csv[self.odfs_csv["TICKER"] == "EGAS"].to_pandas().fillna(value="")
        b = self.pdf_csv[self.pdf_csv["TICKER"] == "EGAS"].fillna(value="")
        assert_frame_equal(a, b, check_dtype=False)

    def test_dfs_from_import_filtering_param_cond_not_equal(self):
        # filtering !=
        a = self.odfs_csv[self.odfs_csv["TICKER"] != "EGAS"].to_pandas().sort_values(['PERMNO', "TICKER"]).reset_index(
            drop=True).fillna(
            value="")
        b = self.pdf_csv[self.pdf_csv["TICKER"] != "EGAS"].sort_values(['PERMNO', "TICKER"]).reset_index(
            drop=True).fillna(value="")
        assert_frame_equal(a, b, check_dtype=False)

    def test_dfs_from_import_filtering_param_cond_less_than_or_equal(self):
        # filtering <=
        a = self.odfs_csv[self.odfs_csv["PRC"] <= 10.25].to_pandas().sort_values(['PERMNO', "TICKER"]).reset_index(
            drop=True).fillna(
            value="")
        b = self.pdf_csv[self.pdf_csv["PRC"] <= 10.25].sort_values(['PERMNO', "TICKER"]).reset_index(drop=True).fillna(
            value="")
        # DIFFS
        # assert_frame_equal(a, b, check_dtype=False)

    def test_dfs_from_import_filtering_param_cond_greater_than_or_equal(self):
        #  filtering >=
        a = self.odfs_csv[self.odfs_csv["PRC"] >= 10.25].to_pandas().sort_values(['PERMNO', "TICKER"]).reset_index(
            drop=True).fillna(
            value="")
        b = self.pdf_csv[self.pdf_csv["PRC"] >= 10.25].sort_values(['PERMNO', "TICKER"]).reset_index(drop=True).fillna(
            value="")
        assert_frame_equal(a, b, check_dtype=False)

    def test_dfs_from_import_filtering_param_cond_less_than(self):
        #  filtering <
        a = self.odfs_csv[self.odfs_csv["PRC"] < 10.25].to_pandas().sort_values(['PERMNO', "TICKER"]).reset_index(
            drop=True).fillna(
            value="")
        b = self.pdf_csv[self.pdf_csv["PRC"] < 10.25].sort_values(['PERMNO', "TICKER"]).reset_index(drop=True).fillna(
            value="")
        # DIFFS
        # assert_frame_equal(a, b, check_dtype=False)

    def test_dfs_from_import_filtering_param_cond_greater_than(self):
        #  filtering >
        a = self.odfs_csv[self.odfs_csv["PRC"] > 10.25].to_pandas().sort_values(['PERMNO', "TICKER"]).reset_index(
            drop=True).fillna(
            value="")
        b = self.pdf_csv[self.pdf_csv["PRC"] > 10.25].sort_values(['PERMNO', "TICKER"]).reset_index(drop=True).fillna(
            value="")
        assert_frame_equal(a, b, check_dtype=False)

    def test_dfs_from_import_filtering_and(self):
        #  filtering  &
        a = self.odfs_csv[(self.odfs_csv["CFACPR"] > 1.4) & (self.odfs_csv["SHROUT"] > 5000)].to_pandas().fillna(
            value="")
        b = self.pdf_csv[(self.pdf_csv["CFACPR"] > 1.4) & (self.pdf_csv["SHROUT"] > 5000)].fillna(value="")
        ad = self.odfs_csv[(self.odfs_csv["CFACPR"] > 1.4), (self.odfs_csv["SHROUT"] > 5000)].to_pandas().fillna(
            value="")
        assert_frame_equal(a, b, check_dtype=False)
        assert_frame_equal(ad, b, check_dtype=False)

    def test_dfs_from_import_filtering_or(self):
        #  filtering |
        a = self.odfs_csv[(self.odfs_csv["SHROUT"] > 16600) | (self.odfs_csv["CFACPR"] > 0.1)].to_pandas()
        b = self.pdf_csv[(self.pdf_csv["SHROUT"] > 16600) | (self.pdf_csv["CFACPR"] > 0.1)]
        a = a.sort_values(['PERMNO', "TICKER"]).reset_index(drop=True).fillna(value="")
        b = b.sort_values(['PERMNO', "TICKER"]).reset_index(drop=True).fillna(value="")
        assert_frame_equal(a, b, check_dtype=False)

    def test_dfs_from_import_filtering_and_and(self):
        #  filtering & &
        a = self.odfs_csv[(self.odfs_csv["OPENPRC"] > 40) & (self.odfs_csv["SHROUT"] > 15000) & (
                    self.odfs_csv["CFACPR"] > 1)].to_pandas().fillna(value="")
        b = self.pdf_csv[
            (self.pdf_csv["OPENPRC"] > 40) & (self.pdf_csv["SHROUT"] > 15000) & (self.pdf_csv["CFACPR"] > 1)].fillna(
            value="")
        ad = self.odfs_csv[(self.odfs_csv["OPENPRC"] > 40), (self.odfs_csv["SHROUT"] > 15000), (
                    self.odfs_csv["CFACPR"] > 1)].to_pandas().fillna(value="")
        assert_frame_equal(a, b, check_dtype=False)
        assert_frame_equal(ad, b, check_dtype=False)

    def test_dfs_from_import_filtering_and_or(self):
        # filtering &
        a = self.odfs_csv[(self.odfs_csv["OPENPRC"] > 40) & (self.odfs_csv["SHROUT"] > 15000) | (
                    self.odfs_csv["TRDSTAT"] == "A")].to_pandas()
        b = self.pdf_csv[
            (self.pdf_csv["OPENPRC"] > 40) & (self.pdf_csv["SHROUT"] > 15000) | (self.pdf_csv["TRDSTAT"] == "A")]
        a = a.sort_values(['PERMNO', "TICKER"]).reset_index(drop=True).fillna(value="")
        b = b.sort_values(['PERMNO', "TICKER"]).reset_index(drop=True).fillna(value="")
        assert_frame_equal(a, b, check_dtype=False)

    def test_dfs_from_import_filtering_or_or(self):
        #  filtering | |
        a = self.odfs_csv[(self.odfs_csv["OPENPRC"] > 40) | (self.odfs_csv["SHROUT"] > 15000) | (
                    self.odfs_csv["TRDSTAT"] == "A")].to_pandas()
        b = self.pdf_csv[
            (self.pdf_csv["OPENPRC"] > 40) | (self.pdf_csv["SHROUT"] > 15000) | (self.pdf_csv["TRDSTAT"] == "A")]
        a = a.sort_values(['PERMNO', "TICKER"]).reset_index(drop=True).fillna(value="")
        b = b.sort_values(['PERMNO', "TICKER"]).reset_index(drop=True).fillna(value="")
        assert_frame_equal(a, b, check_dtype=False)

    def test_dfs_from_import_filtering_or_and(self):
        #  filtering | &
        a = self.odfs_csv[(self.odfs_csv["OPENPRC"] > 40) | (self.odfs_csv["SHROUT"] > 15000) & (
                    self.odfs_csv["TRDSTAT"] == "A")].to_pandas()
        b = self.pdf_csv[
            (self.pdf_csv["OPENPRC"] > 40) | (self.pdf_csv["SHROUT"] > 15000) & (self.pdf_csv["TRDSTAT"] == "A")]
        a = a.sort_values(['PERMNO', "TICKER"]).reset_index(drop=True).fillna(value="")
        b = b.sort_values(['PERMNO', "TICKER"]).reset_index(drop=True).fillna(value="")
        assert_frame_equal(a, b, check_dtype=False)

        ad = self.odfs_csv[(self.odfs_csv["OPENPRC"] > 40) | (self.odfs_csv["SHROUT"] > 15000), (
                    self.odfs_csv["TRDSTAT"] == "A")].to_pandas()
        ad = ad.sort_values(['PERMNO', "TICKER"]).reset_index(drop=True).fillna(value="")
        assert_frame_equal(ad, b, check_dtype=False)

    def test_dfs_from_import_filtering_operation_add(self):
        #  filtering +
        a = (self.odfs_d[self.odfs_d["OPENPRC"] > 40] + self.odfs_d[self.odfs_d["SHROUT"] > 15000]).to_pandas()
        b = self.pdf_d[self.pdf_d["OPENPRC"] > 40] + self.pdf_d[self.pdf_d["SHROUT"] > 15000]
        # TODO: ASSERT RANDOM ORDER
        # assert_frame_equal(a, b, check_dtype=False)

    def test_dfs_from_import_filtering_operation_sub(self):
        #  filtering -
        a = (self.odfs_d[self.odfs_d["OPENPRC"] > 40] - self.odfs_d[self.odfs_d["SHROUT"] > 15000]).to_pandas()
        b = self.pdf_d[self.pdf_d["OPENPRC"] > 40] - self.pdf_d[self.pdf_d["SHROUT"] > 15000]
        # TODO: ASSERT RANDOM ORDER
        # assert_frame_equal(a, b, check_dtype=False)

    def test_dfs_from_import_filtering_operation_mul(self):
        #  filtering *
        a = (self.odfs_d[self.odfs_d["OPENPRC"] > 40] * self.odfs_d[self.odfs_d["SHROUT"] > 15000]).to_pandas()
        b = self.pdf_d[self.pdf_d["OPENPRC"] > 40] * self.pdf_d[self.pdf_d["SHROUT"] > 15000]
        # TODO: ASSERT RANDOM ORDER
        # assert_frame_equal(a, b, check_dtype=False)

    def test_dfs_from_import_filtering_operation_div(self):
        #  filtering /
        a = (self.odfs_d[self.odfs_d["OPENPRC"] > 40] / self.odfs_d[self.odfs_d["SHROUT"] > 15000]).to_pandas()
        b = self.pdf_d[self.pdf_d["OPENPRC"] > 40] / self.pdf_d[self.pdf_d["SHROUT"] > 15000]
        # TODO: ASSERT RANDOM ORDER
        # assert_frame_equal(a, b, check_dtype=False)

    def test_dfs_from_import_filtering_operation_groupby(self):
        #  filtering groupby sum
        a = self.odfs_csv[self.odfs_csv["OPENPRC"] > 40].groupby("date").sum().to_pandas()
        b = self.pdf_csv[self.pdf_csv["OPENPRC"] > 40].groupby("date").sum()
        assert_frame_equal(a, b, check_dtype=False)

    def test_dfs_from_import_filtering_operation_resample(self):
        a = self.odfs_re[self.odfs_re["OPENPRC"] > 40].resample("d", on="date").sum().to_pandas()
        b = self.pdf_re[self.pdf_re["OPENPRC"] > 40].resample("d", on="date").sum()
        # TODO：FILTER.RESAMPLE
        # assert_frame_equal(a, b, check_dtype=False)

    def test_dfs_from_import_filtering_operation_rolling(self):
        a = self.odfs_re[self.odfs_re["OPENPRC"] > 40].rolling(window=2, on="date").sum().to_pandas()
        b = self.pdf_re[self.pdf_re["OPENPRC"] > 40].rolling(window=2, on="date").sum()
        # TODO: ASSERT RANDOM ORDER
        # assert_frame_equal(a, b, check_dtype=False)


if __name__ == '__main__':
    unittest.main()
