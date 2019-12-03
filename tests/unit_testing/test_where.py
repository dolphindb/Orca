import unittest
import orca
import os.path as path
from setup.settings import *
from pandas.util.testing import *


class Csv:
    pdf_csv = None
    odf_csv = None


class WhereTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # configure data directory
        DATA_DIR = path.abspath(path.join(__file__, "../setup/data"))
        fileName = 'USPricesSample.csv'
        data = os.path.join(DATA_DIR, fileName)
        data = data.replace('\\', '/')

        # Orca connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

        # odf from import
        Csv.odf_csv = orca.read_csv(data, dtype={"PERMNO": np.int32, "date": 'DATE', "TRDSTAT": 'SYMBOL',
                                                 "DLSTCD": np.float32,
                                                 "DLPRC": np.float32, "VOL": np.float32, "SHROUT": np.float32})
        # pdf from import
        Csv.pdf_csv = pd.read_csv(data, parse_dates=[1],
                                  dtype={"PERMNO": np.int32, "SHRCD": np.int32, "HEXCD": np.int32, "DLSTCD": np.float32,
                                         "DLPRC": np.float32, "VOL": np.float32, "SHROUT": np.float32})

    # set column 'date' as index
    @property
    def pdf_csv(self):
        return Csv.pdf_csv.set_index("date")

    @property
    def odf_csv(self):
        return Csv.odf_csv.set_index("date")

    # set column 'PERMNO' as index
    @property
    def pdf_re(self):
        return Csv.pdf_csv.set_index("PERMNO")

    @property
    def odf_re(self):
        return Csv.odf_csv.set_index("PERMNO")

    # drop literal columns (temporal column 'date' has been set as index)
    @property
    def pdf_d(self):
        return self.pdf_csv.drop(columns=['TICKER', 'CUSIP', 'TRDSTAT', 'DLRET'])

    @property
    def odf_d(self):
        return self.odf_csv.drop(columns=['TICKER', 'CUSIP', 'TRDSTAT', 'DLRET'])

    @property
    def pdf(self):
        n = 100  # note that n should be a multiple of 10
        re = n / 10
        return pd.DataFrame({
                               'date': np.repeat(pd.date_range('2019.08.01', periods=10, freq='D'), re),
                               'tsymbol': np.repeat(['a', 'b', 'c', 'd', 'e', 'QWW', 'FEA', 'FFW', 'DER', 'POD'], re),
                               'tbool': np.repeat(np.repeat(np.arange(2, dtype='bool'), 5), re),
                               'tchar': np.repeat(np.arange(1, 11, 1, dtype='int8'), re),
                               'tshort': np.repeat(np.arange(1, 11, 1, dtype='int16'), re),
                               'tint': np.repeat(np.arange(1, 11, 1, dtype='int32'), re),
                               'tlong': np.repeat(np.arange(1, 11, 1, dtype='int64'), re),
                               'tfloat': np.repeat(np.arange(1, 11, 1, dtype='float32'), re),
                               'tdouble': np.repeat(np.arange(1, 11, 1, dtype='float64'), re)
                               }, index=pd.Index(np.arange(1, n + 1, 1, dtype='int32'), name="id"))

    @property
    def odf(self):
        return orca.DataFrame(self.pdf)

    # drop temporal, literal and logical columns
    @property
    def pdf_dr(self):
        return self.pdf.drop(columns=['date', 'tsymbol', 'tbool'])

    @property
    def odf_dr(self):
        return self.odf.drop(columns=['date', 'tsymbol', 'tbool'])

    def test_from_pandas_where_param_cond_equal(self):
        # where ==
        a = self.odf[self.odf["tsymbol"] == 'a'].to_pandas()
        b = self.pdf[self.pdf["tsymbol"] == 'a']
        assert_frame_equal(a, b, check_dtype=False)

    def test_from_pandas_where_param_cond_not_equal(self):
        # where !=
        a = self.odf[self.odf["tsymbol"] != 'a'].to_pandas()
        b = self.pdf[self.pdf["tsymbol"] != 'a']
        assert_frame_equal(a, b, check_dtype=False)

    def test_from_pandas_where_param_cond_less_than_or_equal(self):
        # where <=
        a = self.odf[self.odf["tfloat"] <= 3.0].to_pandas()
        b = self.pdf[self.pdf["tfloat"] <= 3.0]
        assert_frame_equal(a, b, check_dtype=False)

        a = self.odf[(self.odf["date"] < orca.Timestamp("2019.08.05"))].to_pandas()
        b = self.pdf[(self.pdf["date"] < pd.Timestamp("2019.08.05"))]
        assert_frame_equal(a, b, check_dtype=False)

        a = self.odf[(self.odf["date"] < "2019.08.05")].to_pandas()
        b = self.pdf[(self.pdf["date"] < "2019.08.05")]
        assert_frame_equal(a, b, check_dtype=False)

    def test_from_pandas_where_param_cond_greater_than_or_equal(self):
        #  where >=
        a = self.odf[self.odf["tfloat"] >= 3.0].to_pandas()
        b = self.pdf[self.pdf["tfloat"] >= 3.0]
        assert_frame_equal(a, b, check_dtype=False)

    def test_from_pandas_where_param_cond_less_than(self):
        #  where <
        a = self.odf[self.odf["tfloat"] < 3.0].to_pandas()
        b = self.pdf[self.pdf["tfloat"] < 3.0]
        assert_frame_equal(a, b, check_dtype=False)

    def test_from_pandas_where_param_cond_greater_than(self):
        #  where >
        a = self.odf[self.odf["tfloat"] > 3.0].to_pandas()
        b = self.pdf[self.pdf["tfloat"] > 3.0]
        assert_frame_equal(a, b, check_dtype=False)

    def test_from_pandas_where_and(self):
        #  where  &
        odf = self.odf
        pdf = self.pdf
        a = odf[(odf["tfloat"] < 3.0) & (odf["tfloat"] > 1.0)].to_pandas()
        b = pdf[(pdf["tfloat"] < 3.0) & (pdf["tfloat"] > 1.0)]
        ad = odf[(odf["tfloat"] < 3.0), (odf["tfloat"] > 1.0)].to_pandas()
        assert_frame_equal(a, b, check_dtype=False)
        assert_frame_equal(ad, b, check_dtype=False)

    def test_from_pandas_where_and_and(self):
        #  where & &
        odf = self.odf
        pdf = self.pdf
        a = odf[(odf["tfloat"] < 3.0) & (odf["tfloat"] > 1.0) & (odf["tint"] > 2)].to_pandas()
        b = pdf[(pdf["tfloat"] < 3.0) & (pdf["tfloat"] > 1.0) & (pdf["tint"] > 2)]
        ad = odf[(odf["tfloat"] < 3.0), (odf["tfloat"] > 1.0), (odf["tint"] > 2)].to_pandas()
        assert_frame_equal(a, b, check_dtype=False)
        assert_frame_equal(ad, b, check_dtype=False)

    def test_from_pandas_where_and_or(self):
        #  where & |
        odf = self.odf
        pdf = self.pdf
        a = odf[(odf["tfloat"] < 3.0) & (odf["tfloat"] > 1.0) | (odf["tfloat"] == 4.0)].to_pandas()
        b = pdf[(pdf["tfloat"] < 3.0) & (pdf["tfloat"] > 1.0) | (pdf["tfloat"] == 4.0)]
        ad = odf[(odf["tfloat"] < 3.0), (odf["tfloat"] > 1.0) | (odf["tfloat"] == 4.0)].to_pandas()
        bd = pdf[(pdf["tfloat"] < 3.0) & ((pdf["tfloat"] > 1.0) | (pdf["tfloat"] == 4.0))]
        assert_frame_equal(a, b, check_dtype=False)
        assert_frame_equal(ad, bd, check_dtype=False)

    def test_from_pandas_where_or(self):
        #  where |
        odf = self.odf
        pdf = self.pdf
        b = pdf[(pdf["tfloat"] < 3.0) | (pdf["tfloat"] > 4.0)]
        a = odf[(odf["tfloat"] < 3.0) | (odf["tfloat"] > 4.0)].to_pandas()
        assert_frame_equal(a, b, check_dtype=False)

    def test_from_pandas_where_or_and(self):
        #  where | &
        odf = self.odf
        pdf = self.pdf
        a = odf[(odf["tfloat"] == 3.0) | (odf["tfloat"] > 1.0) & (odf["tint"] > 2)].to_pandas()
        b = pdf[(pdf["tfloat"] == 3.0) | (pdf["tfloat"] > 1.0) & (pdf["tint"] > 2)]
        ad = odf[(odf["tfloat"] == 3.0) | (odf["tfloat"] > 1.0), (odf["tint"] > 2)].to_pandas()
        assert_frame_equal(a, b, check_dtype=False)
        assert_frame_equal(ad, b, check_dtype=False)

    def test_from_pandas_where_or_or(self):
        #  where | |
        odf = self.odf
        pdf = self.pdf
        a = odf[(odf["tfloat"] < 3.0) | (odf["tfloat"] > 1.0) | (odf["tfloat"] == 4.0)].to_pandas()
        b = pdf[(pdf["tfloat"] < 3.0) | (pdf["tfloat"] > 1.0) | (pdf["tfloat"] == 4.0)]
        assert_frame_equal(a, b, check_dtype=False)

    def test_from_pandas_where_operation_add(self):
        #  where +
        a = (self.odf_dr[self.odf_dr["tfloat"] < 3.0] + self.odf_dr[self.odf_dr["tfloat"] > 1.0]).to_pandas()
        b = self.pdf_dr[self.pdf_dr["tfloat"] < 3.0] + self.pdf_dr[self.pdf_dr["tfloat"] > 1.0]
        assert_frame_equal(a, b, check_dtype=False)

    def test_from_pandas_where_operation_sub(self):
        #  where -
        a = (self.odf_dr[self.odf_dr["tfloat"] < 3.0] - self.odf_dr[self.odf_dr["tfloat"] > 1.0]).to_pandas()
        b = self.pdf_dr[self.pdf_dr["tfloat"] < 3.0] - self.pdf_dr[self.pdf_dr["tfloat"] > 1.0]
        assert_frame_equal(a, b, check_dtype=False)

    def test_from_pandas_where_operation_mul(self):
        #  where *
        a = (self.odf_dr[self.odf_dr["tfloat"] < 3.0] * self.odf_dr[self.odf_dr["tfloat"] > 1.0]).to_pandas()
        b = self.pdf_dr[self.pdf_dr["tfloat"] < 3.0] * self.pdf_dr[self.pdf_dr["tfloat"] > 1.0]
        assert_frame_equal(a, b, check_dtype=False)

    def test_from_pandas_where_operation_div(self):
        #  where /
        # note that operation div is only allowed for numbers, exclusive of temporal, literal and logical
        # columns in Orca.
        a = (self.odf_dr[self.odf_dr["tfloat"] < 3.0] / self.odf_dr[self.odf_dr["tfloat"] > 1.0]).to_pandas()
        b = self.pdf_dr[self.pdf_dr["tfloat"] < 3.0] / self.pdf_dr[self.pdf_dr["tfloat"] > 1.0]
        assert_frame_equal(a, b, check_dtype=False)

    def test_from_pandas_where_operation_groupby(self):
        #  where groupby sum
        a = self.odf[self.odf["tfloat"] < 3.0].groupby("tsymbol").sum().to_pandas()
        b = self.pdf[self.pdf["tfloat"] < 3.0].groupby("tsymbol").sum()
        b['tbool'] = b['tbool'].astype(int)
        assert_frame_equal(a, b, check_dtype=False)

        #  where groupby count
        a = self.odf[self.odf["tfloat"] < 3.0].groupby("tsymbol").count().to_pandas()
        b = self.pdf[self.pdf["tfloat"] < 3.0].groupby("tsymbol").count()
        b['tbool'] = b['tbool'].astype(int)
        assert_frame_equal(a, b, check_dtype=False)

    def test_from_pandas_where_operation_resample(self):
        #  where groupby sum
        a = self.odf[self.odf["tfloat"] < 5.0].resample("d", on="date").sum().to_pandas()
        b = self.pdf[self.pdf["tfloat"] < 5.0].resample("d", on="date").sum()
        b['tbool'] = b['tbool'].astype(int)
        assert_frame_equal(a, b, check_dtype=False)

    def test_from_pandas_where_operation_rolling(self):
        #  where groupby sum
        a = self.odf[self.odf["tfloat"] < 5.0].rolling(window=2, on="date").sum().to_pandas()
        b = self.pdf[self.pdf["tfloat"] < 5.0].rolling(window=2, on="date").sum()
        assert_frame_equal(a, b, check_dtype=False)

    def test_from_import_where_param_cond_equal(self):
        # where ==
        a = self.odf_csv[self.odf_csv["TICKER"] == "EGAS"].to_pandas().fillna(value="")
        b = self.pdf_csv[self.pdf_csv["TICKER"] == "EGAS"].fillna(value="")
        assert_frame_equal(a, b, check_dtype=False)

    def test_from_import_where_param_cond_not_equal(self):
        # where !=
        a = self.odf_csv[self.odf_csv["TICKER"] != "EGAS"].to_pandas().fillna(value="")
        b = self.pdf_csv[self.pdf_csv["TICKER"] != "EGAS"].fillna(value="")
        assert_frame_equal(a, b, check_dtype=False)

    def test_from_import_where_param_cond_less_than_or_equal(self):
        # where <=
        a = self.odf_csv[self.odf_csv["PRC"] <= 10.25].to_pandas().fillna(value="")
        b = self.pdf_csv[self.pdf_csv["PRC"] <= 10.25].fillna(value="")
        # DIFFS
        # assert_frame_equal(a, b, check_dtype=False)

    def test_from_import_where_param_cond_greater_than_or_equal(self):
        #  where >=
        a = self.odf_csv[self.odf_csv["PRC"] >= 10.25].to_pandas().fillna(value="")
        b = self.pdf_csv[self.pdf_csv["PRC"] >= 10.25].fillna(value="")
        assert_frame_equal(a, b, check_dtype=False)

    def test_from_import_where_param_cond_less_than(self):
        #  where <
        a = self.odf_csv[self.odf_csv["PRC"] < 10.25].to_pandas().fillna(value="")
        b = self.pdf_csv[self.pdf_csv["PRC"] < 10.25].fillna(value="")
        # DIFFS
        # assert_frame_equal(a, b, check_dtype=False)

    def test_from_import_where_param_cond_greater_than(self):
        #  where >
        a = self.odf_csv[self.odf_csv["PRC"] > 10.25].to_pandas().fillna(value="")
        b = self.pdf_csv[self.pdf_csv["PRC"] > 10.25].fillna(value="")
        assert_frame_equal(a, b, check_dtype=False)

    def test_from_import_where_and(self):
        # where  &
        a = self.odf_csv[(self.odf_csv["CFACPR"] > 1.4) & (self.odf_csv["SHROUT"] > 5000)].to_pandas().fillna(value="")
        b = self.pdf_csv[(self.pdf_csv["CFACPR"] > 1.4) & (self.pdf_csv["SHROUT"] > 5000)].fillna(value="")
        ad = self.odf_csv[(self.odf_csv["CFACPR"] > 1.4), (self.odf_csv["SHROUT"] > 5000)].to_pandas().fillna(value="")
        assert_frame_equal(a, b, check_dtype=False)
        assert_frame_equal(ad, b, check_dtype=False)

    def test_from_import_where_or(self):
        #  where |
        a = self.odf_csv[(self.odf_csv["SHROUT"] > 16600) | (self.odf_csv["CFACPR"] > 0.1)].to_pandas().fillna(value="")
        b = self.pdf_csv[(self.pdf_csv["SHROUT"] > 16600) | (self.pdf_csv["CFACPR"] > 0.1)].fillna(value="")
        assert_frame_equal(a, b, check_dtype=False)

    def test_from_import_where_and_and(self):
        #  where & &
        a = self.odf_csv[(self.odf_csv["OPENPRC"] > 40) & (self.odf_csv["SHROUT"] > 15000) & (self.odf_csv["CFACPR"] > 1)].to_pandas().fillna(value="")
        b = self.pdf_csv[(self.pdf_csv["OPENPRC"] > 40) & (self.pdf_csv["SHROUT"] > 15000) & (self.pdf_csv["CFACPR"] > 1)].fillna(value="")
        ad = self.odf_csv[(self.odf_csv["OPENPRC"] > 40), (self.odf_csv["SHROUT"] > 15000), (self.odf_csv["CFACPR"] > 1)].to_pandas().fillna(value="")
        assert_frame_equal(a, b, check_dtype=False)
        assert_frame_equal(ad, b, check_dtype=False)

    def test_from_import_where_and_or(self):
        #  where & |
        a = self.odf_csv[(self.odf_csv["OPENPRC"] > 40) & (self.odf_csv["SHROUT"] > 15000) | (self.odf_csv["TRDSTAT"] == "A")].to_pandas().fillna(value="")
        b = self.pdf_csv[(self.pdf_csv["OPENPRC"] > 40) & (self.pdf_csv["SHROUT"] > 15000) | (self.pdf_csv["TRDSTAT"] == "A")].fillna(value="")
        assert_frame_equal(a, b, check_dtype=False)

        ad = self.odf_csv[(self.odf_csv["OPENPRC"] > 40), (self.odf_csv["SHROUT"] > 15000) | (self.odf_csv["TRDSTAT"] == "A")].to_pandas().fillna(value="")
        bd = self.pdf_csv[(self.pdf_csv["OPENPRC"] > 40) & ((self.pdf_csv["SHROUT"] > 15000) | (self.pdf_csv["TRDSTAT"] == "A"))].fillna(value="")
        assert_frame_equal(ad, bd, check_dtype=False)

    def test_from_import_where_or_or(self):
        #  where | |
        a = self.odf_csv[(self.odf_csv["OPENPRC"] > 40) | (self.odf_csv["SHROUT"] > 15000) | (self.odf_csv["TRDSTAT"] == "A")].to_pandas().fillna(value="")
        b = self.pdf_csv[(self.pdf_csv["OPENPRC"] > 40) | (self.pdf_csv["SHROUT"] > 15000) | (self.pdf_csv["TRDSTAT"] == "A")].fillna(value="")
        assert_frame_equal(a, b, check_dtype=False)

    def test_from_import_where_or_and(self):
        #  where | &
        a = self.odf_csv[(self.odf_csv["OPENPRC"] > 40) | (self.odf_csv["SHROUT"] > 15000) & (self.odf_csv["TRDSTAT"] == "A")].to_pandas().fillna(value="")
        b = self.pdf_csv[(self.pdf_csv["OPENPRC"] > 40) | (self.pdf_csv["SHROUT"] > 15000) & (self.pdf_csv["TRDSTAT"] == "A")].fillna(value="")
        assert_frame_equal(a, b, check_dtype=False)

        ad = self.odf_csv[(self.odf_csv["OPENPRC"] > 40) | (self.odf_csv["SHROUT"] > 15000), (self.odf_csv["TRDSTAT"] == "A")].to_pandas().fillna(value="")
        assert_frame_equal(ad, b, check_dtype=False)

    def test_from_import_where_operation_add(self):
        #  where +
        a = (self.odf_d[self.odf_d["OPENPRC"] > 40] + self.odf_d[self.odf_d["SHROUT"] > 15000]).to_pandas()
        b = self.pdf_d[self.pdf_d["OPENPRC"] > 40] + self.pdf_d[self.pdf_d["SHROUT"] > 15000]
        # TODO: ASSERT RANDOM ORDER
        # assert_frame_equal(a, b, check_dtype=False)

    def test_from_import_where_operation_sub(self):
        #  where -
        a = (self.odf_d[self.odf_d["OPENPRC"] > 40] - self.odf_d[self.odf_d["SHROUT"] > 15000]).to_pandas()
        b = self.pdf_d[self.pdf_d["OPENPRC"] > 40] - self.pdf_d[self.pdf_d["SHROUT"] > 15000]
        # TODO: ASSERT RANDOM ORDER
        # assert_frame_equal(a, b, check_dtype=False)

    def test_from_import_where_operation_mul(self):
        #  where *
        a = (self.odf_d[self.odf_d["OPENPRC"] > 40] * self.odf_d[self.odf_d["SHROUT"] > 15000]).to_pandas()
        b = self.pdf_d[self.pdf_d["OPENPRC"] > 40] * self.pdf_d[self.pdf_d["SHROUT"] > 15000]
        # TODO: ASSERT RANDOM ORDER
        # assert_frame_equal(a, b, check_dtype=False)

    def test_from_import_where_operation_div(self):
        # where /
        # note that operation div is only allowed for numbers, exclusive of temporal, literal and logical
        # columns in Orca.
        a = (self.odf_d[self.odf_d["OPENPRC"] > 40] / self.odf_d[self.odf_d["SHROUT"] > 15000]).to_pandas()
        b = self.pdf_d[self.pdf_d["OPENPRC"] > 40] / self.pdf_d[self.pdf_d["SHROUT"] > 15000]
        # TODO: ASSERT RANDOM ORDER
        # assert_frame_equal(a, b, check_dtype=False)

    def test_from_import_where_operation_groupby(self):
        #  where groupby sum
        a = self.odf_csv[self.odf_csv["OPENPRC"] > 40].groupby("date").sum().to_pandas()
        b = self.pdf_csv[self.pdf_csv["OPENPRC"] > 40].groupby("date").sum()
        assert_frame_equal(a, b, check_dtype=False)

    def test_from_import_where_operation_resample(self):
        #  where groupby sum
        a = self.odf_re[self.odf_re["OPENPRC"] > 40]
        b = self.pdf_re[self.pdf_re["OPENPRC"] > 40]
        assert_frame_equal(a.to_pandas().iloc[:, 0:8], b.iloc[:, 0:8], check_dtype=False, check_index_type=False)
        assert_frame_equal(a.to_pandas().iloc[:, 9:], b.iloc[:, 9:], check_dtype=False, check_index_type=False)
        a = self.odf_re[self.odf_re["OPENPRC"] > 40].resample("d", on="date").sum().to_pandas().sort_index()
        b = self.pdf_re[self.pdf_re["OPENPRC"] > 40].resample("d", on="date").sum().sort_index()
        # TODO: FILTER.RESAMPLE
        # assert_frame_equal(a, b, check_dtype=False, check_index_type=False, check_less_precise=1)

    def test_from_import_where_operation_rolling(self):
        #  where groupby sum
        a = self.odf_re[self.odf_re["OPENPRC"] > 40].rolling(window=2, on="date").sum().to_pandas()
        b = self.pdf_re[self.pdf_re["OPENPRC"] > 40].rolling(window=2, on="date").sum()
        # TODO: ASSERT RANDOM ORDER
        # assert_frame_equal(a, b, check_dtype=False, check_index_type=False, check_less_precise=1)


if __name__ == '__main__':
    unittest.main()
