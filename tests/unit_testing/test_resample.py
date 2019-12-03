import unittest
import orca
import os.path as path
from setup.settings import *
from pandas.util.testing import *
from pandas.tseries.offsets import *


class Csv:
    pdf_csv = None
    odf_csv = None


class ResampleTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # configure data directory
        DATA_DIR = path.abspath(path.join(__file__, "../setup/data"))
        fileName = 'allTypesOfColumns.csv'
        data = os.path.join(DATA_DIR, fileName)
        data = data.replace('\\', '/')

        # Orca connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

        Csv.pdf_csv = pd.read_csv(data, parse_dates=[1, 2, 3, 4, 5, 6, 7, 8, 9])
        Csv.odf_csv = orca.read_csv(data,
                                    dtype={"date": 'DATE', "tstring": "STRING", "tsymbol": "SYMBOL", "tbool": "BOOL"})

    @property
    def pdf_csv(self):
        return Csv.pdf_csv

    @property
    def odf_csv(self):
        return Csv.odf_csv

    @property
    def pdf(self):
        n = 1000  # note that n should be a multiple of 10
        re = n / 10
        pdf = pd.DataFrame({'id': np.arange(1, n + 1, 1, dtype='int32'),
                            'month': np.repeat(pd.date_range('2019.08.01', periods=10, freq='M'), re),
                            'date': np.repeat(pd.date_range('2019.08.01', periods=10, freq='D'), re),
                            'hour': np.repeat(pd.date_range('2019.08.01', periods=10, freq='H'), re),
                            'minute': np.repeat(pd.date_range('2019.08.01', periods=10, freq='T'), re),
                            'second': np.repeat(pd.date_range('2019.08.01', periods=10, freq='S'), re),
                            'tbool': np.repeat(np.arange(0, 2, 1, dtype='bool'), re * 5),
                            'tchar': np.repeat(np.arange(1, 11, 1, dtype='int8'), re),
                            'tshort': np.repeat(np.arange(1, 11, 1, dtype='int16'), re),
                            'tint': np.repeat(np.arange(1, 11, 1, dtype='int32'), re),
                            'tlong': np.repeat(np.arange(1, 11, 1, dtype='int64'), re),
                            'tfloat': np.repeat(np.arange(1, 11, 1, dtype='float32'), re),
                            'tdouble': np.repeat(np.arange(1, 11, 1, dtype='float64'), re)
                            })
        return pdf

    @property
    def odf(self):
        odf = orca.DataFrame(self.pdf)
        return odf

    def test_resample_allocation_verification(self):
        self.assertIsInstance(self.odf_csv.resample("A", on="date")['date'].count().to_pandas(), Series)
        with self.assertRaises(KeyError):
            self.odf_csv.resample("A", on="date")['hello'].count()
        with self.assertRaises(KeyError):
            self.odf_csv.resample("A", on="date")[['dare', 5, 0]].count()
        with self.assertRaises(KeyError):
            self.odf_csv.resample("A", on="date")[['hello', 'world']].count()
        with self.assertRaises(KeyError):
            self.odf_csv.resample("A", on="date")[np.array([1, 2, 3])].count()
        # TODO: orca 需要支持resample之后取下标时，下标传入一个类型为orca.Series的参数
        # assert_frame_equal(self.odf_csv.resample("A", on="date")[orca.Series(["month", "date"])].count().to_pandas().reset_index(drop=True),
        #                    self.pdf_csv.resample("A", on="date")[pd.Series(["month", "date"])].count().reset_index(drop=True))

    def test_from_pandas_series_resample_param_rule_date_param_on_date(self):
        os = orca.Series(np.repeat(1, 10000), index=orca.date_range("20090101", periods=10000, freq="D"),
                         name="resample")
        ps = pd.Series(np.repeat(1, 10000), index=pd.date_range("20090101", periods=10000, freq="D"), name="resample")

        assert_series_equal(os.resample("B").count().to_pandas(), ps.resample("B").count(), check_dtype=False)
        assert_series_equal(os.resample("B").mean().to_pandas(), ps.resample("B").mean(), check_dtype=False)
        assert_series_equal(os.resample("B").std().to_pandas(), ps.resample("B").std(), check_dtype=False)
        assert_series_equal(os.resample("B").sum().to_pandas(), ps.resample("B").sum(), check_dtype=False)

        assert_series_equal(os.resample("W").count().to_pandas(), ps.resample("W").count(), check_dtype=False)
        assert_series_equal(os.resample("W").mean().to_pandas(), ps.resample("W").mean(), check_dtype=False)
        assert_series_equal(os.resample("W").std().to_pandas(), ps.resample("W").std(), check_dtype=False)
        assert_series_equal(os.resample("W").sum().to_pandas(), ps.resample("W").sum(), check_dtype=False)

        fq = WeekOfMonth(1)
        # TODO: ORCA dtype='datetime64[ns]', length=330, freq=None)
        #       PANDAS  dtype='datetime64[ns]', length=330, freq='WOM-1MON')
        # assert_series_equal(os.resample(fq).count().to_pandas(), ps.resample(fq).count(), check_dtype=False)
        # assert_series_equal(os.resample(fq).mean().to_pandas(), ps.resample(fq).mean(), check_dtype=False)
        # assert_series_equal(os.resample(fq).std().to_pandas(), ps.resample(fq).std(), check_dtype=False)
        # assert_series_equal(os.resample(fq).sum().to_pandas(), ps.resample(fq).sum(), check_dtype=False)

        fq = LastWeekOfMonth(1)
        assert_series_equal(os.resample(fq).count().to_pandas(), ps.resample(fq).count(), check_dtype=False)
        assert_series_equal(os.resample(fq).mean().to_pandas(), ps.resample(fq).mean(), check_dtype=False)
        assert_series_equal(os.resample(fq).std().to_pandas(), ps.resample(fq).std(), check_dtype=False)
        assert_series_equal(os.resample(fq).sum().to_pandas(), ps.resample(fq).sum(), check_dtype=False)

        assert_series_equal(os.resample("M").count().to_pandas(), ps.resample("M").count(), check_dtype=False)
        assert_series_equal(os.resample("M").mean().to_pandas(), ps.resample("M").mean(), check_dtype=False)
        assert_series_equal(os.resample("M").std().to_pandas(), ps.resample("M").std(), check_dtype=False)
        assert_series_equal(os.resample("M").sum().to_pandas(), ps.resample("M").sum(), check_dtype=False)

        assert_series_equal(os.resample("3M").count().to_pandas(), ps.resample("3M").count(), check_dtype=False)
        assert_series_equal(os.resample("3M").mean().to_pandas(), ps.resample("3M").mean(), check_dtype=False)
        assert_series_equal(os.resample("3M").std().to_pandas(), ps.resample("3M").std(), check_dtype=False)
        assert_series_equal(os.resample("3M").sum().to_pandas(), ps.resample("3M").sum(), check_dtype=False)

        assert_series_equal(os.resample("MS").count().to_pandas(), ps.resample("MS").count(), check_dtype=False)
        assert_series_equal(os.resample("MS").mean().to_pandas(), ps.resample("MS").mean(), check_dtype=False)
        assert_series_equal(os.resample("MS").std().to_pandas(), ps.resample("MS").std(), check_dtype=False)
        assert_series_equal(os.resample("MS").sum().to_pandas(), ps.resample("MS").sum(), check_dtype=False)

        assert_series_equal(os.resample("3MS").count().to_pandas(), ps.resample("3MS").count(), check_dtype=False)
        assert_series_equal(os.resample("3MS").mean().to_pandas(), ps.resample("3MS").mean(), check_dtype=False)
        assert_series_equal(os.resample("3MS").std().to_pandas(), ps.resample("3MS").std(), check_dtype=False)
        assert_series_equal(os.resample("3MS").sum().to_pandas(), ps.resample("3MS").sum(), check_dtype=False)

        # TODO： ASSERT FAIL
        # assert_series_equal(os.resample("BM").count().to_pandas(), ps.resample("BM").count(), check_dtype=False)
        # assert_series_equal(os.resample("BM").mean().to_pandas(), ps.resample("BM").mean(), check_dtype=False)
        # assert_series_equal(os.resample("BM").std().to_pandas(), ps.resample("BM").std(), check_dtype=False)
        # assert_series_equal(os.resample("BM").sum().to_pandas(), ps.resample("BM").sum(), check_dtype=False)
        #
        # assert_series_equal(os.resample("3BM").count().to_pandas(), ps.resample("3BM").count(), check_dtype=False)
        # assert_series_equal(os.resample("3BM").mean().to_pandas(), ps.resample("3BM").mean(), check_dtype=False)
        # assert_series_equal(os.resample("3BM").std().to_pandas(), ps.resample("3BM").std(), check_dtype=False)
        # assert_series_equal(os.resample("3BM").sum().to_pandas(), ps.resample("3BM").sum(), check_dtype=False)

        # assert_series_equal(os.resample("BMS").count().to_pandas(), ps.resample("BMS").count(), check_dtype=False)
        # assert_series_equal(os.resample("BMS").mean().to_pandas(), ps.resample("BMS").mean(), check_dtype=False)
        # assert_series_equal(os.resample("BMS").std().to_pandas(), ps.resample("BMS").std(), check_dtype=False)
        # assert_series_equal(os.resample("BMS").sum().to_pandas(), ps.resample("BMS").sum(), check_dtype=False)
        #
        # assert_series_equal(os.resample("3BMS").count().to_pandas(), ps.resample("3BMS").count(), check_dtype=False)
        # assert_series_equal(os.resample("3BMS").mean().to_pandas(), ps.resample("3BMS").mean(), check_dtype=False)
        # assert_series_equal(os.resample("3BMS").std().to_pandas(), ps.resample("3BMS").std(), check_dtype=False)
        # assert_series_equal(os.resample("3BMS").sum().to_pandas(), ps.resample("3BMS").sum(), check_dtype=False)

        assert_series_equal(os.resample("SM").count().to_pandas(), ps.resample("SM").count(), check_dtype=False)
        assert_series_equal(os.resample("SM").mean().to_pandas(), ps.resample("SM").mean(), check_dtype=False)
        assert_series_equal(os.resample("SM").std().to_pandas(), ps.resample("SM").std(), check_dtype=False)
        assert_series_equal(os.resample("SM").sum().to_pandas(), ps.resample("SM").sum(), check_dtype=False)

        assert_series_equal(os.resample("3SM").count().to_pandas(), ps.resample("3SM").count(), check_dtype=False)
        assert_series_equal(os.resample("3SM").mean().to_pandas(), ps.resample("3SM").mean(), check_dtype=False)
        assert_series_equal(os.resample("3SM").std().to_pandas(), ps.resample("3SM").std(), check_dtype=False)
        assert_series_equal(os.resample("3SM").sum().to_pandas(), ps.resample("3SM").sum(), check_dtype=False)

        assert_series_equal(os.resample("SMS").count().to_pandas(), ps.resample("SMS").count(), check_dtype=False)
        assert_series_equal(os.resample("SMS").mean().to_pandas(), ps.resample("SMS").mean(), check_dtype=False)
        assert_series_equal(os.resample("SMS").std().to_pandas(), ps.resample("SMS").std(), check_dtype=False)
        assert_series_equal(os.resample("SMS").sum().to_pandas(), ps.resample("SMS").sum(), check_dtype=False)

        assert_series_equal(os.resample("3SMS").count().to_pandas(), ps.resample("3SMS").count(), check_dtype=False)
        assert_series_equal(os.resample("3SMS").mean().to_pandas(), ps.resample("3SMS").mean(), check_dtype=False)
        assert_series_equal(os.resample("3SMS").std().to_pandas(), ps.resample("3SMS").std(), check_dtype=False)
        assert_series_equal(os.resample("3SMS").sum().to_pandas(), ps.resample("3SMS").sum(), check_dtype=False)

        assert_series_equal(os.resample("Q").count().to_pandas(), ps.resample("Q").count(), check_dtype=False)
        assert_series_equal(os.resample("Q").mean().to_pandas(), ps.resample("Q").mean(), check_dtype=False)
        assert_series_equal(os.resample("Q").std().to_pandas(), ps.resample("Q").std(), check_dtype=False)
        assert_series_equal(os.resample("Q").sum().to_pandas(), ps.resample("Q").sum(), check_dtype=False)

        assert_series_equal(os.resample("3Q").count().to_pandas(), ps.resample("3Q").count(), check_dtype=False)
        assert_series_equal(os.resample("3Q").mean().to_pandas(), ps.resample("3Q").mean(), check_dtype=False)
        assert_series_equal(os.resample("3Q").std().to_pandas(), ps.resample("3Q").std(), check_dtype=False)
        assert_series_equal(os.resample("3Q").sum().to_pandas(), ps.resample("3Q").sum(), check_dtype=False)

        assert_series_equal(os.resample("QS").count().to_pandas(), ps.resample("QS").count(), check_dtype=False)
        assert_series_equal(os.resample("QS").mean().to_pandas(), ps.resample("QS").mean(), check_dtype=False)
        assert_series_equal(os.resample("QS").std().to_pandas(), ps.resample("QS").std(), check_dtype=False)
        assert_series_equal(os.resample("QS").sum().to_pandas(), ps.resample("QS").sum(), check_dtype=False)

        assert_series_equal(os.resample("3QS").count().to_pandas(), ps.resample("3QS").count(), check_dtype=False)
        assert_series_equal(os.resample("3QS").mean().to_pandas(), ps.resample("3QS").mean(), check_dtype=False)
        assert_series_equal(os.resample("3QS").std().to_pandas(), ps.resample("3QS").std(), check_dtype=False)
        assert_series_equal(os.resample("3QS").sum().to_pandas(), ps.resample("3QS").sum(), check_dtype=False)

        # TODO： ASSERT FAIL
        # assert_series_equal(os.resample("BQ").count().to_pandas(), ps.resample("BQ").count(), check_dtype=False)
        # assert_series_equal(os.resample("BQ").mean().to_pandas(), ps.resample("BQ").mean(), check_dtype=False)
        # assert_series_equal(os.resample("BQ").std().to_pandas(), ps.resample("BQ").std(), check_dtype=False)
        # assert_series_equal(os.resample("BQ").sum().to_pandas(), ps.resample("BQ").sum(), check_dtype=False)
        #
        # assert_series_equal(os.resample("3BQ").count().to_pandas(), ps.resample("3BQ").count(), check_dtype=False)
        # assert_series_equal(os.resample("3BQ").mean().to_pandas(), ps.resample("3BQ").mean(), check_dtype=False)
        # assert_series_equal(os.resample("3BQ").std().to_pandas(), ps.resample("3BQ").std(), check_dtype=False)
        # assert_series_equal(os.resample("3BQ").sum().to_pandas(), ps.resample("3BQ").sum(), check_dtype=False)

        # assert_series_equal(os.resample("BQS").count().to_pandas(), ps.resample("BQS").count(), check_dtype=False)
        # assert_series_equal(os.resample("BQS").mean().to_pandas(), ps.resample("BQS").mean(), check_dtype=False)
        # assert_series_equal(os.resample("BQS").std().to_pandas(), ps.resample("BQS").std(), check_dtype=False)
        # assert_series_equal(os.resample("BQS").sum().to_pandas(), ps.resample("BQS").sum(), check_dtype=False)
        #
        # assert_series_equal(os.resample("3BQS").count().to_pandas(), ps.resample("3BQS").count(), check_dtype=False)
        # assert_series_equal(os.resample("3BQS").mean().to_pandas(), ps.resample("3BQS").mean(), check_dtype=False)
        # assert_series_equal(os.resample("3BQS").std().to_pandas(), ps.resample("3BQS").std(), check_dtype=False)
        # assert_series_equal(os.resample("3BQS").sum().to_pandas(), ps.resample("3BQS").sum(), check_dtype=False)

        fq = FY5253Quarter(1)
        assert_series_equal(os.resample(fq).count().to_pandas(), ps.resample(fq).count(), check_dtype=False)
        assert_series_equal(os.resample(fq).mean().to_pandas(), ps.resample(fq).mean(), check_dtype=False)
        assert_series_equal(os.resample(fq).std().to_pandas(), ps.resample(fq).std(), check_dtype=False)
        assert_series_equal(os.resample(fq).sum().to_pandas(), ps.resample(fq).sum(), check_dtype=False)

        fq = FY5253Quarter(3)
        assert_series_equal(os.resample(fq).count().to_pandas(), ps.resample(fq).count(), check_dtype=False)
        assert_series_equal(os.resample(fq).mean().to_pandas(), ps.resample(fq).mean(), check_dtype=False)
        assert_series_equal(os.resample(fq).std().to_pandas(), ps.resample(fq).std(), check_dtype=False)
        assert_series_equal(os.resample(fq).sum().to_pandas(), ps.resample(fq).sum(), check_dtype=False)

        assert_series_equal(os.resample("A").count().to_pandas(), ps.resample("A").count(), check_dtype=False)
        assert_series_equal(os.resample("A").mean().to_pandas(), ps.resample("A").mean(), check_dtype=False)
        assert_series_equal(os.resample("A").std().to_pandas(), ps.resample("A").std(), check_dtype=False)
        assert_series_equal(os.resample("A").sum().to_pandas(), ps.resample("A").sum(), check_dtype=False)

        assert_series_equal(os.resample("3A").count().to_pandas(), ps.resample("3A").count(), check_dtype=False)
        assert_series_equal(os.resample("3A").mean().to_pandas(), ps.resample("3A").mean(), check_dtype=False)
        assert_series_equal(os.resample("3A").std().to_pandas(), ps.resample("3A").std(), check_dtype=False)
        assert_series_equal(os.resample("3A").sum().to_pandas(), ps.resample("3A").sum(), check_dtype=False)

        assert_series_equal(os.resample("AS").count().to_pandas(), ps.resample("AS").count(), check_dtype=False)
        assert_series_equal(os.resample("AS").mean().to_pandas(), ps.resample("AS").mean(), check_dtype=False)
        assert_series_equal(os.resample("AS").std().to_pandas(), ps.resample("AS").std(), check_dtype=False)
        assert_series_equal(os.resample("AS").sum().to_pandas(), ps.resample("AS").sum(), check_dtype=False)

        assert_series_equal(os.resample("3AS").count().to_pandas(), ps.resample("3AS").count(), check_dtype=False)
        assert_series_equal(os.resample("3AS").mean().to_pandas(), ps.resample("3AS").mean(), check_dtype=False)
        assert_series_equal(os.resample("3AS").std().to_pandas(), ps.resample("3AS").std(), check_dtype=False)
        assert_series_equal(os.resample("3AS").sum().to_pandas(), ps.resample("3AS").sum(), check_dtype=False)

        assert_series_equal(os.resample("BA").count().to_pandas(), ps.resample("BA").count(), check_dtype=False)
        assert_series_equal(os.resample("BA").mean().to_pandas(), ps.resample("BA").mean(), check_dtype=False)
        assert_series_equal(os.resample("BA").std().to_pandas(), ps.resample("BA").std(), check_dtype=False)
        assert_series_equal(os.resample("BA").sum().to_pandas(), ps.resample("BA").sum(), check_dtype=False)

        assert_series_equal(os.resample("3BA").count().to_pandas(), ps.resample("3BA").count(), check_dtype=False)
        assert_series_equal(os.resample("3BA").mean().to_pandas(), ps.resample("3BA").mean(), check_dtype=False)
        assert_series_equal(os.resample("3BA").std().to_pandas(), ps.resample("3BA").std(), check_dtype=False)
        assert_series_equal(os.resample("3BA").sum().to_pandas(), ps.resample("3BA").sum(), check_dtype=False)

        assert_series_equal(os.resample("BAS").count().to_pandas(), ps.resample("BAS").count(), check_dtype=False)
        assert_series_equal(os.resample("BAS").mean().to_pandas(), ps.resample("BAS").mean(), check_dtype=False)
        assert_series_equal(os.resample("BAS").std().to_pandas(), ps.resample("BAS").std(), check_dtype=False)
        assert_series_equal(os.resample("BAS").sum().to_pandas(), ps.resample("BAS").sum(), check_dtype=False)

        assert_series_equal(os.resample("3BAS").count().to_pandas(), ps.resample("3BAS").count(), check_dtype=False)
        assert_series_equal(os.resample("3BAS").mean().to_pandas(), ps.resample("3BAS").mean(), check_dtype=False)
        assert_series_equal(os.resample("3BAS").std().to_pandas(), ps.resample("3BAS").std(), check_dtype=False)
        assert_series_equal(os.resample("3BAS").sum().to_pandas(), ps.resample("3BAS").sum(), check_dtype=False)

        # TODO：BUG ValueError: Unsupported offset name RE-N-JAN-MON
        fq = FY5253(1)
        assert_series_equal(os.resample(fq).count().to_pandas(), ps.resample(fq).count(), check_dtype=False)
        assert_series_equal(os.resample(fq).mean().to_pandas(), ps.resample(fq).mean(), check_dtype=False)
        assert_series_equal(os.resample(fq).std().to_pandas(), ps.resample(fq).std(), check_dtype=False)
        assert_series_equal(os.resample(fq).sum().to_pandas(), ps.resample(fq).sum(), check_dtype=False)

        fq = FY5253(3)
        assert_series_equal(os.resample(fq).count().to_pandas(), ps.resample(fq).count(), check_dtype=False)
        assert_series_equal(os.resample(fq).mean().to_pandas(), ps.resample(fq).mean(), check_dtype=False)
        assert_series_equal(os.resample(fq).std().to_pandas(), ps.resample(fq).std(), check_dtype=False)
        assert_series_equal(os.resample(fq).sum().to_pandas(), ps.resample(fq).sum(), check_dtype=False)

        assert_series_equal(os.resample("D").count().to_pandas(), ps.resample("D").count(), check_dtype=False)
        assert_series_equal(os.resample("D").mean().to_pandas(), ps.resample("D").mean(), check_dtype=False)
        assert_series_equal(os.resample("D").std().to_pandas(), ps.resample("D").std(), check_dtype=False)
        assert_series_equal(os.resample("D").sum().to_pandas(), ps.resample("D").sum(), check_dtype=False)

        assert_series_equal(os.resample("3D").count().to_pandas(), ps.resample("3D").count(), check_dtype=False)
        assert_series_equal(os.resample("3D").mean().to_pandas(), ps.resample("3D").mean(), check_dtype=False)
        assert_series_equal(os.resample("3D").std().to_pandas(), ps.resample("3D").std(), check_dtype=False)
        assert_series_equal(os.resample("3D").sum().to_pandas(), ps.resample("3D").sum(), check_dtype=False)

        os = orca.Series(np.repeat(1, 360000), index=orca.date_range("20090101", periods=360000, freq="ms"),
                         name="resample")
        ps = pd.Series(np.repeat(1, 360000), index=pd.date_range("20090101", periods=360000, freq="ms"),
                       name="resample")

        # TODO： ORCA ValueError: Unsupported offset name BH
        # assert_series_equal(os.resample("BH").count().to_pandas(), ps.resample("BH").count(), check_dtype=False)
        # assert_series_equal(os.resample("BH").mean().to_pandas(), ps.resample("BH").mean(), check_dtype=False)
        # assert_series_equal(os.resample("BH").std().to_pandas(), ps.resample("BH").std(), check_dtype=False)
        # assert_series_equal(os.resample("BH").sum().to_pandas(), ps.resample("BH").sum(), check_dtype=False)
        #
        # assert_series_equal(os.resample("3BH").count().to_pandas(), ps.resample("3BH").count(), check_dtype=False)
        # assert_series_equal(os.resample("3BH").mean().to_pandas(), ps.resample("3BH").mean(), check_dtype=False)
        # assert_series_equal(os.resample("3BH").std().to_pandas(), ps.resample("3BH").std(), check_dtype=False)
        # assert_series_equal(os.resample("3BH").sum().to_pandas(), ps.resample("3BH").sum(), check_dtype=False)

        assert_series_equal(os.resample("H").count().to_pandas(), ps.resample("H").count(), check_dtype=False)
        assert_series_equal(os.resample("H").mean().to_pandas(), ps.resample("H").mean(), check_dtype=False)
        assert_series_equal(os.resample("H").std().to_pandas(), ps.resample("H").std(), check_dtype=False)
        assert_series_equal(os.resample("H").sum().to_pandas(), ps.resample("H").sum(), check_dtype=False)

        assert_series_equal(os.resample("3H").count().to_pandas(), ps.resample("3H").count(), check_dtype=False)
        assert_series_equal(os.resample("3H").mean().to_pandas(), ps.resample("3H").mean(), check_dtype=False)
        assert_series_equal(os.resample("3H").std().to_pandas(), ps.resample("3H").std(), check_dtype=False)
        assert_series_equal(os.resample("3H").sum().to_pandas(), ps.resample("3H").sum(), check_dtype=False)

        os = orca.Series(np.repeat(1, 2000000), index=orca.date_range("20090101", periods=2000000, freq="ms"), name="resample")
        ps = pd.Series(np.repeat(1, 2000000), index=pd.date_range("20090101", periods=2000000, freq="ms"), name="resample")

        assert_series_equal(os.resample("T").count().to_pandas(), ps.resample("T").count(), check_dtype=False)
        assert_series_equal(os.resample("T").mean().to_pandas(), ps.resample("T").mean(), check_dtype=False)
        assert_series_equal(os.resample("T").std().to_pandas(), ps.resample("T").std(), check_dtype=False)
        assert_series_equal(os.resample("T").sum().to_pandas(), ps.resample("T").sum(), check_dtype=False)

        assert_series_equal(os.resample("3T").count().to_pandas(), ps.resample("3T").count(), check_dtype=False)
        assert_series_equal(os.resample("3T").mean().to_pandas(), ps.resample("3T").mean(), check_dtype=False)
        assert_series_equal(os.resample("3T").std().to_pandas(), ps.resample("3T").std(), check_dtype=False)
        assert_series_equal(os.resample("3T").sum().to_pandas(), ps.resample("3T").sum(), check_dtype=False)

        assert_series_equal(os.resample("S").count().to_pandas(), ps.resample("S").count(), check_dtype=False)
        assert_series_equal(os.resample("S").mean().to_pandas(), ps.resample("S").mean(), check_dtype=False)
        assert_series_equal(os.resample("S").std().to_pandas(), ps.resample("S").std(), check_dtype=False)
        assert_series_equal(os.resample("S").sum().to_pandas(), ps.resample("S").sum(), check_dtype=False)

        assert_series_equal(os.resample("3S").count().to_pandas(), ps.resample("3S").count(), check_dtype=False)
        assert_series_equal(os.resample("3S").mean().to_pandas(), ps.resample("3S").mean(), check_dtype=False)
        assert_series_equal(os.resample("3S").std().to_pandas(), ps.resample("3S").std(), check_dtype=False)
        assert_series_equal(os.resample("3S").sum().to_pandas(), ps.resample("3S").sum(), check_dtype=False)

        assert_series_equal(os.resample("L").count().to_pandas(), ps.resample("L").count(), check_dtype=False)
        assert_series_equal(os.resample("L").mean().to_pandas(), ps.resample("L").mean(), check_dtype=False)
        assert_series_equal(os.resample("L").std().to_pandas(), ps.resample("L").std(), check_dtype=False)
        assert_series_equal(os.resample("L").sum().to_pandas(), ps.resample("L").sum(), check_dtype=False)

        assert_series_equal(os.resample("3L").count().to_pandas(), ps.resample("3L").count(), check_dtype=False)
        assert_series_equal(os.resample("3L").mean().to_pandas(), ps.resample("3L").mean(), check_dtype=False)
        assert_series_equal(os.resample("3L").std().to_pandas(), ps.resample("3L").std(), check_dtype=False)
        assert_series_equal(os.resample("3L").sum().to_pandas(), ps.resample("3L").sum(), check_dtype=False)

        os = orca.Series(np.repeat(1, 2000000), index=orca.date_range("20090101", periods=2000000, freq="N"),
                         name="resample")
        ps = pd.Series(np.repeat(1, 2000000), index=pd.date_range("20090101", periods=2000000, freq="N"),
                       name="resample")

        assert_series_equal(os.resample("U").count().to_pandas(), ps.resample("U").count(), check_dtype=False)
        assert_series_equal(os.resample("U").mean().to_pandas(), ps.resample("U").mean(), check_dtype=False)
        assert_series_equal(os.resample("U").std().to_pandas(), ps.resample("U").std(), check_dtype=False)
        assert_series_equal(os.resample("U").sum().to_pandas(), ps.resample("U").sum(), check_dtype=False)

        assert_series_equal(os.resample("3U").count().to_pandas(), ps.resample("3U").count(), check_dtype=False)
        assert_series_equal(os.resample("3U").mean().to_pandas(), ps.resample("3U").mean(), check_dtype=False)
        assert_series_equal(os.resample("3U").std().to_pandas(), ps.resample("3U").std(), check_dtype=False)
        assert_series_equal(os.resample("3U").sum().to_pandas(), ps.resample("3U").sum(), check_dtype=False)

        assert_series_equal(os.resample("N").count().to_pandas(), ps.resample("N").count(), check_dtype=False)
        assert_series_equal(os.resample("N").mean().to_pandas(), ps.resample("N").mean(), check_dtype=False)
        assert_series_equal(os.resample("N").std().to_pandas(), ps.resample("N").std(), check_dtype=False)
        assert_series_equal(os.resample("N").sum().to_pandas(), ps.resample("N").sum(), check_dtype=False)

        assert_series_equal(os.resample("3N").count().to_pandas(), ps.resample("3N").count(), check_dtype=False)
        assert_series_equal(os.resample("3N").mean().to_pandas(), ps.resample("3N").mean(), check_dtype=False)
        assert_series_equal(os.resample("3N").std().to_pandas(), ps.resample("3N").std(), check_dtype=False)
        assert_series_equal(os.resample("3N").sum().to_pandas(), ps.resample("3N").sum(), check_dtype=False)

    def test_resample(self):
        a = pd.date_range("20080101", periods=1470, freq='d')
        pdf = pd.DataFrame({"date": a, "value": np.repeat(1, 1470)})
        sme = SemiMonthEnd(2, day_of_month=3)
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.resample(sme, on="date").sum().to_pandas(), pdf.resample(sme, on="date").sum())

    def test_from_pandas_dataframe_resample_param_rule_businessday_param_on_date_count(self):
        a = self.odf.resample("B", on="date").count()
        b = self.pdf.resample("B", on="date").count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        self.odf.resample("3B", on="date").count()
        self.pdf.resample("3B", on="date").count()

        with self.assertRaises(TypeError):
            self.odf.resample("B").count()
        with self.assertRaises(TypeError):
            self.pdf.resample("B").count()

        a = self.odf.resample("B", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = self.pdf.resample("B", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("B", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdf_dai.resample("B", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('date')
        pdf_dai = self.pdf.set_index('date')
        a = odf_dai.resample("B")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdf_dai.resample("B")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_businessday_param_on_date_max(self):
        a = self.odf.resample("B", on="date").max()
        b = self.pdf.resample("B", on="date").max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf.resample("B").max()
        with self.assertRaises(TypeError):
            self.pdf.resample("B").max()

        a = self.odf.resample("B", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = self.pdf.resample("B", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("B", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdf_dai.resample("B", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('date')
        pdf_dai = self.pdf.set_index('date')
        a = odf_dai.resample("B")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdf_dai.resample("B")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_businessday_param_on_date_mean(self):
        a = self.odf.resample("B", on="date").mean()
        b = self.pdf.resample("B", on="date").mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf.resample("B").mean()
        with self.assertRaises(TypeError):
            self.pdf.resample("B").mean()

        a = self.odf.resample("B", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = self.pdf.resample("B", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("B", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdf_dai.resample("B", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('date')
        pdf_dai = self.pdf.set_index('date')
        a = odf_dai.resample("B")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdf_dai.resample("B")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_businessday_param_on_date_min(self):
        a = self.odf.resample("B", on="date").min()
        b = self.pdf.resample("B", on="date").min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf.resample("B").min()
        with self.assertRaises(TypeError):
            self.pdf.resample("B").min()

        a = self.odf.resample("B", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = self.pdf.resample("B", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("B", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdf_dai.resample("B", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('date')
        pdf_dai = self.pdf.set_index('date')
        a = odf_dai.resample("B")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdf_dai.resample("B")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_businessday_param_on_date_std(self):
        a = self.odf.resample("B", on="date").std()
        pdf = self.pdf
        pdf.tbool = pdf.tbool.astype('float32')
        b = pdf.resample("B", on="date").std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf.resample("B").std()
        with self.assertRaises(TypeError):
            pdf.resample("B").std()

        a = self.odf.resample("B", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdf.resample("B", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = pdf.set_index('id')
        a = odf_dai.resample("B", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdf_dai.resample("B", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('date')
        pdf_dai = pdf.set_index('date')
        a = odf_dai.resample("B")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdf_dai.resample("B")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_businessday_param_on_date_sum(self):
        a = self.odf.resample("B", on="date").sum()
        b = self.pdf.resample("B", on="date").sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf.resample("B").sum()
        with self.assertRaises(TypeError):
            self.pdf.resample("B").sum()

        a = self.odf.resample("B", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = self.pdf.resample("B", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("B", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdf_dai.resample("B", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('date')
        pdf_dai = self.pdf.set_index('date')
        a = odf_dai.resample("B")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdf_dai.resample("B")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_businessday_param_on_date_var(self):
        a = self.odf.resample("B", on="date").var()
        b = self.pdf.resample("B", on="date").var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf.resample("B").var()
        with self.assertRaises(TypeError):
            self.pdf.resample("B").var()

        a = self.odf.resample("B", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = self.pdf.resample("B", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("B", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdf_dai.resample("B", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('date')
        pdf_dai = self.pdf.set_index('date')
        a = odf_dai.resample("B")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdf_dai.resample("B")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_weekofmonth_param_on_date_count(self):
        fq = WeekOfMonth(1)
        a = self.odf.resample(fq, on="date").count()
        b = self.pdf.resample(fq, on="date").count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)
        fq = WeekOfMonth(2)
        self.odf.resample(fq, on="date").count()
        self.pdf.resample(fq, on="date").count()
        fq = WeekOfMonth(1)
        with self.assertRaises(TypeError):
            self.odf.resample(fq).count()
        with self.assertRaises(TypeError):
            self.pdf.resample(fq).count()

        a = self.odf.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = self.pdf.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdf_dai.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('date')
        pdf_dai = self.pdf.set_index('date')
        a = odf_dai.resample(fq)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdf_dai.resample(fq)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_weekofmonth_param_on_date_max(self):
        fq = WeekOfMonth(1)
        a = self.odf.resample(fq, on="date").max()
        b = self.pdf.resample(fq, on="date").max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)
        fq = WeekOfMonth(2)
        self.odf.resample(fq, on="date").max()
        self.pdf.resample(fq, on="date").max()
        fq = WeekOfMonth(1)
        with self.assertRaises(TypeError):
            self.odf.resample(fq).max()
        with self.assertRaises(TypeError):
            self.pdf.resample(fq).max()

        a = self.odf.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = self.pdf.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdf_dai.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('date')
        pdf_dai = self.pdf.set_index('date')
        a = odf_dai.resample(fq)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdf_dai.resample(fq)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_weekofmonth_param_on_date_mean(self):
        fq = WeekOfMonth(1)
        a = self.odf.resample(fq, on="date").mean()
        b = self.pdf.resample(fq, on="date").mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)
        fq = WeekOfMonth(2)
        self.odf.resample(fq, on="date").mean()
        self.pdf.resample(fq, on="date").mean()
        fq = WeekOfMonth(1)
        with self.assertRaises(TypeError):
            self.odf.resample(fq).mean()
        with self.assertRaises(TypeError):
            self.pdf.resample(fq).mean()

        a = self.odf.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = self.pdf.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdf_dai.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('date')
        pdf_dai = self.pdf.set_index('date')
        a = odf_dai.resample(fq)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdf_dai.resample(fq)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_weekofmonth_param_on_date_min(self):
        fq = WeekOfMonth(1)
        a = self.odf.resample(fq, on="date").min()
        b = self.pdf.resample(fq, on="date").min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)
        fq = WeekOfMonth(2)
        self.odf.resample(fq, on="date").min()
        self.pdf.resample(fq, on="date").min()
        fq = WeekOfMonth(1)
        with self.assertRaises(TypeError):
            self.odf.resample(fq).min()
        with self.assertRaises(TypeError):
            self.pdf.resample(fq).min()

        a = self.odf.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = self.pdf.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdf_dai.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('date')
        pdf_dai = self.pdf.set_index('date')
        a = odf_dai.resample(fq)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdf_dai.resample(fq)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_weekofmonth_param_on_date_std(self):
        fq = WeekOfMonth(1)
        a = self.odf.resample(fq, on="date").std()
        b = self.pdf.resample(fq, on="date").std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)
        fq = WeekOfMonth(2)
        self.odf.resample(fq, on="date").std()
        self.pdf.resample(fq, on="date").std()
        fq = WeekOfMonth(1)
        with self.assertRaises(TypeError):
            self.odf.resample(fq).std()
        with self.assertRaises(TypeError):
            self.pdf.resample(fq).std()

        a = self.odf.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = self.pdf.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdf_dai.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('date')
        pdf_dai = self.pdf.set_index('date')
        a = odf_dai.resample(fq)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdf_dai.resample(fq)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_weekofmonth_param_on_date_sum(self):
        fq = WeekOfMonth(1)
        a = self.odf.resample(fq, on="date").sum()
        b = self.pdf.resample(fq, on="date").sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)
        fq = WeekOfMonth(2)
        self.odf.resample(fq, on="date").sum()
        self.pdf.resample(fq, on="date").sum()
        fq = WeekOfMonth(1)
        with self.assertRaises(TypeError):
            self.odf.resample(fq).sum()
        with self.assertRaises(TypeError):
            self.pdf.resample(fq).sum()

        a = self.odf.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = self.pdf.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdf_dai.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('date')
        pdf_dai = self.pdf.set_index('date')
        a = odf_dai.resample(fq)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdf_dai.resample(fq)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_weekofmonth_param_on_date_var(self):
        fq = WeekOfMonth(1)
        a = self.odf.resample(fq, on="date").var()
        b = self.pdf.resample(fq, on="date").var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)
        fq = WeekOfMonth(2)
        self.odf.resample(fq, on="date").var()
        self.pdf.resample(fq, on="date").var()
        fq = WeekOfMonth(1)
        with self.assertRaises(TypeError):
            self.odf.resample(fq).var()
        with self.assertRaises(TypeError):
            self.pdf.resample(fq).var()

        a = self.odf.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = self.pdf.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdf_dai.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('date')
        pdf_dai = self.pdf.set_index('date')
        a = odf_dai.resample(fq)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdf_dai.resample(fq)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_lastweekofmonth_param_on_date_count(self):
        fq = LastWeekOfMonth(1)
        a = self.odf.resample(fq, on="date").count()
        b = self.pdf.resample(fq, on="date").count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)
        fq = LastWeekOfMonth(2)
        self.odf.resample(fq, on="date").count()
        self.pdf.resample(fq, on="date").count()
        fq = LastWeekOfMonth(1)
        with self.assertRaises(TypeError):
            self.odf.resample(fq).count()
        with self.assertRaises(TypeError):
            self.pdf.resample(fq).count()

        a = self.odf.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = self.pdf.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdf_dai.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('date')
        pdf_dai = self.pdf.set_index('date')
        a = odf_dai.resample(fq)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdf_dai.resample(fq)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_lastweekofmonth_param_on_date_max(self):
        fq = LastWeekOfMonth(1)
        a = self.odf.resample(fq, on="date").max()
        b = self.pdf.resample(fq, on="date").max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)
        fq = LastWeekOfMonth(2)
        self.odf.resample(fq, on="date").max()
        self.pdf.resample(fq, on="date").max()
        fq = LastWeekOfMonth(1)
        with self.assertRaises(TypeError):
            self.odf.resample(fq).max()
        with self.assertRaises(TypeError):
            self.pdf.resample(fq).max()

        a = self.odf.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = self.pdf.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdf_dai.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('date')
        pdf_dai = self.pdf.set_index('date')
        a = odf_dai.resample(fq)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdf_dai.resample(fq)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_lastweekofmonth_param_on_date_mean(self):
        fq = LastWeekOfMonth(1)
        a = self.odf.resample(fq, on="date").mean()
        b = self.pdf.resample(fq, on="date").mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)
        fq = LastWeekOfMonth(2)
        self.odf.resample(fq, on="date").mean()
        self.pdf.resample(fq, on="date").mean()
        fq = LastWeekOfMonth(1)
        with self.assertRaises(TypeError):
            self.odf.resample(fq).mean()
        with self.assertRaises(TypeError):
            self.pdf.resample(fq).mean()

        a = self.odf.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = self.pdf.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdf_dai.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('date')
        pdf_dai = self.pdf.set_index('date')
        a = odf_dai.resample(fq)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdf_dai.resample(fq)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_lastweekofmonth_param_on_date_min(self):
        fq = LastWeekOfMonth(1)
        a = self.odf.resample(fq, on="date").min()
        b = self.pdf.resample(fq, on="date").min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)
        fq = LastWeekOfMonth(2)
        self.odf.resample(fq, on="date").min()
        self.pdf.resample(fq, on="date").min()
        fq = LastWeekOfMonth(1)
        with self.assertRaises(TypeError):
            self.odf.resample(fq).min()
        with self.assertRaises(TypeError):
            self.pdf.resample(fq).min()

        a = self.odf.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = self.pdf.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdf_dai.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('date')
        pdf_dai = self.pdf.set_index('date')
        a = odf_dai.resample(fq)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdf_dai.resample(fq)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_lastweekofmonth_param_on_date_std(self):
        fq = LastWeekOfMonth(1)
        a = self.odf.resample(fq, on="date").std()
        b = self.pdf.resample(fq, on="date").std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)
        fq = LastWeekOfMonth(2)
        self.odf.resample(fq, on="date").std()
        self.pdf.resample(fq, on="date").std()
        fq = LastWeekOfMonth(1)
        with self.assertRaises(TypeError):
            self.odf.resample(fq).std()
        with self.assertRaises(TypeError):
            self.pdf.resample(fq).std()

        a = self.odf.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = self.pdf.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdf_dai.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('date')
        pdf_dai = self.pdf.set_index('date')
        a = odf_dai.resample(fq)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdf_dai.resample(fq)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_lastweekofmonth_param_on_date_sum(self):
        fq = LastWeekOfMonth(1)
        a = self.odf.resample(fq, on="date").sum()
        b = self.pdf.resample(fq, on="date").sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)
        fq = LastWeekOfMonth(2)
        self.odf.resample(fq, on="date").sum()
        self.pdf.resample(fq, on="date").sum()
        fq = LastWeekOfMonth(1)
        with self.assertRaises(TypeError):
            self.odf.resample(fq).sum()
        with self.assertRaises(TypeError):
            self.pdf.resample(fq).sum()

        a = self.odf.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = self.pdf.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdf_dai.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('date')
        pdf_dai = self.pdf.set_index('date')
        a = odf_dai.resample(fq)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdf_dai.resample(fq)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_lastweekofmonth_param_on_date_var(self):
        fq = LastWeekOfMonth(1)
        a = self.odf.resample(fq, on="date").var()
        b = self.pdf.resample(fq, on="date").var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)
        fq = LastWeekOfMonth(2)
        self.odf.resample(fq, on="date").var()
        self.pdf.resample(fq, on="date").var()
        fq = LastWeekOfMonth(1)
        with self.assertRaises(TypeError):
            self.odf.resample(fq).var()
        with self.assertRaises(TypeError):
            self.pdf.resample(fq).var()

        a = self.odf.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = self.pdf.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdf_dai.resample(fq, on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('date')
        pdf_dai = self.pdf.set_index('date')
        a = odf_dai.resample(fq)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdf_dai.resample(fq)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_monthbegin_param_on_month_count(self):
        fq = MonthBegin(1)
        a = self.odf.resample(fq, on="month").count()
        b = self.pdf.resample(fq, on="month").count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        fq = MonthBegin(2)
        self.odf.resample(fq, on="month").count()
        self.pdf.resample(fq, on="month").count()
        fq = MonthBegin(1)
        with self.assertRaises(TypeError):
            self.odf.resample(fq).count()
        with self.assertRaises(TypeError):
            self.pdf.resample(fq).count()

        a = self.odf.resample(fq, on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = self.pdf.resample(fq, on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample(fq, on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdf_dai.resample(fq, on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('month')
        pdf_dai = self.pdf.set_index('month')
        a = odf_dai.resample(fq)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdf_dai.resample(fq)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_monthbegin_param_on_month_max(self):
        fq = MonthBegin(1)
        a = self.odf.resample(fq, on="month").max()
        b = self.pdf.resample(fq, on="month").max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        fq = MonthBegin(2)
        self.odf.resample(fq, on="month").max()
        self.pdf.resample(fq, on="month").max()
        fq = MonthBegin(1)
        with self.assertRaises(TypeError):
            self.odf.resample(fq).max()
        with self.assertRaises(TypeError):
            self.pdf.resample(fq).max()

        a = self.odf.resample(fq, on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = self.pdf.resample(fq, on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample(fq, on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdf_dai.resample(fq, on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('month')
        pdf_dai = self.pdf.set_index('month')
        a = odf_dai.resample(fq)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdf_dai.resample(fq)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_monthbegin_param_on_month_mean(self):
        fq = MonthBegin(1)
        a = self.odf.resample(fq, on="month").mean()
        b = self.pdf.resample(fq, on="month").mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        fq = MonthBegin(2)
        self.odf.resample(fq, on="month").mean()
        self.pdf.resample(fq, on="month").mean()
        fq = MonthBegin(1)
        with self.assertRaises(TypeError):
            self.odf.resample(fq).mean()
        with self.assertRaises(TypeError):
            self.pdf.resample(fq).mean()

        a = self.odf.resample(fq, on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = self.pdf.resample(fq, on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample(fq, on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdf_dai.resample(fq, on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('month')
        pdf_dai = self.pdf.set_index('month')
        a = odf_dai.resample(fq)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdf_dai.resample(fq)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_monthbegin_param_on_month_min(self):
        fq = MonthBegin(1)
        a = self.odf.resample(fq, on="month").min()
        b = self.pdf.resample(fq, on="month").min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        fq = MonthBegin(2)
        self.odf.resample(fq, on="month").min()
        self.pdf.resample(fq, on="month").min()
        fq = MonthBegin(1)
        with self.assertRaises(TypeError):
            self.odf.resample(fq).min()
        with self.assertRaises(TypeError):
            self.pdf.resample(fq).min()

        a = self.odf.resample(fq, on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = self.pdf.resample(fq, on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample(fq, on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdf_dai.resample(fq, on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('month')
        pdf_dai = self.pdf.set_index('month')
        a = odf_dai.resample(fq)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdf_dai.resample(fq)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_monthbegin_param_on_month_std(self):
        fq = MonthBegin(1)
        a = self.odf.resample(fq, on="month").std()
        pdf = self.pdf
        pdf.tbool = pdf.tbool.astype('float32')
        b = pdf.resample(fq, on="month").std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        fq = MonthBegin(2)
        self.odf.resample(fq, on="month").std()
        pdf.resample(fq, on="month").std()
        fq = MonthBegin(1)
        with self.assertRaises(TypeError):
            self.odf.resample(fq).std()
        with self.assertRaises(TypeError):
            self.pdf.resample(fq).std()

        a = self.odf.resample(fq, on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdf.resample(fq, on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = pdf.set_index('id')
        a = odf_dai.resample(fq, on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdf_dai.resample(fq, on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('month')
        pdf_dai = pdf.set_index('month')
        a = odf_dai.resample(fq)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdf_dai.resample(fq)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_monthbegin_param_on_month_sum(self):
        fq = MonthBegin(1)
        a = self.odf.resample(fq, on="month").sum()
        b = self.pdf.resample(fq, on="month").sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        fq = MonthBegin(2)
        self.odf.resample(fq, on="month").sum()
        self.pdf.resample(fq, on="month").sum()
        fq = MonthBegin(1)
        with self.assertRaises(TypeError):
            self.odf.resample(fq).sum()
        with self.assertRaises(TypeError):
            self.pdf.resample(fq).sum()

        a = self.odf.resample(fq, on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = self.pdf.resample(fq, on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample(fq, on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdf_dai.resample(fq, on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('month')
        pdf_dai = self.pdf.set_index('month')
        a = odf_dai.resample(fq)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdf_dai.resample(fq)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_monthbegin_param_on_month_var(self):
        fq = MonthBegin(1)
        a = self.odf.resample(fq, on="month").var()
        b = self.pdf.resample(fq, on="month").var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        fq = MonthBegin(2)
        self.odf.resample(fq, on="month").var()
        self.pdf.resample(fq, on="month").var()
        fq = MonthBegin(1)
        with self.assertRaises(TypeError):
            self.odf.resample(fq).var()
        with self.assertRaises(TypeError):
            self.pdf.resample(fq).var()

        a = self.odf.resample(fq, on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = self.pdf.resample(fq, on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample(fq, on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdf_dai.resample(fq, on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('month')
        pdf_dai = self.pdf.set_index('month')
        a = odf_dai.resample(fq)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdf_dai.resample(fq)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_businessmonth_param_on_month_count(self):
        a = self.odf.resample("BM", on="month").count()
        b = self.pdf.resample("BM", on="month").count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b[~(b['tint'] == 0)].reset_index(drop=True), check_dtype=False)

        self.odf.resample("3BM", on="month").count()
        self.pdf.resample("3BM", on="month").count()

        with self.assertRaises(TypeError):
            self.odf.resample("BM").count()
        with self.assertRaises(TypeError):
            self.pdf.resample("BM").count()

        a = self.odf.resample("BM", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = self.pdf.resample("BM", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b[~(b['tint'] == 0)].reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("BM", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdf_dai.resample("BM", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b[~(b['tint'] == 0)].reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('month')
        pdf_dai = self.pdf.set_index('month')
        a = odf_dai.resample("BM")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdf_dai.resample("BM")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b[~(b['tint'] == 0)].reset_index(drop=True), check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_businessmonth_param_on_month_max(self):
        a = self.odf.resample("BM", on="month").max()
        b = self.pdf.resample("BM", on="month").max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b[~(b['tint'].isna())].reset_index(drop=True), check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf.resample("BM").max()
        with self.assertRaises(TypeError):
            self.pdf.resample("BM").max()

        a = self.odf.resample("BM", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = self.pdf.resample("BM", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b[~(b['tint'].isna())].reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("BM", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdf_dai.resample("BM", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b[~(b['tint'].isna())].reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('month')
        pdf_dai = self.pdf.set_index('month')
        a = odf_dai.resample("BM")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdf_dai.resample("BM")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b[~(b['tint'].isna())].reset_index(drop=True), check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_businessmonth_param_on_month_mean(self):
        a = self.odf.resample("BM", on="month").mean()
        b = self.pdf.resample("BM", on="month").mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b[~(b['tint'].isna())].reset_index(drop=True), check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf.resample("BM").mean()
        with self.assertRaises(TypeError):
            self.pdf.resample("BM").mean()

        a = self.odf.resample("BM", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = self.pdf.resample("BM", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b[~(b['tint'].isna())].reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("BM", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdf_dai.resample("BM", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b[~(b['tint'].isna())].reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('month')
        pdf_dai = self.pdf.set_index('month')
        a = odf_dai.resample("BM")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdf_dai.resample("BM")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b[~(b['tint'].isna())].reset_index(drop=True), check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_businessmonth_param_on_month_min(self):
        a = self.odf.resample("BM", on="month").min()
        b = self.pdf.resample("BM", on="month").min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b[~(b['tint'].isna())].reset_index(drop=True), check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf.resample("BM").min()
        with self.assertRaises(TypeError):
            self.pdf.resample("BM").min()

        a = self.odf.resample("BM", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = self.pdf.resample("BM", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b[~(b['tint'].isna())].reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("BM", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdf_dai.resample("BM", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b[~(b['tint'].isna())].reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('month')
        pdf_dai = self.pdf.set_index('month')
        a = odf_dai.resample("BM")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdf_dai.resample("BM")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b[~(b['tint'].isna())].reset_index(drop=True), check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_businessmonth_param_on_month_std(self):
        a = self.odf.resample("BM", on="month").std()
        pdf = self.pdf
        pdf.tbool = pdf.tbool.astype('float32')
        b = pdf.resample("BM", on="month").std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b[~(b['tint'].isna())].reset_index(drop=True), check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf.resample("BM").std()
        with self.assertRaises(TypeError):
            pdf.resample("BM").std()

        a = self.odf.resample("BM", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdf.resample("BM", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b[~(b['tint'].isna())].reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = pdf.set_index('id')
        a = odf_dai.resample("BM", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdf_dai.resample("BM", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b[~(b['tint'].isna())].reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('month')
        pdf_dai = pdf.set_index('month')
        a = odf_dai.resample("BM")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdf_dai.resample("BM")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b[~(b['tint'].isna())].reset_index(drop=True), check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_businessmonth_param_on_month_sum(self):
        a = self.odf.resample("BM", on="month").sum()
        b = self.pdf.resample("BM", on="month").sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b[~(b['tint'] == 0)].reset_index(drop=True), check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf.resample("BM").sum()
        with self.assertRaises(TypeError):
            self.pdf.resample("BM").sum()

        a = self.odf.resample("BM", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = self.pdf.resample("BM", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b[~(b['tint'] == 0)].reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("BM", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdf_dai.resample("BM", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b[~(b['tint'] == 0)].reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('month')
        pdf_dai = self.pdf.set_index('month')
        a = odf_dai.resample("BM")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdf_dai.resample("BM")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b[~(b['tint'] == 0)].reset_index(drop=True), check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_businessmonth_param_on_month_var(self):
        a = self.odf.resample("BM", on="month").var()
        b = self.pdf.resample("BM", on="month").var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b[~(b['tint'].isna())].reset_index(drop=True), check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf.resample("BM").var()
        with self.assertRaises(TypeError):
            self.pdf.resample("BM").var()

        a = self.odf.resample("BM", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = self.pdf.resample("BM", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b[~(b['tint'].isna())].reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("BM", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdf_dai.resample("BM", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b[~(b['tint'].isna())].reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('month')
        pdf_dai = self.pdf.set_index('month')
        a = odf_dai.resample("BM")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdf_dai.resample("BM")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b[~(b['tint'].isna())].reset_index(drop=True), check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_year_param_on_month_count(self):
        a = self.odf.resample("A", on="month").count()
        b = self.pdf.resample("A", on="month").count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        self.odf.resample("2A", on="month").count()
        self.pdf.resample("2A", on="month").count()

        with self.assertRaises(TypeError):
            self.odf.resample("A").count()
        with self.assertRaises(TypeError):
            self.pdf.resample("A").count()

        a = self.odf.resample("A", on="month")[
            'month', 'tchar', 'tbool', 'tdouble', 'tshort', 'tint', 'tlong', 'tfloat'].count()
        b = self.pdf.resample("A", on="month")[
            'month', 'tchar', 'tbool', 'tdouble', 'tshort', 'tint', 'tlong', 'tfloat'].count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("A", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdf_dai.resample("A", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('month')
        pdf_dai = self.pdf.set_index('month')
        a = odf_dai.resample("A")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdf_dai.resample("A")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_year_param_on_month_max(self):
        a = self.odf.resample("A", on="month").max()
        b = self.pdf.resample("A", on="month").max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        a = self.odf.resample("2A", on="month").max()
        b = self.pdf.resample("2A", on="month").max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf.resample("A").max()
        with self.assertRaises(TypeError):
            self.pdf.resample("A").max()

        a = self.odf.resample("A", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = self.pdf.resample("A", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("A", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdf_dai.resample("A", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('month')
        pdf_dai = self.pdf.set_index('month')
        a = odf_dai.resample("A")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdf_dai.resample("A")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_year_param_on_month_mean(self):
        a = self.odf.resample("A", on="month").mean()
        b = self.pdf.resample("A", on="month").mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf.resample("A").mean()
        with self.assertRaises(TypeError):
            self.pdf.resample("A").mean()

        a = self.odf.resample("A", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = self.pdf.resample("A", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("A", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdf_dai.resample("A", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('month')
        pdf_dai = self.pdf.set_index('month')
        a = odf_dai.resample("A")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdf_dai.resample("A")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_year_param_on_month_min(self):
        a = self.odf.resample("A", on="month").min()
        b = self.pdf.resample("A", on="month").min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf.resample("A").min()
        with self.assertRaises(TypeError):
            self.pdf.resample("A").min()

        a = self.odf.resample("A", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = self.pdf.resample("A", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("A", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdf_dai.resample("A", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('month')
        pdf_dai = self.pdf.set_index('month')
        a = odf_dai.resample("A")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdf_dai.resample("A")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_year_param_on_month_std(self):
        a = self.odf.resample("A", on="month").std()
        pdf = self.pdf
        pdf.tbool = pdf.tbool.astype('float32')
        b = pdf.resample("A", on="month").std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf.resample("A").std()
        with self.assertRaises(TypeError):
            pdf.resample("A").std()

        a = self.odf.resample("A", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdf.resample("A", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = pdf.set_index('id')
        a = odf_dai.resample("A", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdf_dai.resample("A", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('month')
        pdf_dai = pdf.set_index('month')
        a = odf_dai.resample("A")[
            'tchar', 'tbool', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdf_dai.resample("A")[
            'tchar', 'tbool', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_year_param_on_month_sum(self):
        a = self.odf.resample("A", on="month").sum()
        b = self.pdf.resample("A", on="month").sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf.resample("A").sum()
        with self.assertRaises(TypeError):
            self.pdf.resample("A").sum()

        a = self.odf.resample("A", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = self.pdf.resample("A", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("A", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdf_dai.resample("A", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('month')
        pdf_dai = self.pdf.set_index('month')
        a = odf_dai.resample("A")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdf_dai.resample("A")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_year_param_on_month_var(self):
        a = self.odf.resample("A", on="month").var()
        b = self.pdf.resample("A", on="month").var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf.resample("A").var()
        with self.assertRaises(TypeError):
            self.pdf.resample("A").var()

        a = self.odf.resample("A", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = self.pdf.resample("A", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("A", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdf_dai.resample("A", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('month')
        pdf_dai = self.pdf.set_index('month')
        a = odf_dai.resample("A")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdf_dai.resample("A")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_month_param_on_month_count(self):
        a = self.odf.resample("M", on="month").count()
        b = self.pdf.resample("M", on="month").count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        self.odf.resample("3M", on="month").count()
        self.pdf.resample("3M", on="month").count()

        with self.assertRaises(TypeError):
            self.odf.resample("M").count()
        with self.assertRaises(TypeError):
            self.pdf.resample("M").count()

        a = self.odf.resample("M", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = self.pdf.resample("M", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("M", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdf_dai.resample("M", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('month')
        pdf_dai = self.pdf.set_index('month')
        a = odf_dai.resample("M")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdf_dai.resample("M")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_month_param_on_month_max(self):
        a = self.odf.resample("M", on="month").max()
        b = self.pdf.resample("M", on="month").max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf.resample("M").max()
        with self.assertRaises(TypeError):
            self.pdf.resample("M").max()

        a = self.odf.resample("M", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = self.pdf.resample("M", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("M", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdf_dai.resample("M", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('month')
        pdf_dai = self.pdf.set_index('month')
        a = odf_dai.resample("M")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdf_dai.resample("M")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_month_param_on_month_mean(self):
        a = self.odf.resample("M", on="month").mean()
        b = self.pdf.resample("M", on="month").mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf.resample("M").mean()
        with self.assertRaises(TypeError):
            self.pdf.resample("M").mean()

        a = self.odf.resample("M", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = self.pdf.resample("M", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("M", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdf_dai.resample("M", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('month')
        pdf_dai = self.pdf.set_index('month')
        a = odf_dai.resample("M")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdf_dai.resample("M")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_month_param_on_month_min(self):
        a = self.odf.resample("M", on="month").min()
        b = self.pdf.resample("M", on="month").min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf.resample("M").min()
        with self.assertRaises(TypeError):
            self.pdf.resample("M").min()

        a = self.odf.resample("M", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = self.pdf.resample("M", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("M", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdf_dai.resample("M", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('month')
        pdf_dai = self.pdf.set_index('month')
        a = odf_dai.resample("M")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdf_dai.resample("M")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_month_param_on_month_std(self):
        a = self.odf.resample("M", on="month").std()
        pdf = self.pdf
        pdf.tbool = pdf.tbool.astype('float32')
        b = pdf.resample("M", on="month").std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf.resample("M").std()
        with self.assertRaises(TypeError):
            pdf.resample("M").std()

        a = self.odf.resample("M", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdf.resample("M", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = pdf.set_index('id')
        a = odf_dai.resample("M", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdf_dai.resample("M", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('month')
        pdf_dai = pdf.set_index('month')
        a = odf_dai.resample("M")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdf_dai.resample("M")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_month_param_on_month_sum(self):
        a = self.odf.resample("M", on="month").sum()
        b = self.pdf.resample("M", on="month").sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf.resample("M").sum()
        with self.assertRaises(TypeError):
            self.pdf.resample("M").sum()

        a = self.odf.resample("M", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = self.pdf.resample("M", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("M", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdf_dai.resample("M", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('month')
        pdf_dai = self.pdf.set_index('month')
        a = odf_dai.resample("M")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdf_dai.resample("M")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_month_param_on_month_var(self):
        a = self.odf.resample("M", on="month").var()
        b = self.pdf.resample("M", on="month").var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf.resample("M").var()
        with self.assertRaises(TypeError):
            self.pdf.resample("M").var()

        a = self.odf.resample("M", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = self.pdf.resample("M", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("M", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdf_dai.resample("M", on="month")[
            'month', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('month')
        pdf_dai = self.pdf.set_index('month')
        a = odf_dai.resample("M")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdf_dai.resample("M")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_day_param_on_date_count(self):
        a = self.odf.resample("D", on="date").count()
        b = self.pdf.resample("D", on="date").count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        self.odf.resample("3D", on="date").count()
        self.pdf.resample("3D", on="date").count()

        with self.assertRaises(TypeError):
            self.odf.resample("D").count()
        with self.assertRaises(TypeError):
            self.pdf.resample("D").count()

        a = self.odf.resample("D", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = self.pdf.resample("D", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("D", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdf_dai.resample("D", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('date')
        pdf_dai = self.pdf.set_index('date')
        a = odf_dai.resample("D")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdf_dai.resample("D")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_day_param_on_date_max(self):
        a = self.odf.resample("D", on="date").max()
        b = self.pdf.resample("D", on="date").max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf.resample("D").max()
        with self.assertRaises(TypeError):
            self.pdf.resample("D").max()

        a = self.odf.resample("D", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = self.pdf.resample("D", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("D", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdf_dai.resample("D", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('date')
        pdf_dai = self.pdf.set_index('date')
        a = odf_dai.resample("D")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdf_dai.resample("D")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_day_param_on_date_mean(self):
        a = self.odf.resample("D", on="date").mean()
        b = self.pdf.resample("D", on="date").mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf.resample("D").mean()
        with self.assertRaises(TypeError):
            self.pdf.resample("D").mean()

        a = self.odf.resample("D", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = self.pdf.resample("D", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("D", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdf_dai.resample("D", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('date')
        pdf_dai = self.pdf.set_index('date')
        a = odf_dai.resample("D")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdf_dai.resample("D")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_day_param_on_date_min(self):
        a = self.odf.resample("D", on="date").min()
        b = self.pdf.resample("D", on="date").min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf.resample("D").min()
        with self.assertRaises(TypeError):
            self.pdf.resample("D").min()

        a = self.odf.resample("D", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = self.pdf.resample("D", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("D", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdf_dai.resample("D", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('date')
        pdf_dai = self.pdf.set_index('date')
        a = odf_dai.resample("D")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdf_dai.resample("D")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_day_param_on_date_std(self):
        a = self.odf.resample("D", on="date").std()
        pdf = self.pdf
        pdf.tbool = pdf.tbool.astype('float32')
        b = pdf.resample("D", on="date").std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf.resample("D").std()
        with self.assertRaises(TypeError):
            pdf.resample("D").std()

        a = self.odf.resample("D", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdf.resample("D", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = pdf.set_index('id')
        a = odf_dai.resample("D", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdf_dai.resample("D", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('date')
        pdf_dai = pdf.set_index('date')
        a = odf_dai.resample("D")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdf_dai.resample("D")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_day_param_on_date_sum(self):
        a = self.odf.resample("D", on="date").sum()
        b = self.pdf.resample("D", on="date").sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf.resample("D").sum()
        with self.assertRaises(TypeError):
            self.pdf.resample("D").sum()

        a = self.odf.resample("D", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = self.pdf.resample("D", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("D", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdf_dai.resample("D", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('date')
        pdf_dai = self.pdf.set_index('date')
        a = odf_dai.resample("D")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdf_dai.resample("D")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_day_param_on_date_var(self):
        a = self.odf.resample("D", on="date").var()
        b = self.pdf.resample("D", on="date").var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf.resample("D").var()
        with self.assertRaises(TypeError):
            self.pdf.resample("D").var()

        a = self.odf.resample("D", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = self.pdf.resample("D", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("D", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdf_dai.resample("D", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('date')
        pdf_dai = self.pdf.set_index('date')
        a = odf_dai.resample("D")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdf_dai.resample("D")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_week_param_on_date_count(self):
        a = self.odf.resample("W", on="date").count()
        self.pdf.tbool = self.pdf.tbool.astype('float32')
        b = self.pdf.resample("W", on="date").count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        self.odf.resample("3W", on="date").count()
        self.pdf.resample("3W", on="date").count()

        with self.assertRaises(TypeError):
            self.odf.resample("W").count()
        with self.assertRaises(TypeError):
            self.pdf.resample("W").count()

        a = self.odf.resample("W", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = self.pdf.resample("W", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("W", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdf_dai.resample("W", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('date')
        pdf_dai = self.pdf.set_index('date')
        a = odf_dai.resample("W")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdf_dai.resample("W")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_week_param_on_date_max(self):
        a = self.odf.resample("W", on="date").max()
        b = self.pdf.resample("W", on="date").max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf.resample("W").max()
        with self.assertRaises(TypeError):
            self.pdf.resample("W").max()

        a = self.odf.resample("W", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = self.pdf.resample("W", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("W", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdf_dai.resample("W", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('date')
        pdf_dai = self.pdf.set_index('date')
        a = odf_dai.resample("W")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdf_dai.resample("W")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_week_param_on_date_mean(self):
        a = self.odf.resample("W", on="date").mean()
        b = self.pdf.resample("W", on="date").mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf.resample("W").mean()
        with self.assertRaises(TypeError):
            self.pdf.resample("W").mean()

        a = self.odf.resample("W", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = self.pdf.resample("W", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("W", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdf_dai.resample("W", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('date')
        pdf_dai = self.pdf.set_index('date')
        a = odf_dai.resample("W")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdf_dai.resample("W")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_week_param_on_date_min(self):
        a = self.odf.resample("W", on="date").min()
        b = self.pdf.resample("W", on="date").min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf.resample("W").min()
        with self.assertRaises(TypeError):
            self.pdf.resample("W").min()

        a = self.odf.resample("W", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = self.pdf.resample("W", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("W", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdf_dai.resample("W", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('date')
        pdf_dai = self.pdf.set_index('date')
        a = odf_dai.resample("W")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdf_dai.resample("W")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_week_param_on_date_std(self):
        a = self.odf.resample("W", on="date").std()
        self.pdf.tbool = self.pdf.tbool.astype('float32')
        b = self.pdf.resample("W", on="date").std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf.resample("W").std()
        with self.assertRaises(TypeError):
            self.pdf.resample("W").std()

        a = self.odf.resample("W", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = self.pdf.resample("W", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("W", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdf_dai.resample("W", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('date')
        pdf_dai = self.pdf.set_index('date')
        a = odf_dai.resample("W")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdf_dai.resample("W")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_week_param_on_date_sum(self):
        a = self.odf.resample("W", on="date").sum()
        b = self.pdf.resample("W", on="date").sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf.resample("W").sum()
        with self.assertRaises(TypeError):
            self.pdf.resample("W").sum()

        a = self.odf.resample("W", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = self.pdf.resample("W", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("W", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdf_dai.resample("W", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('date')
        pdf_dai = self.pdf.set_index('date')
        a = odf_dai.resample("W")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdf_dai.resample("W")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_week_param_on_date_var(self):
        a = self.odf.resample("W", on="date").var()
        b = self.pdf.resample("W", on="date").var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf.resample("W").var()
        with self.assertRaises(TypeError):
            self.pdf.resample("W").var()

        a = self.odf.resample("W", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = self.pdf.resample("W", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("W", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdf_dai.resample("W", on="date")[
            'date', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('date')
        pdf_dai = self.pdf.set_index('date')
        a = odf_dai.resample("W")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdf_dai.resample("W")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_hour_param_on_hour_count(self):
        a = self.odf.resample("H", on="hour").count()
        b = self.pdf.resample("H", on="hour").count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        self.odf.resample("3H", on="hour").count()
        self.pdf.resample("3H", on="hour").count()

        with self.assertRaises(TypeError):
            self.odf.resample("H").count()
        with self.assertRaises(TypeError):
            self.pdf.resample("H").count()

        a = self.odf.resample("H", on="hour")[
            'hour', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = self.pdf.resample("H", on="hour")[
            'hour', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("H", on="hour")[
            'hour', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdf_dai.resample("H", on="hour")[
            'hour', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('hour')
        pdf_dai = self.pdf.set_index('hour')
        a = odf_dai.resample("H")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdf_dai.resample("H")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_hour_param_on_hour_max(self):
        a = self.odf.resample("H", on="hour").max()
        b = self.pdf.resample("H", on="hour").max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf.resample("H").max()
        with self.assertRaises(TypeError):
            self.pdf.resample("H").max()

        a = self.odf.resample("H", on="hour")[
            'hour', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = self.pdf.resample("H", on="hour")[
            'hour', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("H", on="hour")[
            'hour', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdf_dai.resample("H", on="hour")[
            'hour', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('hour')
        pdf_dai = self.pdf.set_index('hour')
        a = odf_dai.resample("H")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdf_dai.resample("H")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_hour_param_on_hour_mean(self):
        a = self.odf.resample("H", on="hour").mean()
        b = self.pdf.resample("H", on="hour").mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf.resample("H").mean()
        with self.assertRaises(TypeError):
            self.pdf.resample("H").mean()

        a = self.odf.resample("H", on="hour")[
            'hour', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = self.pdf.resample("H", on="hour")[
            'hour', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("H", on="hour")[
            'hour', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdf_dai.resample("H", on="hour")[
            'hour', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('hour')
        pdf_dai = self.pdf.set_index('hour')
        a = odf_dai.resample("H")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdf_dai.resample("H")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_hour_param_on_hour_min(self):
        a = self.odf.resample("H", on="hour").min()
        b = self.pdf.resample("H", on="hour").min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf.resample("H").min()
        with self.assertRaises(TypeError):
            self.pdf.resample("H").min()

        a = self.odf.resample("H", on="hour")[
            'hour', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = self.pdf.resample("H", on="hour")[
            'hour', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("H", on="hour")[
            'hour', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdf_dai.resample("H", on="hour")[
            'hour', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('hour')
        pdf_dai = self.pdf.set_index('hour')
        a = odf_dai.resample("H")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdf_dai.resample("H")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_hour_param_on_hour_std(self):
        a = self.odf.resample("H", on="hour").std()
        pdf = self.pdf
        pdf.tbool = pdf.tbool.astype('float32')
        b = pdf.resample("H", on="hour").std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf.resample("H").std()
        with self.assertRaises(TypeError):
            pdf.resample("H").std()

        a = self.odf.resample("H", on="hour")[
            'hour', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdf.resample("H", on="hour")[
            'hour', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = pdf.set_index('id')
        a = odf_dai.resample("H", on="hour")[
            'hour', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdf_dai.resample("H", on="hour")[
            'hour', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('hour')
        pdf_dai = pdf.set_index('hour')
        a = odf_dai.resample("H")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdf_dai.resample("H")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_hour_param_on_hour_sum(self):
        a = self.odf.resample("H", on="hour").sum()
        b = self.pdf.resample("H", on="hour").sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf.resample("H").sum()
        with self.assertRaises(TypeError):
            self.pdf.resample("H").sum()

        a = self.odf.resample("H", on="hour")[
            'hour', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = self.pdf.resample("H", on="hour")[
            'hour', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("H", on="hour")[
            'hour', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdf_dai.resample("H", on="hour")[
            'hour', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('hour')
        pdf_dai = self.pdf.set_index('hour')
        a = odf_dai.resample("H")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdf_dai.resample("H")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_hour_param_on_hour_var(self):
        a = self.odf.resample("H", on="hour").var()
        b = self.pdf.resample("H", on="hour").var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf.resample("H").var()
        with self.assertRaises(TypeError):
            self.pdf.resample("H").var()

        a = self.odf.resample("H", on="hour")[
            'hour', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = self.pdf.resample("H", on="hour")[
            'hour', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("H", on="hour")[
            'hour', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdf_dai.resample("H", on="hour")[
            'hour', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('hour')
        pdf_dai = self.pdf.set_index('hour')
        a = odf_dai.resample("H")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdf_dai.resample("H")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_minute_param_on_minute_count(self):
        a = self.odf.resample("T", on="minute").count()
        b = self.pdf.resample("T", on="minute").count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        self.odf.resample("3T", on="minute").count()
        self.pdf.resample("3T", on="minute").count()

        with self.assertRaises(TypeError):
            self.odf.resample("T").count()
        with self.assertRaises(TypeError):
            self.pdf.resample("T").count()

        a = self.odf.resample("T", on="minute")[
            'minute', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = self.pdf.resample("T", on="minute")[
            'minute', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("T", on="minute")[
            'minute', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdf_dai.resample("T", on="minute")[
            'minute', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('minute')
        pdf_dai = self.pdf.set_index('minute')
        a = odf_dai.resample("T")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdf_dai.resample("T")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_minute_param_on_minute_max(self):
        a = self.odf.resample("T", on="minute").max()
        b = self.pdf.resample("T", on="minute").max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf.resample("T").max()
        with self.assertRaises(TypeError):
            self.pdf.resample("T").max()

        a = self.odf.resample("T", on="minute")[
            'minute', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = self.pdf.resample("T", on="minute")[
            'minute', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("T", on="minute")[
            'minute', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdf_dai.resample("T", on="minute")[
            'minute', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('minute')
        pdf_dai = self.pdf.set_index('minute')
        a = odf_dai.resample("T")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdf_dai.resample("T")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_minute_param_on_minute_mean(self):
        a = self.odf.resample("T", on="minute").mean()
        b = self.pdf.resample("T", on="minute").mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf.resample("T").mean()
        with self.assertRaises(TypeError):
            self.pdf.resample("T").mean()

        a = self.odf.resample("T", on="minute")[
            'minute', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = self.pdf.resample("T", on="minute")[
            'minute', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("T", on="minute")[
            'minute', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdf_dai.resample("T", on="minute")[
            'minute', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('minute')
        pdf_dai = self.pdf.set_index('minute')
        a = odf_dai.resample("T")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdf_dai.resample("T")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_minute_param_on_minute_min(self):
        a = self.odf.resample("T", on="minute").min()
        b = self.pdf.resample("T", on="minute").min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf.resample("T").min()
        with self.assertRaises(TypeError):
            self.pdf.resample("T").min()

        a = self.odf.resample("T", on="minute")[
            'minute', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = self.pdf.resample("T", on="minute")[
            'minute', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("T", on="minute")[
            'minute', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdf_dai.resample("T", on="minute")[
            'minute', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('minute')
        pdf_dai = self.pdf.set_index('minute')
        a = odf_dai.resample("T")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdf_dai.resample("T")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_minute_param_on_minute_std(self):
        a = self.odf.resample("T", on="minute").std()
        pdf = self.pdf
        pdf.tbool = pdf.tbool.astype('float32')
        b = pdf.resample("T", on="minute").std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf.resample("T").std()
        with self.assertRaises(TypeError):
            pdf.resample("T").std()

        a = self.odf.resample("T", on="minute")[
            'minute', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdf.resample("T", on="minute")[
            'minute', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = pdf.set_index('id')
        a = odf_dai.resample("T", on="minute")[
            'minute', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdf_dai.resample("T", on="minute")[
            'minute', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('minute')
        pdf_dai = pdf.set_index('minute')
        a = odf_dai.resample("T")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdf_dai.resample("T")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_minute_param_on_minute_sum(self):
        a = self.odf.resample("T", on="minute").sum()
        b = self.pdf.resample("T", on="minute").sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf.resample("T").sum()
        with self.assertRaises(TypeError):
            self.pdf.resample("T").sum()

        a = self.odf.resample("T", on="minute")[
            'minute', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = self.pdf.resample("T", on="minute")[
            'minute', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("T", on="minute")[
            'minute', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdf_dai.resample("T", on="minute")[
            'minute', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('minute')
        pdf_dai = self.pdf.set_index('minute')
        a = odf_dai.resample("T")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdf_dai.resample("T")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_minute_param_on_minute_var(self):
        a = self.odf.resample("T", on="minute").var()
        b = self.pdf.resample("T", on="minute").var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf.resample("T").var()
        with self.assertRaises(TypeError):
            self.pdf.resample("T").var()

        a = self.odf.resample("T", on="minute")[
            'minute', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = self.pdf.resample("T", on="minute")[
            'minute', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("T", on="minute")[
            'minute', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdf_dai.resample("T", on="minute")[
            'minute', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('minute')
        pdf_dai = self.pdf.set_index('minute')
        a = odf_dai.resample("T")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdf_dai.resample("T")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_second_param_on_second_count(self):
        a = self.odf.resample("S", on="second").count()
        b = self.pdf.resample("S", on="second").count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        self.odf.resample("3S", on="second").count()
        self.pdf.resample("3S", on="second").count()

        with self.assertRaises(TypeError):
            self.odf.resample("S").count()
        with self.assertRaises(TypeError):
            self.pdf.resample("S").count()

        a = self.odf.resample("S", on="second")[
            'second', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = self.pdf.resample("S", on="second")[
            'second', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("S", on="second")[
            'second', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdf_dai.resample("S", on="second")[
            'second', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('second')
        pdf_dai = self.pdf.set_index('second')
        a = odf_dai.resample("S")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdf_dai.resample("S")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_second_param_on_second_max(self):
        a = self.odf.resample("S", on="second").max()
        b = self.pdf.resample("S", on="second").max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf.resample("S").max()
        with self.assertRaises(TypeError):
            self.pdf.resample("S").max()

        a = self.odf.resample("S", on="second")[
            'second', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = self.pdf.resample("S", on="second")[
            'second', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("S", on="second")[
            'second', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdf_dai.resample("S", on="second")[
            'second', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('second')
        pdf_dai = self.pdf.set_index('second')
        a = odf_dai.resample("S")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdf_dai.resample("S")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_second_param_on_second_mean(self):
        a = self.odf.resample("S", on="second").mean()
        b = self.pdf.resample("S", on="second").mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf.resample("S").mean()
        with self.assertRaises(TypeError):
            self.pdf.resample("S").mean()

        a = self.odf.resample("S", on="second")[
            'second', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = self.pdf.resample("S", on="second")[
            'second', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("S", on="second")[
            'second', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdf_dai.resample("S", on="second")[
            'second', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('second')
        pdf_dai = self.pdf.set_index('second')
        a = odf_dai.resample("S")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdf_dai.resample("S")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_second_param_on_second_min(self):
        a = self.odf.resample("S", on="second").min()
        b = self.pdf.resample("S", on="second").min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf.resample("S").min()
        with self.assertRaises(TypeError):
            self.pdf.resample("S").min()

        a = self.odf.resample("S", on="second")[
            'second', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = self.pdf.resample("S", on="second")[
            'second', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("S", on="second")[
            'second', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdf_dai.resample("S", on="second")[
            'second', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('second')
        pdf_dai = self.pdf.set_index('second')
        a = odf_dai.resample("S")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdf_dai.resample("S")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_second_param_on_second_std(self):
        a = self.odf.resample("S", on="second").std()
        pdf = self.pdf
        pdf.tbool = pdf.tbool.astype('float32')
        b = pdf.resample("S", on="second").std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf.resample("S").std()
        with self.assertRaises(TypeError):
            pdf.resample("S").std()

        a = self.odf.resample("S", on="second")[
            'second', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdf.resample("S", on="second")[
            'second', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = pdf.set_index('id')
        a = odf_dai.resample("S", on="second")[
            'second', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdf_dai.resample("S", on="second")[
            'second', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('second')
        pdf_dai = pdf.set_index('second')
        a = odf_dai.resample("S")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdf_dai.resample("S")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_second_param_on_second_sum(self):
        a = self.odf.resample("S", on="second").sum()
        b = self.pdf.resample("S", on="second").sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf.resample("S").sum()
        with self.assertRaises(TypeError):
            self.pdf.resample("S").sum()

        a = self.odf.resample("S", on="second")[
            'second', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = self.pdf.resample("S", on="second")[
            'second', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("S", on="second")[
            'second', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdf_dai.resample("S", on="second")[
            'second', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('second')
        pdf_dai = self.pdf.set_index('second')
        a = odf_dai.resample("S")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdf_dai.resample("S")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_pandas_dataframe_resample_param_rule_second_param_on_second_var(self):
        a = self.odf.resample("S", on="second").var()
        b = self.pdf.resample("S", on="second").var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf.resample("S").var()
        with self.assertRaises(TypeError):
            self.pdf.resample("S").var()

        a = self.odf.resample("S", on="second")[
            'second', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = self.pdf.resample("S", on="second")[
            'second', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('id')
        pdf_dai = self.pdf.set_index('id')
        a = odf_dai.resample("S", on="second")[
            'second', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdf_dai.resample("S", on="second")[
            'second', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odf_dai = self.odf.set_index('second')
        pdf_dai = self.pdf.set_index('second')
        a = odf_dai.resample("S")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdf_dai.resample("S")[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_businessday_param_on_date_count(self):
        a = self.odf_csv.resample("B", on="date").count()
        b = self.pdf_csv.resample("B", on="date").count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        self.odf_csv.resample("3B", on="date").count()
        self.pdf_csv.resample("3B", on="date").count()

        with self.assertRaises(TypeError):
            self.odf_csv.resample("B").count()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("B").count()

        a = self.odf_csv.resample("B", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = self.pdf_csv.resample("B", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("B", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdfi.resample("B", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("B")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdfi.resample("B")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_businessday_param_on_date_max(self):
        a = self.odf_csv.resample("B", on="date").max()
        b = self.pdf_csv.resample("B", on="date").max()
        assert_frame_equal(a.to_pandas().iloc[:, 0:3], b.iloc[:, 0:3], check_dtype=False)
        assert_frame_equal(a.to_pandas().iloc[:, 6:8], b.iloc[:, 6:8], check_dtype=False)
        assert_frame_equal(a.to_pandas().iloc[:, 9:19], b.iloc[:, 9:19], check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf_csv.resample("B").max()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("B").max()

        a = self.odf_csv.resample("B", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = self.pdf_csv.resample("B", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("B", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdfi.resample("B", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("B")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdfi.resample("B")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_businessday_param_on_date_mean(self):
        a = self.odf_csv.resample("B", on="date").mean()
        b = self.pdf_csv.resample("B", on="date").mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf_csv.resample("B").mean()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("B").mean()

        a = self.odf_csv.resample("B", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = self.pdf_csv.resample("B", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("B", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdfi.resample("B", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("B")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdfi.resample("B")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_businessday_param_on_date_min(self):
        a = self.odf_csv.resample("B", on="date").min()
        b = self.pdf_csv.resample("B", on="date").min()
        assert_frame_equal(a.to_pandas().iloc[:, 0:3], b.iloc[:, 0:3], check_dtype=False)
        assert_frame_equal(a.to_pandas().iloc[:, 6:8], b.iloc[:, 6:8], check_dtype=False)
        assert_frame_equal(a.to_pandas().iloc[:, 9:19], b.iloc[:, 9:19], check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf_csv.resample("B").min()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("B").min()

        a = self.odf_csv.resample("B", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = self.pdf_csv.resample("B", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("B", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdfi.resample("B", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("B")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdfi.resample("B")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_businessday_param_on_date_std(self):
        a = self.odf_csv.resample("B", on="date").std()
        b = self.pdf_csv.resample("B", on="date").std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf_csv.resample("B").std()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("B").std()

        a = self.odf_csv.resample("B", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = self.pdf_csv.resample("B", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("B", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdfi.resample("B", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("B")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdfi.resample("B")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_businessday_param_on_date_sum(self):
        a = self.odf_csv.resample("B", on="date").sum()
        b = self.pdf_csv.resample("B", on="date").sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf_csv.resample("B").sum()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("B").sum()

        a = self.odf_csv.resample("B", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = self.pdf_csv.resample("B", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("B", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdfi.resample("B", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("B")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdfi.resample("B")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_businessday_param_on_date_var(self):
        a = self.odf_csv.resample("B", on="date").var()
        b = self.pdf_csv.resample("B", on="date").var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf_csv.resample("B").var()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("B").var()

        a = self.odf_csv.resample("B", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = self.pdf_csv.resample("B", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("B", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdfi.resample("B", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("B")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdfi.resample("B")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_weekofmonth_param_on_date_count(self):
        fq = WeekOfMonth(1)
        a = self.odf_csv.resample(fq, on="date").count()
        b = self.pdf_csv.resample(fq, on="date").count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False, check_index_type=False)
        fq = WeekOfMonth(2)
        self.odf_csv.resample(fq, on="date").count()
        self.pdf_csv.resample(fq, on="date").count()

        fq = WeekOfMonth(1)
        with self.assertRaises(TypeError):
            self.odf_csv.resample(fq).count()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample(fq).count()

        a = self.odf_csv.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = self.pdf_csv.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdfi.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample(fq)[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdfi.resample(fq)[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_weekofmonth_param_on_date_max(self):
        fq = WeekOfMonth(1)
        a = self.odf_csv.resample(fq, on="date").max()
        b = self.pdf_csv.resample(fq, on="date").max()
        assert_frame_equal(a.to_pandas().iloc[:, 0:3], b.iloc[:, 0:3], check_dtype=False)
        assert_frame_equal(a.to_pandas().iloc[:, 6:8], b.iloc[:, 6:8], check_dtype=False)
        assert_frame_equal(a.to_pandas().iloc[:, 9:19], b.iloc[:, 9:19], check_dtype=False)
        fq = WeekOfMonth(2)
        self.odf_csv.resample(fq, on="date").max()
        self.pdf_csv.resample(fq, on="date").max()

        fq = WeekOfMonth(1)
        with self.assertRaises(TypeError):
            self.odf_csv.resample(fq).max()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample(fq).max()

        a = self.odf_csv.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = self.pdf_csv.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdfi.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample(fq)[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdfi.resample(fq)[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_weekofmonth_param_on_date_mean(self):
        fq = WeekOfMonth(1)
        a = self.odf_csv.resample(fq, on="date").mean()
        b = self.pdf_csv.resample(fq, on="date").mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)
        fq = WeekOfMonth(2)
        self.odf_csv.resample(fq, on="date").mean()
        self.pdf_csv.resample(fq, on="date").mean()

        fq = WeekOfMonth(1)
        with self.assertRaises(TypeError):
            self.odf_csv.resample(fq).mean()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample(fq).mean()

        a = self.odf_csv.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = self.pdf_csv.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdfi.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample(fq)[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdfi.resample(fq)[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_weekofmonth_param_on_date_min(self):
        fq = WeekOfMonth(1)
        a = self.odf_csv.resample(fq, on="date").min()
        b = self.pdf_csv.resample(fq, on="date").min()
        assert_frame_equal(a.to_pandas().iloc[:, 0:3], b.iloc[:, 0:3], check_dtype=False)
        assert_frame_equal(a.to_pandas().iloc[:, 6:8], b.iloc[:, 6:8], check_dtype=False)
        assert_frame_equal(a.to_pandas().iloc[:, 9:19], b.iloc[:, 9:19], check_dtype=False)
        fq = WeekOfMonth(2)
        self.odf_csv.resample(fq, on="date").min()
        self.pdf_csv.resample(fq, on="date").min()

        fq = WeekOfMonth(1)
        with self.assertRaises(TypeError):
            self.odf_csv.resample(fq).min()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample(fq).min()

        a = self.odf_csv.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = self.pdf_csv.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdfi.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample(fq)[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdfi.resample(fq)[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_weekofmonth_param_on_date_std(self):
        fq = WeekOfMonth(1)
        a = self.odf_csv.resample(fq, on="date").std()
        b = self.pdf_csv.resample(fq, on="date").std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)
        fq = WeekOfMonth(2)
        self.odf_csv.resample(fq, on="date").std()
        self.pdf_csv.resample(fq, on="date").std()

        fq = WeekOfMonth(1)
        with self.assertRaises(TypeError):
            self.odf_csv.resample(fq).std()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample(fq).std()

        a = self.odf_csv.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = self.pdf_csv.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdfi.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample(fq)[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdfi.resample(fq)[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_weekofmonth_param_on_date_sum(self):
        fq = WeekOfMonth(1)
        a = self.odf_csv.resample(fq, on="date").sum()
        b = self.pdf_csv.resample(fq, on="date").sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)
        fq = WeekOfMonth(2)
        self.odf_csv.resample(fq, on="date").sum()
        self.pdf_csv.resample(fq, on="date").sum()

        fq = WeekOfMonth(1)
        with self.assertRaises(TypeError):
            self.odf_csv.resample(fq).sum()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample(fq).sum()

        a = self.odf_csv.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = self.pdf_csv.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdfi.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample(fq)[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdfi.resample(fq)[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_weekofmonth_param_on_date_var(self):
        fq = WeekOfMonth(1)
        a = self.odf_csv.resample(fq, on="date").var()
        b = self.pdf_csv.resample(fq, on="date").var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)
        fq = WeekOfMonth(2)
        self.odf_csv.resample(fq, on="date").var()
        self.pdf_csv.resample(fq, on="date").var()

        fq = WeekOfMonth(1)
        with self.assertRaises(TypeError):
            self.odf_csv.resample(fq).var()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample(fq).var()

        a = self.odf_csv.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = self.pdf_csv.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdfi.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample(fq)[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdfi.resample(fq)[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_lastweekofmonth_param_on_date_count(self):
        fq = LastWeekOfMonth(1)
        a = self.odf_csv.resample(fq, on="date").count()
        b = self.pdf_csv.resample(fq, on="date").count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        fq = LastWeekOfMonth(2)
        self.odf_csv.resample(fq, on="date").count()
        self.pdf_csv.resample(fq, on="date").count()

        fq = LastWeekOfMonth(1)
        with self.assertRaises(TypeError):
            self.odf_csv.resample(fq).count()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample(fq).count()

        a = self.odf_csv.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = self.pdf_csv.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdfi.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample(fq)[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdfi.resample(fq)[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_lastweekofmonth_param_on_date_max(self):
        fq = LastWeekOfMonth(1)
        a = self.odf_csv.resample(fq, on="date").max()
        b = self.pdf_csv.resample(fq, on="date").max()
        assert_frame_equal(a.to_pandas().iloc[:, 0:3], b.iloc[:, 0:3], check_dtype=False)
        assert_frame_equal(a.to_pandas().iloc[:, 6:8], b.iloc[:, 6:8], check_dtype=False)
        assert_frame_equal(a.to_pandas().iloc[:, 9:], b.iloc[:, 9:], check_dtype=False)
        fq = LastWeekOfMonth(2)

        self.odf_csv.resample(fq, on="date").max()
        self.pdf_csv.resample(fq, on="date").max()

        fq = LastWeekOfMonth(1)
        with self.assertRaises(TypeError):
            self.odf_csv.resample(fq).max()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample(fq).max()

        a = self.odf_csv.resample(fq, on="date")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = self.pdf_csv.resample(fq, on="date")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdfi.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample(fq)[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdfi.resample(fq)[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_lastweekofmonth_param_on_date_mean(self):
        fq = LastWeekOfMonth(1)
        a = self.odf_csv.resample(fq, on="date").mean()
        b = self.pdf_csv.resample(fq, on="date").mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)
        fq = LastWeekOfMonth(2)
        self.odf_csv.resample(fq, on="date").mean()
        self.pdf_csv.resample(fq, on="date").mean()

        fq = LastWeekOfMonth(1)
        with self.assertRaises(TypeError):
            self.odf_csv.resample(fq).mean()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample(fq).mean()

        a = self.odf_csv.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = self.pdf_csv.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdfi.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample(fq)[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdfi.resample(fq)[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_lastweekofmonth_param_on_date_min(self):
        fq = LastWeekOfMonth(1)
        a = self.odf_csv.resample(fq, on="date").min()
        b = self.pdf_csv.resample(fq, on="date").min()
        assert_frame_equal(a.to_pandas().iloc[:, 0:3], b.iloc[:, 0:3], check_dtype=False)
        assert_frame_equal(a.to_pandas().iloc[:, 6:8], b.iloc[:, 6:8], check_dtype=False)
        assert_frame_equal(a.to_pandas().iloc[:, 9:], b.iloc[:, 9:], check_dtype=False)
        fq = LastWeekOfMonth(2)
        self.odf_csv.resample(fq, on="date").min()
        self.pdf_csv.resample(fq, on="date").min()

        fq = LastWeekOfMonth(1)
        with self.assertRaises(TypeError):
            self.odf_csv.resample(fq).min()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample(fq).min()

        a = self.odf_csv.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = self.pdf_csv.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdfi.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample(fq)[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdfi.resample(fq)[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_lastweekofmonth_param_on_date_std(self):
        fq = LastWeekOfMonth(1)
        a = self.odf_csv.resample(fq, on="date").std()
        b = self.pdf_csv.resample(fq, on="date").std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)
        fq = LastWeekOfMonth(2)
        self.odf_csv.resample(fq, on="date").std()
        self.pdf_csv.resample(fq, on="date").std()

        fq = LastWeekOfMonth(1)
        with self.assertRaises(TypeError):
            self.odf_csv.resample(fq).std()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample(fq).std()

        a = self.odf_csv.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = self.pdf_csv.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdfi.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample(fq)[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdfi.resample(fq)[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_lastweekofmonth_param_on_date_sum(self):
        fq = LastWeekOfMonth(1)
        a = self.odf_csv.resample(fq, on="date").sum()
        b = self.pdf_csv.resample(fq, on="date").sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)
        fq = LastWeekOfMonth(2)
        self.odf_csv.resample(fq, on="date").sum()
        self.pdf_csv.resample(fq, on="date").sum()

        fq = LastWeekOfMonth(1)
        with self.assertRaises(TypeError):
            self.odf_csv.resample(fq).sum()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample(fq).sum()

        a = self.odf_csv.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = self.pdf_csv.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdfi.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample(fq)[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdfi.resample(fq)[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_lastweekofmonth_param_on_date_var(self):
        fq = LastWeekOfMonth(1)
        a = self.odf_csv.resample(fq, on="date").var()
        b = self.pdf_csv.resample(fq, on="date").var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)
        fq = LastWeekOfMonth(2)
        self.odf_csv.resample(fq, on="date").var()
        self.pdf_csv.resample(fq, on="date").var()

        fq = LastWeekOfMonth(1)
        with self.assertRaises(TypeError):
            self.odf_csv.resample(fq).var()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample(fq).var()

        a = self.odf_csv.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = self.pdf_csv.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdfi.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample(fq)[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdfi.resample(fq)[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_monthbegin_param_on_date_count(self):
        fq = MonthBegin(1)
        a = self.odf_csv.resample(fq, on="date").count()
        b = self.pdf_csv.resample(fq, on="date").count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        fq = MonthBegin(2)
        self.odf_csv.resample(fq, on="date").count()
        self.pdf_csv.resample(fq, on="date").count()

        fq = MonthBegin(1)
        with self.assertRaises(TypeError):
            self.odf_csv.resample(fq).count()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample(fq).count()

        a = self.odf_csv.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = self.pdf_csv.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdfi.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample(fq)[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdfi.resample(fq)[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_monthbegin_param_on_date_max(self):
        fq = MonthBegin(1)
        a = self.odf_csv.resample(fq, on="date").max()
        b = self.pdf_csv.resample(fq, on="date").max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True).iloc[:, 0:3], b.reset_index(drop=True).iloc[:, 0:3],
                           check_dtype=False)
        assert_frame_equal(a.to_pandas().reset_index(drop=True).iloc[:, 6:8], b.reset_index(drop=True).iloc[:, 6:8],
                           check_dtype=False)
        assert_frame_equal(a.to_pandas().reset_index(drop=True).iloc[:, 9:19], b.reset_index(drop=True).iloc[:, 9:19],
                           check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf_csv.resample(fq).max()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample(fq).max()

        a = self.odf_csv.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = self.pdf_csv.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdfi.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample(fq)[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdfi.resample(fq)[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_monthbegin_param_on_date_mean(self):
        fq = MonthBegin(1)
        a = self.odf_csv.resample(fq, on="date").mean()
        b = self.pdf_csv.resample(fq, on="date").mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf_csv.resample(fq).mean()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample(fq).mean()

        a = self.odf_csv.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = self.pdf_csv.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdfi.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample(fq)[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdfi.resample(fq)[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_monthbegin_param_on_date_min(self):
        fq = MonthBegin(1)
        a = self.odf_csv.resample(fq, on="date").min()
        b = self.pdf_csv.resample(fq, on="date").min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True).iloc[:, 0:3], b.reset_index(drop=True).iloc[:, 0:3],
                           check_dtype=False)
        assert_frame_equal(a.to_pandas().reset_index(drop=True).iloc[:, 6:8], b.reset_index(drop=True).iloc[:, 6:8],
                           check_dtype=False)
        assert_frame_equal(a.to_pandas().reset_index(drop=True).iloc[:, 9:19], b.reset_index(drop=True).iloc[:, 9:19],
                           check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf_csv.resample(fq).min()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample(fq).min()

        a = self.odf_csv.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = self.pdf_csv.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdfi.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample(fq)[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdfi.resample(fq)[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_monthbegin_param_on_date_std(self):
        fq = MonthBegin(1)
        a = self.odf_csv.resample(fq, on="date").std()
        b = self.pdf_csv.resample(fq, on="date").std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf_csv.resample(fq).std()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample(fq).std()

        a = self.odf_csv.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = self.pdf_csv.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdfi.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample(fq)[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdfi.resample(fq)[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_monthbegin_param_on_date_sum(self):
        fq = MonthBegin(1)
        a = self.odf_csv.resample(fq, on="date").sum()
        b = self.pdf_csv.resample(fq, on="date").sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf_csv.resample(fq).sum()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample(fq).sum()

        a = self.odf_csv.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = self.pdf_csv.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdfi.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample(fq)[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdfi.resample(fq)[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_monthbegin_param_on_date_var(self):
        fq = MonthBegin(1)
        a = self.odf_csv.resample(fq, on="date").var()
        b = self.pdf_csv.resample(fq, on="date").var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf_csv.resample(fq).var()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample(fq).var()

        a = self.odf_csv.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = self.pdf_csv.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdfi.resample(fq, on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample(fq)[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdfi.resample(fq)[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_year_param_on_date_count(self):
        a = self.odf_csv.resample("A", on="date").count()
        b = self.pdf_csv.resample("A", on="date").count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        self.odf_csv.resample("3A", on="date").count()
        self.pdf_csv.resample("3A", on="date").count()

        with self.assertRaises(TypeError):
            self.odf_csv.resample("A").count()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("A").count()

        a = self.odf_csv.resample("A", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = self.pdf_csv.resample("A", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("A", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdfi.resample("A", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("A")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdfi.resample("A")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_year_param_on_date_max(self):
        a = self.odf_csv.resample("A", on="date").max()
        b = self.pdf_csv.resample("A", on="date").max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True).iloc[:, 0:3], b.reset_index(drop=True).iloc[:, 0:3],
                           check_dtype=False)
        assert_frame_equal(a.to_pandas().reset_index(drop=True).iloc[:, 6:8], b.reset_index(drop=True).iloc[:, 6:8],
                           check_dtype=False)
        assert_frame_equal(a.to_pandas().reset_index(drop=True).iloc[:, 9:19], b.reset_index(drop=True).iloc[:, 9:19],
                           check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf_csv.resample("A").max()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("A").max()

        a = self.odf_csv.resample("A", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = self.pdf_csv.resample("A", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("A", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdfi.resample("A", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("A")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdfi.resample("A")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_year_param_on_date_mean(self):
        a = self.odf_csv.resample("A", on="date").mean()
        b = self.pdf_csv.resample("A", on="date").mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf_csv.resample("A").mean()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("A").mean()

        a = self.odf_csv.resample("A", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = self.pdf_csv.resample("A", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("A", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdfi.resample("A", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("A")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdfi.resample("A")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_year_param_on_date_min(self):
        a = self.odf_csv.resample("A", on="date").min()
        b = self.pdf_csv.resample("A", on="date").min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True).iloc[:, 0:3], b.reset_index(drop=True).iloc[:, 0:3],
                           check_dtype=False)
        assert_frame_equal(a.to_pandas().reset_index(drop=True).iloc[:, 6:8], b.reset_index(drop=True).iloc[:, 6:8],
                           check_dtype=False)
        assert_frame_equal(a.to_pandas().reset_index(drop=True).iloc[:, 9:19], b.reset_index(drop=True).iloc[:, 9:19],
                           check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf_csv.resample("A").min()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("A").min()

        a = self.odf_csv.resample("A", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = self.pdf_csv.resample("A", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("A", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdfi.resample("A", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("A")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdfi.resample("A")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_year_param_on_date_std(self):
        a = self.odf_csv.resample("A", on="date").std()
        b = self.pdf_csv.resample("A", on="date").std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf_csv.resample("A").std()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("A").std()

        a = self.odf_csv.resample("A", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = self.pdf_csv.resample("A", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("A", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdfi.resample("A", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("A")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdfi.resample("A")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_year_param_on_date_sum(self):
        a = self.odf_csv.resample("A", on="date").sum()
        b = self.pdf_csv.resample("A", on="date").sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf_csv.resample("A").sum()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("A").sum()

        a = self.odf_csv.resample("A", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = self.pdf_csv.resample("A", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("A", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdfi.resample("A", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("A")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdfi.resample("A")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_year_param_on_date_var(self):
        a = self.odf_csv.resample("A", on="date").var()
        b = self.pdf_csv.resample("A", on="date").var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf_csv.resample("A").var()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("A").var()

        a = self.odf_csv.resample("A", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = self.pdf_csv.resample("A", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("A", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdfi.resample("A", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("A")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdfi.resample("A")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_month_param_on_date_count(self):
        a = self.odf_csv.resample("M", on="date").count()
        b = self.pdf_csv.resample("M", on="date").count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        self.odf_csv.resample("3M", on="date").count()
        self.pdf_csv.resample("3M", on="date").count()

        with self.assertRaises(TypeError):
            self.odf_csv.resample("M").count()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("M").count()

        a = self.odf_csv.resample("M", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = self.pdf_csv.resample("M", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("M", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdfi.resample("M", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("M")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdfi.resample("M")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_month_param_on_date_max(self):
        a = self.odf_csv.resample("M", on="date").max()
        b = self.pdf_csv.resample("M", on="date").max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True).iloc[:, 0:3], b.reset_index(drop=True).iloc[:, 0:3],
                           check_dtype=False)
        assert_frame_equal(a.to_pandas().reset_index(drop=True).iloc[:, 6:8], b.reset_index(drop=True).iloc[:, 6:8],
                           check_dtype=False)
        assert_frame_equal(a.to_pandas().reset_index(drop=True).iloc[:, 9:19], b.reset_index(drop=True).iloc[:, 9:19],
                           check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf_csv.resample("M").max()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("M").max()

        a = self.odf_csv.resample("M", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = self.pdf_csv.resample("M", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("M", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdfi.resample("M", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("M")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdfi.resample("M")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_month_param_on_date_mean(self):
        a = self.odf_csv.resample("M", on="date").mean()
        b = self.pdf_csv.resample("M", on="date").mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf_csv.resample("M").mean()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("M").mean()

        a = self.odf_csv.resample("M", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = self.pdf_csv.resample("M", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("M", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdfi.resample("M", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("M")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdfi.resample("M")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_month_param_on_date_min(self):
        a = self.odf_csv.resample("M", on="date").min()
        b = self.pdf_csv.resample("M", on="date").min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True).iloc[:, 0:3], b.reset_index(drop=True).iloc[:, 0:3],
                           check_dtype=False)
        assert_frame_equal(a.to_pandas().reset_index(drop=True).iloc[:, 6:8], b.reset_index(drop=True).iloc[:, 6:8],
                           check_dtype=False)
        assert_frame_equal(a.to_pandas().reset_index(drop=True).iloc[:, 9:19], b.reset_index(drop=True).iloc[:, 9:19],
                           check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf_csv.resample("M").min()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("M").min()

        a = self.odf_csv.resample("M", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = self.pdf_csv.resample("M", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("M", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdfi.resample("M", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("M")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdfi.resample("M")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_month_param_on_date_std(self):
        a = self.odf_csv.resample("M", on="date").std()
        b = self.pdf_csv.resample("M", on="date").std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf_csv.resample("M").std()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("M").std()

        a = self.odf_csv.resample("M", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = self.pdf_csv.resample("M", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("M", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdfi.resample("M", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("M")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdfi.resample("M")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_month_param_on_date_sum(self):
        a = self.odf_csv.resample("M", on="date").sum()
        b = self.pdf_csv.resample("M", on="date").sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf_csv.resample("M").sum()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("M").sum()

        a = self.odf_csv.resample("M", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = self.pdf_csv.resample("M", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("M", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdfi.resample("M", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("M")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdfi.resample("M")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_month_param_on_date_var(self):
        a = self.odf_csv.resample("M", on="date").var()
        b = self.pdf_csv.resample("M", on="date").var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf_csv.resample("M").var()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("M").var()

        a = self.odf_csv.resample("M", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = self.pdf_csv.resample("M", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("M", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdfi.resample("M", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("M")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdfi.resample("M")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_day_param_on_date_count(self):
        a = self.odf_csv.resample("D", on="date").count()
        b = self.pdf_csv.resample("D", on="date").count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        self.odf_csv.resample("3D", on="date").count()
        self.pdf_csv.resample("3D", on="date").count()

        with self.assertRaises(TypeError):
            self.odf_csv.resample("D").count()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("D").count()

        a = self.odf_csv.resample("D", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = self.pdf_csv.resample("D", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("D", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdfi.resample("D", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("D")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdfi.resample("D")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_day_param_on_date_max(self):
        a = self.odf_csv.resample("D", on="date").max()
        b = self.pdf_csv.resample("D", on="date").max()
        assert_frame_equal(a.to_pandas().iloc[:, 0:3], b.iloc[:, 0:3], check_dtype=False)
        assert_frame_equal(a.to_pandas().iloc[:, 6:8], b.iloc[:, 6:8], check_dtype=False)
        assert_frame_equal(a.to_pandas().iloc[:, 9:19], b.iloc[:, 9:19], check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf_csv.resample("D").max()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("D").max()

        a = self.odf_csv.resample("D", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = self.pdf_csv.resample("D", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("D", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdfi.resample("D", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("D")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdfi.resample("D")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_day_param_on_date_mean(self):
        a = self.odf_csv.resample("D", on="date").mean()
        b = self.pdf_csv.resample("D", on="date").mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf_csv.resample("D").mean()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("D").mean()

        a = self.odf_csv.resample("D", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = self.pdf_csv.resample("D", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("D", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdfi.resample("D", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("D")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdfi.resample("D")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_day_param_on_date_min(self):
        a = self.odf_csv.resample("D", on="date").min()
        b = self.pdf_csv.resample("D", on="date").min()
        assert_frame_equal(a.to_pandas().iloc[:, 0:3], b.iloc[:, 0:3], check_dtype=False)
        assert_frame_equal(a.to_pandas().iloc[:, 6:8], b.iloc[:, 6:8], check_dtype=False)
        assert_frame_equal(a.to_pandas().iloc[:, 9:19], b.iloc[:, 9:19], check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf_csv.resample("D").min()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("D").min()

        a = self.odf_csv.resample("D", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = self.pdf_csv.resample("D", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("D", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdfi.resample("D", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("D")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdfi.resample("D")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_day_param_on_date_std(self):
        a = self.odf_csv.resample("D", on="date").std()
        b = self.pdf_csv.resample("D", on="date").std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf_csv.resample("D").std()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("D").std()

        a = self.odf_csv.resample("D", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = self.pdf_csv.resample("D", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("D", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdfi.resample("D", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("D")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdfi.resample("D")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_day_param_on_date_sum(self):
        a = self.odf_csv.resample("D", on="date").sum()
        b = self.pdf_csv.resample("D", on="date").sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf_csv.resample("D").sum()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("D").sum()

        a = self.odf_csv.resample("D", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = self.pdf_csv.resample("D", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("D", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdfi.resample("D", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("D")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdfi.resample("D")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_day_param_on_date_var(self):
        a = self.odf_csv.resample("D", on="date").var()
        b = self.pdf_csv.resample("D", on="date").var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf_csv.resample("D").var()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("D").var()

        a = self.odf_csv.resample("D", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = self.pdf_csv.resample("D", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("D", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdfi.resample("D", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("D")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdfi.resample("D")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_week_param_on_date_count(self):
        a = self.odf_csv.resample("W", on="date").count()
        b = self.pdf_csv.resample("W", on="date").count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        self.odf_csv.resample("3W", on="date").count()
        self.pdf_csv.resample("3W", on="date").count()

        with self.assertRaises(TypeError):
            self.odf_csv.resample("W").count()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("W").count()

        a = self.odf_csv.resample("W", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = self.pdf_csv.resample("W", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("W", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdfi.resample("W", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("W")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdfi.resample("W")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_week_param_on_date_max(self):
        a = self.odf_csv.resample("W", on="date").max()
        b = self.pdf_csv.resample("W", on="date").max()
        assert_frame_equal(a.to_pandas().iloc[:, 0:3], b.iloc[:, 0:3], check_dtype=False)
        assert_frame_equal(a.to_pandas().iloc[:, 6:8], b.iloc[:, 6:8], check_dtype=False)
        assert_frame_equal(a.to_pandas().iloc[:, 9:19], b.iloc[:, 9:19], check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf_csv.resample("W").max()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("W").max()

        a = self.odf_csv.resample("W", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = self.pdf_csv.resample("W", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("W", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdfi.resample("W", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("W")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdfi.resample("W")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_week_param_on_date_mean(self):
        a = self.odf_csv.resample("W", on="date").mean()
        b = self.pdf_csv.resample("W", on="date").mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf_csv.resample("W").mean()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("W").mean()

        a = self.odf_csv.resample("W", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = self.pdf_csv.resample("W", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("W", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdfi.resample("W", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("W")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdfi.resample("W")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_week_param_on_date_min(self):
        a = self.odf_csv.resample("W", on="date").min()
        b = self.pdf_csv.resample("W", on="date").min()
        assert_frame_equal(a.to_pandas().iloc[:, 0:3], b.iloc[:, 0:3], check_dtype=False)
        assert_frame_equal(a.to_pandas().iloc[:, 6:8], b.iloc[:, 6:8], check_dtype=False)
        assert_frame_equal(a.to_pandas().iloc[:, 9:19], b.iloc[:, 9:19], check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf_csv.resample("W").min()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("W").min()

        a = self.odf_csv.resample("W", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = self.pdf_csv.resample("W", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("W", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdfi.resample("W", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("W")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdfi.resample("W")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_week_param_on_date_std(self):
        a = self.odf_csv.resample("W", on="date").std()
        b = self.pdf_csv.resample("W", on="date").std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf_csv.resample("W").std()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("W").std()

        a = self.odf_csv.resample("W", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = self.pdf_csv.resample("W", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("W", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdfi.resample("W", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("W")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdfi.resample("W")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_week_param_on_date_sum(self):
        a = self.odf_csv.resample("W", on="date").sum()
        b = self.pdf_csv.resample("W", on="date").sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf_csv.resample("W").sum()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("W").sum()

        a = self.odf_csv.resample("W", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = self.pdf_csv.resample("W", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("W", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdfi.resample("W", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("W")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdfi.resample("W")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_week_param_on_date_var(self):
        a = self.odf_csv.resample("W", on="date").var()
        b = self.pdf_csv.resample("W", on="date").var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf_csv.resample("W").var()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("W").var()

        a = self.odf_csv.resample("W", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = self.pdf_csv.resample("W", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("W", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdfi.resample("W", on="date")[
            'date', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('date')
        pdfi = self.pdf_csv.set_index('date')
        a = odfi.resample("W")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdfi.resample("W")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_hour_param_on_timestamp_count(self):
        a = self.odf_csv.resample("H", on="timestamp").count()
        b = self.pdf_csv.resample("H", on="timestamp").count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        self.odf_csv.resample("3H", on="timestamp").count()
        self.pdf_csv.resample("3H", on="timestamp").count()

        with self.assertRaises(TypeError):
            self.odf_csv.resample("H").count()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("H").count()

        a = self.odf_csv.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = self.pdf_csv.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdfi.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('timestamp')
        pdfi = self.pdf_csv.set_index('timestamp')
        a = odfi.resample("H")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdfi.resample("H")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_hour_param_on_timestamp_max(self):
        a = self.odf_csv.resample("H", on="timestamp").max()
        b = self.pdf_csv.resample("H", on="timestamp").max()
        assert_frame_equal(a.to_pandas().iloc[:, 0:3], b.iloc[:, 0:3], check_dtype=False)
        assert_frame_equal(a.to_pandas().iloc[:, 6:8], b.iloc[:, 6:8], check_dtype=False)
        assert_frame_equal(a.to_pandas().iloc[:, 9:19], b.iloc[:, 9:19], check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf_csv.resample("H").max()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("H").max()

        a = self.odf_csv.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = self.pdf_csv.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdfi.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('timestamp')
        pdfi = self.pdf_csv.set_index('timestamp')
        a = odfi.resample("H")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdfi.resample("H")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_hour_param_on_timestamp_mean(self):
        a = self.odf_csv.resample("H", on="timestamp").mean()
        b = self.pdf_csv.resample("H", on="timestamp").mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf_csv.resample("H").mean()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("H").mean()

        a = self.odf_csv.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = self.pdf_csv.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdfi.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('timestamp')
        pdfi = self.pdf_csv.set_index('timestamp')
        a = odfi.resample("H")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdfi.resample("H")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_hour_param_on_timestamp_min(self):
        a = self.odf_csv.resample("H", on="timestamp").min()
        b = self.pdf_csv.resample("H", on="timestamp").min()
        assert_frame_equal(a.to_pandas().iloc[:, 0:3], b.iloc[:, 0:3], check_dtype=False)
        assert_frame_equal(a.to_pandas().iloc[:, 6:8], b.iloc[:, 6:8], check_dtype=False)
        assert_frame_equal(a.to_pandas().iloc[:, 9:19], b.iloc[:, 9:19], check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf_csv.resample("H").min()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("H").min()

        a = self.odf_csv.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = self.pdf_csv.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdfi.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('timestamp')
        pdfi = self.pdf_csv.set_index('timestamp')
        a = odfi.resample("H")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdfi.resample("H")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_hour_param_on_timestamp_std(self):
        a = self.odf_csv.resample("H", on="timestamp").std()
        b = self.pdf_csv.resample("H", on="timestamp").std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf_csv.resample("H").std()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("H").std()

        a = self.odf_csv.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = self.pdf_csv.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdfi.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('timestamp')
        pdfi = self.pdf_csv.set_index('timestamp')
        a = odfi.resample("H")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdfi.resample("H")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_hour_param_on_timestamp_sum(self):
        a = self.odf_csv.resample("H", on="timestamp").sum()
        b = self.pdf_csv.resample("H", on="timestamp").sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf_csv.resample("H").sum()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("H").sum()

        a = self.odf_csv.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = self.pdf_csv.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdfi.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('timestamp')
        pdfi = self.pdf_csv.set_index('timestamp')
        a = odfi.resample("H")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdfi.resample("H")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_hour_param_on_timestamp_var(self):
        a = self.odf_csv.resample("H", on="timestamp").var()
        b = self.pdf_csv.resample("H", on="timestamp").var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf_csv.resample("H").var()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("H").var()

        a = self.odf_csv.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = self.pdf_csv.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdfi.resample("H", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('timestamp')
        pdfi = self.pdf_csv.set_index('timestamp')
        a = odfi.resample("H")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdfi.resample("H")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_minute_param_on_timestamp_count(self):
        a = self.odf_csv.resample("T", on="timestamp").count()
        b = self.pdf_csv.resample("T", on="timestamp").count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        self.odf_csv.resample("3T", on="timestamp").count()
        self.pdf_csv.resample("3T", on="timestamp").count()

        with self.assertRaises(TypeError):
            self.odf_csv.resample("T").count()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("T").count()

        a = self.odf_csv.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = self.pdf_csv.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdfi.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('timestamp')
        pdfi = self.pdf_csv.set_index('timestamp')
        a = odfi.resample("T")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdfi.resample("T")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_minute_param_on_timestamp_max(self):
        a = self.odf_csv.resample("T", on="timestamp").max()
        b = self.pdf_csv.resample("T", on="timestamp").max()
        assert_frame_equal(a.to_pandas().iloc[:, 0:3], b.iloc[:, 0:3], check_dtype=False)
        assert_frame_equal(a.to_pandas().iloc[:, 6:8], b.iloc[:, 6:8], check_dtype=False)
        assert_frame_equal(a.to_pandas().iloc[:, 9:19], b.iloc[:, 9:19], check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf_csv.resample("T").max()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("T").max()

        a = self.odf_csv.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = self.pdf_csv.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdfi.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('timestamp')
        pdfi = self.pdf_csv.set_index('timestamp')
        a = odfi.resample("T")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdfi.resample("T")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_minute_param_on_timestamp_mean(self):
        a = self.odf_csv.resample("T", on="timestamp").mean()
        b = self.pdf_csv.resample("T", on="timestamp").mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf_csv.resample("T").mean()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("T").mean()

        a = self.odf_csv.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = self.pdf_csv.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdfi.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('timestamp')
        pdfi = self.pdf_csv.set_index('timestamp')
        a = odfi.resample("T")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdfi.resample("T")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_minute_param_on_timestamp_min(self):
        a = self.odf_csv.resample("T", on="timestamp").min()
        b = self.pdf_csv.resample("T", on="timestamp").min()
        assert_frame_equal(a.to_pandas().iloc[:, 0:3], b.iloc[:, 0:3], check_dtype=False)
        assert_frame_equal(a.to_pandas().iloc[:, 6:8], b.iloc[:, 6:8], check_dtype=False)
        assert_frame_equal(a.to_pandas().iloc[:, 9:19], b.iloc[:, 9:19], check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf_csv.resample("T").min()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("T").min()

        a = self.odf_csv.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = self.pdf_csv.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdfi.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('timestamp')
        pdfi = self.pdf_csv.set_index('timestamp')
        a = odfi.resample("T")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdfi.resample("T")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_minute_param_on_timestamp_std(self):
        a = self.odf_csv.resample("T", on="timestamp").std()
        b = self.pdf_csv.resample("T", on="timestamp").std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf_csv.resample("T").std()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("T").std()

        a = self.odf_csv.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = self.pdf_csv.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdfi.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('timestamp')
        pdfi = self.pdf_csv.set_index('timestamp')
        a = odfi.resample("T")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdfi.resample("T")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_minute_param_on_timestamp_sum(self):
        a = self.odf_csv.resample("T", on="timestamp").sum()
        b = self.pdf_csv.resample("T", on="timestamp").sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf_csv.resample("T").sum()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("T").sum()

        a = self.odf_csv.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = self.pdf_csv.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdfi.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('timestamp')
        pdfi = self.pdf_csv.set_index('timestamp')
        a = odfi.resample("T")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdfi.resample("T")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_minute_param_on_timestamp_var(self):
        a = self.odf_csv.resample("T", on="timestamp").var()
        b = self.pdf_csv.resample("T", on="timestamp").var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf_csv.resample("T").var()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("T").var()

        a = self.odf_csv.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = self.pdf_csv.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdfi.resample("T", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('timestamp')
        pdfi = self.pdf_csv.set_index('timestamp')
        a = odfi.resample("T")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdfi.resample("T")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_second_param_on_timestamp_count(self):
        a = self.odf_csv.resample("S", on="timestamp").count()
        b = self.pdf_csv.resample("S", on="timestamp").count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        self.odf_csv.resample("3S", on="timestamp").count()
        self.pdf_csv.resample("3S", on="timestamp").count()

        with self.assertRaises(TypeError):
            self.odf_csv.resample("S").count()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("S").count()

        a = self.odf_csv.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = self.pdf_csv.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdfi.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('timestamp')
        pdfi = self.pdf_csv.set_index('timestamp')
        a = odfi.resample("S")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdfi.resample("S")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_second_param_on_timestamp_max(self):
        a = self.odf_csv.resample("S", on="timestamp").max()
        b = self.pdf_csv.resample("S", on="timestamp").max()
        assert_frame_equal(a.to_pandas().iloc[:, 0:3], b.iloc[:, 0:3], check_dtype=False)
        assert_frame_equal(a.to_pandas().iloc[:, 6:8], b.iloc[:, 6:8], check_dtype=False)
        assert_frame_equal(a.to_pandas().iloc[:, 9:19], b.iloc[:, 9:19], check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf_csv.resample("S").max()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("S").max()

        a = self.odf_csv.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = self.pdf_csv.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdfi.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('timestamp')
        pdfi = self.pdf_csv.set_index('timestamp')
        a = odfi.resample("S")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdfi.resample("S")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_second_param_on_timestamp_mean(self):
        a = self.odf_csv.resample("S", on="timestamp").mean()
        b = self.pdf_csv.resample("S", on="timestamp").mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf_csv.resample("S").mean()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("S").mean()

        a = self.odf_csv.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = self.pdf_csv.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdfi.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('timestamp')
        pdfi = self.pdf_csv.set_index('timestamp')
        a = odfi.resample("S")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdfi.resample("S")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_second_param_on_timestamp_min(self):
        a = self.odf_csv.resample("S", on="timestamp").min()
        b = self.pdf_csv.resample("S", on="timestamp").min()
        assert_frame_equal(a.to_pandas().iloc[:, 0:3], b.iloc[:, 0:3], check_dtype=False)
        assert_frame_equal(a.to_pandas().iloc[:, 6:8], b.iloc[:, 6:8], check_dtype=False)
        assert_frame_equal(a.to_pandas().iloc[:, 9:19], b.iloc[:, 9:19], check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf_csv.resample("S").min()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("S").min()

        a = self.odf_csv.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = self.pdf_csv.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdfi.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('timestamp')
        pdfi = self.pdf_csv.set_index('timestamp')
        a = odfi.resample("S")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdfi.resample("S")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_second_param_on_timestamp_std(self):
        a = self.odf_csv.resample("S", on="timestamp").std()
        b = self.pdf_csv.resample("S", on="timestamp").std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf_csv.resample("S").std()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("S").std()

        a = self.odf_csv.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = self.pdf_csv.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdfi.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('timestamp')
        pdfi = self.pdf_csv.set_index('timestamp')
        a = odfi.resample("S")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdfi.resample("S")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_second_param_on_timestamp_sum(self):
        a = self.odf_csv.resample("S", on="timestamp").sum()
        b = self.pdf_csv.resample("S", on="timestamp").sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf_csv.resample("S").sum()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("S").sum()

        a = self.odf_csv.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = self.pdf_csv.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdfi.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('timestamp')
        pdfi = self.pdf_csv.set_index('timestamp')
        a = odfi.resample("S")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdfi.resample("S")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_from_import_dataframe_resample_param_rule_second_param_on_timestamp_var(self):
        a = self.odf_csv.resample("S", on="timestamp").var()
        b = self.pdf_csv.resample("S", on="timestamp").var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        with self.assertRaises(TypeError):
            self.odf_csv.resample("S").var()
        with self.assertRaises(TypeError):
            self.pdf_csv.resample("S").var()

        a = self.odf_csv.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = self.pdf_csv.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('id')
        pdfi = self.pdf_csv.set_index('id')
        a = odfi.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdfi.resample("S", on="timestamp")[
            'timestamp', 'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        odfi = self.odf_csv.set_index('timestamp')
        pdfi = self.pdf_csv.set_index('timestamp')
        a = odfi.resample("S")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdfi.resample("S")[
            'tstring', 'tsymbol', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)


if __name__ == '__main__':
    unittest.main()
