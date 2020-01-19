import unittest
import orca
from setup.settings import *
from pandas.util.testing import *


class SeriesDtTest(unittest.TestCase):
    def setUp(self):
        self.PRECISION = 5

    @classmethod
    def setUpClass(cls):
        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

    @property
    def ps(self):
        return pd.Series(['Foo', 'ss ', 'sW', 'qa'], name='x')

    @property
    def os(self):
        return orca.Series(self.ps)

    @property
    def psa(self):
        return pd.Series([10, 1, 19, np.nan], index=['a', 'b', 'c', 'd'])

    @property
    def psb(self):
        return pd.Series([-1, np.nan, 1, np.nan], index=['a', 'b', 'd', 'e'])

    def test_series_datetime_properties_dt_date(self):
        ps = pd.date_range(start='2017-01-01 08:10:50', periods=10, freq='5h').to_series()
        os = orca.Series(ps)
        assert_series_equal(os.dt.date.to_pandas(), pd.to_datetime(ps.dt.date), check_dtype=False)

    def test_series_datetime_properties_dt_time(self):
        ps = pd.date_range(start='2017-01-01 08:10:50', periods=10, freq='5h').to_series()
        os = orca.Series(ps)
        # orca add date automatically
        result = pd.Series(pd.to_datetime(['1970-01-01T08:10:50', '1970-01-01T13:10:50',
                                           '1970-01-01T18:10:50', '1970-01-01T23:10:50',
                                           '1970-01-01T04:10:50', '1970-01-01T09:10:50',
                                           '1970-01-01T14:10:50', '1970-01-01T19:10:50',
                                           '1970-01-01T00:10:50', '1970-01-01T05:10:50']), index=ps)
        assert_series_equal(os.dt.time.to_pandas(), result, check_dtype=False)

    def test_series_datetime_properties_dt_year(self):
        ps = pd.date_range('2016-12-31 ', '2020-01-08', freq='m').to_series()
        os = orca.Series(ps)
        assert_series_equal(os.dt.year.to_pandas(), ps.dt.year, check_dtype=False)
        ps = pd.to_datetime(
            ["20190101", np.nan, "20190105", "20170101", "20171231", "20170615", "20181231"]).to_series()
        os = orca.Series(ps)
        assert_series_equal(os.dt.year.to_pandas(), ps.dt.year, check_dtype=False)

    def test_series_datetime_properties_dt_month(self):
        ps = pd.date_range('2016-12-31 ', '2017-01-08', freq='D').to_series()
        os = orca.Series(ps)
        assert_series_equal(os.dt.month.to_pandas(), ps.dt.month, check_dtype=False)
        ps = pd.to_datetime(
            ["20190101", np.nan, "20190105", "20170101", "20171231", "20170615", "20181231"]).to_series()
        os = orca.Series(ps)
        assert_series_equal(os.dt.month.to_pandas(), ps.dt.month, check_dtype=False)

    def test_series_datetime_properties_dt_day(self):
        ps = pd.date_range('2016-12-31 ', '2017-01-08', freq='h').to_series()
        os = orca.Series(ps)
        assert_series_equal(os.dt.day.to_pandas(), ps.dt.day, check_dtype=False)
        ps = pd.to_datetime(
            ["20190101", np.nan, "20190105", "20170101", "20171231", "20170615", "20181231"]).to_series()
        os = orca.Series(ps)
        assert_series_equal(os.dt.day.to_pandas(), ps.dt.day, check_dtype=False)

    def test_series_datetime_properties_dt_hour(self):
        ps = pd.date_range('2016-12-31 ', '2017-01-01', freq='t').to_series()
        os = orca.Series(ps)
        assert_series_equal(os.dt.hour.to_pandas(), ps.dt.hour, check_dtype=False)

    def test_series_datetime_properties_dt_minute(self):
        ps = pd.date_range('2016-12-31 ', '2017-01-08', freq='s').to_series()
        os = orca.Series(ps)
        assert_series_equal(os.dt.minute.to_pandas(), ps.dt.minute, check_dtype=False)

    def test_series_datetime_properties_dt_second(self):
        ps = pd.date_range('2016-12-31 ', periods=500, freq='ms').to_series()
        os = orca.Series(ps)
        assert_series_equal(os.dt.second.to_pandas(), ps.dt.second, check_dtype=False)

    def test_series_datetime_properties_dt_microsecond(self):
        ps = pd.date_range('2016-12-31 00:00:00', periods=500, freq='us').to_series()
        os = orca.Series(ps)
        assert_series_equal(os.dt.microsecond.to_pandas(), ps.dt.microsecond, check_dtype=False)

    def test_series_datetime_properties_dt_nanosecond(self):
        ps = pd.date_range('2016-12-31 ', periods=200, freq='n').to_series()
        os = orca.Series(ps)
        assert_series_equal(os.dt.nanosecond.to_pandas(), ps.dt.nanosecond, check_dtype=False)

    def test_series_datetime_properties_dt_week(self):
        ps = pd.date_range('2016-12-31 ', periods=500, freq='d').to_series()
        os = orca.Series(ps)
        assert_series_equal(os.dt.week.to_pandas(), ps.dt.week, check_dtype=False)

    def test_series_datetime_properties_dt_weekofyear(self):
        ps = pd.date_range('2016-12-31 ', periods=500, freq='d').to_series()
        os = orca.Series(ps)
        assert_series_equal(os.dt.weekofyear.to_pandas(), ps.dt.weekofyear, check_dtype=False)

    def test_series_datetime_properties_dt_dayofweek(self):
        ps = pd.date_range('2016-12-31 ', periods=500, freq='d').to_series()
        os = orca.Series(ps)
        assert_series_equal(os.dt.dayofweek.to_pandas(), ps.dt.dayofweek, check_dtype=False)

    def test_series_datetime_properties_dt_weekday(self):
        ps = pd.date_range('2016-12-31 ', periods=500, freq='d').to_series()
        os = orca.Series(ps)
        assert_series_equal(os.dt.weekday.to_pandas(), ps.dt.weekday, check_dtype=False)

    def test_series_datetime_properties_dt_dayofyear(self):
        ps = pd.date_range('2016-12-31 ', periods=500, freq='d').to_series()
        os = orca.Series(ps)
        assert_series_equal(os.dt.dayofyear.to_pandas(), ps.dt.dayofyear, check_dtype=False)

    def test_series_datetime_properties_dt_quarter(self):
        ps = pd.date_range('2016-12-31 ', periods=500, freq='d').to_series()
        os = orca.Series(ps)
        assert_series_equal(os.dt.quarter.to_pandas(), ps.dt.quarter, check_dtype=False)

    def test_series_datetime_properties_dt_is_month_start(self):
        ps = pd.date_range('2016-12-31 ', periods=500, freq='d').to_series()
        os = orca.Series(ps)
        assert_series_equal(os.dt.is_month_start.to_pandas(), ps.dt.is_month_start, check_dtype=False)

    def test_series_datetime_properties_dt_is_month_end(self):
        ps = pd.date_range('2016-12-31 ', periods=500, freq='d').to_series()
        os = orca.Series(ps)
        assert_series_equal(os.dt.is_month_end.to_pandas(), ps.dt.is_month_end, check_dtype=False)

    def test_series_datetime_properties_dt_is_quarter_start(self):
        ps = pd.date_range('2016-12-31 ', periods=500, freq='d').to_series()
        os = orca.Series(ps)
        assert_series_equal(os.dt.is_quarter_start.to_pandas(), ps.dt.is_quarter_start, check_dtype=False)

    def test_series_datetime_properties_dt_is_quarter_end(self):
        ps = pd.date_range('2016-12-31 ', periods=500, freq='d').to_series()
        os = orca.Series(ps)
        assert_series_equal(os.dt.is_quarter_end.to_pandas(), ps.dt.is_quarter_end, check_dtype=False)

    def test_series_datetime_properties_dt_is_year_start(self):
        ps = pd.to_datetime(
            ["20190101", np.nan, "20190105", "20170101", "20171231", "20170615", "20181231"]).to_series()
        os = orca.Series(ps)
        assert_series_equal(os.dt.is_year_start.to_pandas().fillna(False), ps.dt.is_year_start, check_dtype=False)

    def test_series_datetime_properties_dt_is_year_end(self):
        ps = pd.to_datetime(
            ["20190101", np.nan, "20190105", "20170101", "20171231", "20170615", "20181231"]).to_series()
        os = orca.Series(ps)
        assert_series_equal(os.dt.is_year_end.to_pandas().fillna(False), ps.dt.is_year_end, check_dtype=False)

    def test_series_datetime_properties_dt_is_leap_year(self):
        ps = pd.date_range("2012-01-01", "2015-01-01", freq="Y").to_series()
        os = orca.Series(ps)
        assert_series_equal(os.dt.is_leap_year.to_pandas(), ps.dt.is_leap_year, check_dtype=False)

    def test_series_datetime_properties_dt_daysinmonth(self):
        ps = pd.date_range('2016-12-31 ', periods=500, freq='d').to_series()
        os = orca.Series(ps)
        assert_series_equal(os.dt.daysinmonth.to_pandas(), ps.dt.daysinmonth, check_dtype=False)

    def test_series_datetime_properties_dt_days_in_month(self):
        ps = pd.date_range('2016-12-31 ', periods=500, freq='d').to_series()
        os = orca.Series(ps)
        assert_series_equal(os.dt.days_in_month.to_pandas(), ps.dt.days_in_month, check_dtype=False)

    def test_topic_series_datetime_groupby_properties_dt_month(self):
        ps = pd.date_range('2016-12-31 ', '2017-01-08', freq='D').to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, 74.19005356, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.month).sum().to_pandas(), pdf.groupby(pdf.date.dt.month).sum(),
                           check_dtype=False)

        ps = pd.to_datetime(
            ["20190101", np.nan, "20190105", "20170101", "20171231", "20170615", "20170131", "20170501",
             "20180517"]).to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, np.nan, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.month).sum().to_pandas().iloc[1:],
                           pdf.groupby(pdf.date.dt.month).sum(),
                           check_dtype=False)

        ps = pd.to_datetime(
            ["20190101", "20190101", "20190105", "20170101", "20171231", "20170615", "20170131", "20170501",
             "20180517"]).to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, np.nan, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.month).sum().to_pandas(),
                           pdf.groupby(pdf.date.dt.month).sum(),
                           check_dtype=False)

    def test_topic_series_datetime_groupby_properties_dt_year(self):
        ps = pd.date_range('2016-12-31 ', '2017-01-08', freq='D').to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, 74.19005356, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.year).sum().to_pandas(), pdf.groupby(pdf.date.dt.year).sum(),
                           check_dtype=False)
        ps = pd.to_datetime(
            ["20190101", np.nan, "20190105", "20170101", "20171231", "20170615", "20170131", "20170501",
             "20180517"]).to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, np.nan, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.year).sum().to_pandas().iloc[1:],
                           pdf.groupby(pdf.date.dt.year).sum(),
                           check_dtype=False)

        ps = pd.to_datetime(
            ["20190101", "20190101", "20190105", "20170101", "20171231", "20170615", "20170131", "20170501",
             "20180517"]).to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, np.nan, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.year).sum().to_pandas(),
                           pdf.groupby(pdf.date.dt.year).sum(),
                           check_dtype=False)

    def test_topic_series_datetime_groupby_properties_dt_day(self):
        ps = pd.date_range('2016-12-31 ', '2017-01-08', freq='D').to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, 74.19005356, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.day).sum().to_pandas(), pdf.groupby(pdf.date.dt.day).sum(),
                           check_dtype=False)
        ps = pd.to_datetime(
            ["20190101", np.nan, "20190105", "20170101", "20171231", "20170615", "20170131", "20170501",
             "20180517"]).to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, np.nan, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.day).sum().to_pandas().iloc[1:], pdf.groupby(pdf.date.dt.day).sum(),
                           check_dtype=False)
        ps = pd.to_datetime(
            ["20190101", "20190101", "20190105", "20170101", "20171231", "20170615", "20170131", "20170501",
             "20180517"]).to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, np.nan, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.day).sum().to_pandas(),
                           pdf.groupby(pdf.date.dt.day).sum(),
                           check_dtype=False)

    def test_topic_series_datetime_groupby_properties_dt_date(self):
        ps = pd.date_range(start='2017-01-01 08:10:50', periods=9, freq='5h').to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, 74.19005356, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.date).sum().to_pandas(), pdf.groupby(pdf.date.dt.date).sum(),
                           check_dtype=False, check_like=True)

        ps = pd.to_datetime(
            ["20190101 12:00:00", "20190105 12:50:00", "20190105 12:00:00", "20170101 12:00:00", "20171231 12:00:00",
             "20170615 12:00:00", "20170131 12:00:00", "20170501 12:00:00",
             "20180517 12:00:00"]).to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, np.nan, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.date).sum().to_pandas(), pdf.groupby(pdf.date.dt.date).sum(),
                           check_dtype=False, check_like=True)

        ps = pd.to_datetime(
            ["20190101 12:00:00", np.nan, "20190105 12:00:00", "20170101 12:00:00", "20171231 12:00:00",
             "20170615 12:00:00", "20170131 12:00:00", "20170501 12:00:00",
             "20180517 12:00:00"]).to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, np.nan, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.date).sum().to_pandas().iloc[1:],
                           pdf.groupby(pdf.date.dt.date).sum(), check_dtype=False, check_like=True)

    def test_topic_series_datetime_groupby_properties_dt_time(self):
        ps = pd.date_range(start='2017-01-01 08:10:50', periods=9, freq='5h').to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, 74.19005356, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.time).sum().to_pandas().reset_index(drop=True),
                           pdf.groupby(pdf.date.dt.time).sum().reset_index(drop=True), check_dtype=False)

        ps = pd.to_datetime(
            ["20190101 12:00:00", np.nan, "20190105 21:00:00", "20170101 12:00:00", "20171231 12:16:50",
             "20170615 12:50:00", "20170131 21:00:00", "20170501 21:10:00",
             "20180517 15:45:43"]).to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, np.nan, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.time).sum().to_pandas().iloc[1:].reset_index(drop=True),
                           pdf.groupby(pdf.date.dt.time).sum().reset_index(drop=True), check_dtype=False)

        ps = pd.to_datetime(
            ["20190101 12:00:00", "20190105 12:50:00", "20190105 12:00:00", "20170101 12:00:00", "20171231 12:00:00",
             "20170615 12:00:00", "20170131 12:00:00", "20170501 12:00:00",
             "20180517 12:00:00"]).to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, np.nan, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.time).sum().reset_index(drop=True).to_pandas(),
                           pdf.groupby(pdf.date.dt.time).sum().reset_index(drop=True),
                           check_dtype=False)

    def test_topic_series_datetime_groupby_properties_dt_hour(self):
        ps = pd.date_range('2016-12-31 16:45:50', periods=9, freq='15t').to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, 74.19005356, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.hour).sum().to_pandas(), pdf.groupby(pdf.date.dt.hour).sum(),
                           check_dtype=False)
        ps = pd.to_datetime(
            ["20190101 16:45:50", np.nan, "20190105 17:45:50", "20170101 16:50:50", "20171231 10:45:50",
             "20170615 16:45:50", "20170131 16:46:50", "20170501 19:45:50",
             "20180517 12:00:00"]).to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, np.nan, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.hour).sum().to_pandas().iloc[1:],
                           pdf.groupby(pdf.date.dt.hour).sum(),
                           check_dtype=False)
        ps = pd.to_datetime(
            ["20190101 16:45:50", "20190105 12:50:00", "20190105 17:45:50", "20170101 16:50:50", "20171231 10:45:50",
             "20170615 16:45:50", "20170131 16:46:50", "20170501 19:45:50",
             "20180517 12:00:00"]).to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, np.nan, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.hour).sum().to_pandas(),
                           pdf.groupby(pdf.date.dt.hour).sum(),
                           check_dtype=False)

    def test_topic_series_datetime_groupby_properties_dt_minute(self):
        ps = pd.date_range('2016-12-31 16:45:50', periods=9, freq='25s').to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, 74.19005356, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.minute).sum().to_pandas(), pdf.groupby(pdf.date.dt.minute).sum(),
                           check_dtype=False)
        ps = pd.to_datetime(
            ["20190101 16:45:50", np.nan, "20190105 17:45:50", "20170101 16:50:50", "20171231 10:45:50",
             "20170615 16:45:50", "20170131 16:46:50", "20170501 19:45:50",
             "20180517 12:00:00"]).to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, np.nan, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.minute).sum().to_pandas().iloc[1:],
                           pdf.groupby(pdf.date.dt.minute).sum(),
                           check_dtype=False)
        ps = pd.to_datetime(
            ["20190101 16:45:50", "20190105 12:50:00", "20190105 17:45:50", "20170101 16:50:50", "20171231 10:45:50",
             "20170615 16:45:50", "20170131 16:46:50", "20170501 19:45:50",
             "20180517 12:00:00"]).to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, np.nan, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.minute).sum().to_pandas(),
                           pdf.groupby(pdf.date.dt.minute).sum(),
                           check_dtype=False)

    def test_topic_series_datetime_groupby_properties_dt_second(self):
        ps = pd.date_range('2016-12-31 16:45:50', periods=9, freq='500ms').to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, 74.19005356, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.second).sum().to_pandas(), pdf.groupby(pdf.date.dt.second).sum(),
                           check_dtype=False)
        ps = pd.to_datetime(
            ["20190101 16:45:50", np.nan, "20190105 17:45:50", "20170101 16:50:50", "20171231 10:45:50",
             "20170615 16:45:50", "20170131 16:46:50", "20170501 19:45:50",
             "20180517 12:00:00"]).to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, np.nan, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.second).sum().to_pandas().iloc[1:],
                           pdf.groupby(pdf.date.dt.second).sum(),
                           check_dtype=False)
        ps = pd.to_datetime(
            ["20190101 16:45:50", "20190105 12:50:00", "20190105 17:45:50", "20170101 16:50:50",
             "20171231 10:45:50",
             "20170615 16:45:50", "20170131 16:46:50", "20170501 19:45:50",
             "20180517 12:00:00"]).to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, np.nan, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.second).sum().to_pandas(),
                           pdf.groupby(pdf.date.dt.second).sum(),
                           check_dtype=False)

    def test_topic_series_datetime_groupby_properties_dt_microsecond(self):
        ps = pd.date_range('2016-12-31 16:45:50', periods=9, freq='500n').to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, 74.19005356, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.microsecond).sum().to_pandas(),
                           pdf.groupby(pdf.date.dt.microsecond).sum(),
                           check_dtype=False)
        ps = pd.to_datetime(
            ["20190101 16:45:50.000000500", np.nan, "20190105 17:45:50.10000500", "20170101 16:50:50.0000001000",
             "20171231 10:45:50.000001500",
             "20170615 16:45:50.000002000", "20170131 16:46:50.000002500", "20170501 19:45:50.000003000",
             "20180517 12:00:00.000003500"]).to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, np.nan, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.microsecond).sum().to_pandas().iloc[1:],
                           pdf.groupby(pdf.date.dt.microsecond).sum(),
                           check_dtype=False)
        ps = pd.to_datetime(
            ["20190101 16:45:50.000000000", "20190105 12:50:00.000000000", "20190105 17:45:50.000001000",
             "20170101 16:50:50.000001200",
             "20171231 10:45:50.000002000",
             "20170615 16:45:50.000002500", "20170131 16:46:50.000003000", "20170501 19:45:50.000003500",
             "20180517 12:00:00.000004000"]).to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, np.nan, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.microsecond).sum().to_pandas(),
                           pdf.groupby(pdf.date.dt.microsecond).sum(),
                           check_dtype=False)

    def test_topic_series_datetime_groupby_properties_dt_nanosecond(self):
        ps = pd.date_range('2016-12-31 16:45:50', periods=9, freq='n').to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, 74.19005356, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.nanosecond).sum().to_pandas(),
                           pdf.groupby(pdf.date.dt.nanosecond).sum(),
                           check_dtype=False)
        ps = pd.to_datetime(
            ["20190101 16:45:50.000000000", np.nan, "20190105 17:45:50.000000000", "20170101 16:50:50.0000000002",
             "20171231 10:45:50.000000005",
             "20170615 16:45:50.000002000", "20170131 16:46:50.000002500", "20170501 19:45:50.000003000",
             "20180517 12:00:00.000003500"]).to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, np.nan, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.nanosecond).sum().to_pandas().iloc[1:],
                           pdf.groupby(pdf.date.dt.nanosecond).sum(),
                           check_dtype=False)
        ps = pd.to_datetime(
            ["20190101 16:45:50.000000001", "20190105 12:50:00.000000001", "20190105 17:45:50.000000001",
             "20170101 16:50:50.000001200",
             "20171231 10:45:50.000002000",
             "20170615 16:45:50.000002500", "20170131 16:46:50.000003000", "20170501 19:45:50.000003500",
             "20180517 12:00:00.000004000"]).to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, np.nan, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.nanosecond).sum().to_pandas(),
                           pdf.groupby(pdf.date.dt.nanosecond).sum(),
                           check_dtype=False)

    def test_topic_series_datetime_groupby_properties_dt_week(self):
        ps = pd.date_range('2016-12-31 ', '2017-01-11', freq='D').to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, 74.19005356, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879, 56.48779794, 75.46484444, 4.44999789]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.week).sum().to_pandas(), pdf.groupby(pdf.date.dt.week).sum(),
                           check_dtype=False)
        ps = pd.to_datetime(
            ["20190101", np.nan, "20190105", "20170101", "20171231", "20170615", "20170131", "20170501",
             "20180517", "20170102", "20170103", "20170104"]).to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, np.nan, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879, 56.48779794, 75.46484444, 4.44999789]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.week).sum().to_pandas().iloc[1:],
                           pdf.groupby(pdf.date.dt.week).sum(),
                           check_dtype=False)
        ps = pd.to_datetime(
            ["20190101", "20190101", "20190105", "20170101", "20171231", "20170615", "20170131", "20170501",
             "20180517", "20170102", "20170103", "20170104"]).to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, np.nan, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879, 56.48779794, 75.46484444, 4.44999789]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.week).sum().to_pandas(),
                           pdf.groupby(pdf.date.dt.week).sum(),
                           check_dtype=False)

    def test_topic_series_datetime_groupby_properties_dt_weekofyear(self):
        ps = pd.date_range('2016-12-31 ', '2017-01-11', freq='D').to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, 74.19005356, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879, 56.48779794, 75.46484444, 4.44999789]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.weekofyear).sum().to_pandas(),
                           pdf.groupby(pdf.date.dt.weekofyear).sum(),
                           check_dtype=False)
        ps = pd.to_datetime(
            ["20190101", np.nan, "20190105", "20170101", "20171231", "20170615", "20170131", "20170501",
             "20180517", "20170102", "20170103", "20170104"]).to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, np.nan, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879, 56.48779794, 75.46484444, 4.44999789]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.weekofyear).sum().to_pandas().iloc[1:],
                           pdf.groupby(pdf.date.dt.weekofyear).sum(),
                           check_dtype=False)
        ps = pd.to_datetime(
            ["20190101", "20190101", "20190105", "20170101", "20171231", "20170615", "20170131", "20170501",
             "20180517", "20170102", "20170103", "20170104"]).to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, np.nan, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879, 56.48779794, 75.46484444, 4.44999789]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.weekofyear).sum().to_pandas(),
                           pdf.groupby(pdf.date.dt.weekofyear).sum(),
                           check_dtype=False)

    def test_topic_series_datetime_groupby_properties_dt_dayofweek(self):
        ps = pd.date_range('2016-12-31 ', '2017-01-11', freq='D').to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, 74.19005356, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879, 56.48779794, 75.46484444, 4.44999789]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.dayofweek).sum().to_pandas(),
                           pdf.groupby(pdf.date.dt.dayofweek).sum(),
                           check_dtype=False)
        ps = pd.to_datetime(
            ["20190101", np.nan, "20190105", "20170101", "20171231", "20170615", "20170131", "20170501",
             "20180517", "20170102", "20170103", "20170104"]).to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, np.nan, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879, 56.48779794, 75.46484444, 4.44999789]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.dayofweek).sum().iloc[1:].to_pandas(),
                           pdf.groupby(pdf.date.dt.dayofweek).sum(),
                           check_dtype=False, check_index_type=False)
        ps = pd.to_datetime(
            ["20190101", "20190101", "20190105", "20170101", "20171231", "20170615", "20170131", "20170501",
             "20180517", "20170102", "20170103", "20170104"]).to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, np.nan, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879, 56.48779794, 75.46484444, 4.44999789]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.dayofweek).sum().to_pandas(),
                           pdf.groupby(pdf.date.dt.dayofweek).sum(),
                           check_dtype=False, check_index_type=False)

    def test_topic_series_datetime_groupby_properties_dt_weekday(self):
        ps = pd.date_range('2016-12-31 ', '2017-01-11', freq='D').to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, 74.19005356, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879, 56.48779794, 75.46484444, 4.44999789]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.weekday).sum().to_pandas(),
                           pdf.groupby(pdf.date.dt.weekday).sum(),
                           check_dtype=False)
        ps = pd.to_datetime(
            ["20190101", np.nan, "20190105", "20170101", "20171231", "20170615", "20170131", "20170501",
             "20180517", "20170102", "20170103", "20170104"]).to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, np.nan, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879, 56.48779794, 75.46484444, 4.44999789]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.weekday).sum().iloc[1:].to_pandas(),
                           pdf.groupby(pdf.date.dt.weekday).sum(),
                           check_dtype=False, check_index_type=False)
        ps = pd.to_datetime(
            ["20190101", "20190101", "20190105", "20170101", "20171231", "20170615", "20170131", "20170501",
             "20180517", "20170102", "20170103", "20170104"]).to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, np.nan, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879, 56.48779794, 75.46484444, 4.44999789]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.weekday).sum().to_pandas(),
                           pdf.groupby(pdf.date.dt.weekday).sum(),
                           check_dtype=False, check_index_type=False)

    def test_topic_series_datetime_groupby_properties_dt_dayofyear(self):
        ps = pd.date_range('2016-12-31 ', '2017-01-11', freq='D').to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, 74.19005356, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879, 56.48779794, 75.46484444, 4.44999789]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.dayofyear).sum().to_pandas(),
                           pdf.groupby(pdf.date.dt.dayofyear).sum(),
                           check_dtype=False)
        ps = pd.to_datetime(
            ["20190101", np.nan, "20190105", "20170101", "20171231", "20170615", "20170131", "20170501",
             "20180517", "20170102", "20170103", "20170104"]).to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, np.nan, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879, 56.48779794, 75.46484444, 4.44999789]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.dayofyear).sum().iloc[1:].to_pandas(),
                           pdf.groupby(pdf.date.dt.dayofyear).sum(),
                           check_dtype=False, check_index_type=False)
        ps = pd.to_datetime(
            ["20190101", "20190101", "20190105", "20170101", "20171231", "20170615", "20170131", "20170501",
             "20180517", "20170102", "20170103", "20170104"]).to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, np.nan, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879, 56.48779794, 75.46484444, 4.44999789]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.dayofyear).sum().to_pandas(),
                           pdf.groupby(pdf.date.dt.dayofyear).sum(),
                           check_dtype=False, check_index_type=False)

    def test_topic_series_datetime_groupby_properties_dt_quarter(self):
        ps = pd.date_range('2016-12-31 ', periods=12, freq='2m').to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, 74.19005356, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879, 56.48779794, 75.46484444, 4.44999789]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.quarter).sum().to_pandas(),
                           pdf.groupby(pdf.date.dt.quarter).sum(),
                           check_dtype=False)
        ps = pd.to_datetime(
            ["20190101", np.nan, "20190105", "20170101", "20171231", "20170615", "20170131", "20170501",
             "20180517", "20170102", "20170103", "20170104"]).to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, np.nan, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879, 56.48779794, 75.46484444, 4.44999789]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.quarter).sum().iloc[1:].to_pandas(),
                           pdf.groupby(pdf.date.dt.quarter).sum(),
                           check_dtype=False, check_index_type=False)
        ps = pd.to_datetime(
            ["20190101", "20190101", "20190105", "20170101", "20171231", "20170615", "20170131", "20170501",
             "20180517", "20170102", "20170103", "20170104"]).to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, np.nan, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879, 56.48779794, 75.46484444, 4.44999789]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.quarter).sum().to_pandas(),
                           pdf.groupby(pdf.date.dt.quarter).sum(),
                           check_dtype=False, check_index_type=False)

    def test_topic_series_datetime_groupby_properties_dt_is_month_end(self):
        ps = pd.date_range('2016-12-31 ', periods=12, freq='31d').to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, 74.19005356, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879, 56.48779794, 75.46484444, 4.44999789]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.is_month_end).sum().to_pandas(),
                           pdf.groupby(pdf.date.dt.is_month_end).sum(),
                           check_dtype=False)
        ps = pd.to_datetime(
            ["20190101", np.nan, "20190105", "20170101", "20171231", "20170630", "20170131", "20170501",
             "20180517", "20170102", "20170103", "20170104"]).to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, np.nan, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879, 56.48779794, 75.46484444, 4.44999789]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.is_month_end).sum().iloc[1:].to_pandas(),
                           pdf.groupby(pdf.date.dt.is_month_end).sum(),
                           check_dtype=False)
        ps = pd.to_datetime(
            ["20190101", "20190101", "20190105", "20170101", "20171231", "20170630", "20170131", "20170501",
             "20180517", "20170102", "20170103", "20170104"]).to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, np.nan, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879, 56.48779794, 75.46484444, 4.44999789]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.is_month_end).sum().to_pandas(),
                           pdf.groupby(pdf.date.dt.is_month_end).sum(),
                           check_dtype=False)

    def test_topic_series_datetime_groupby_properties_dt_is_month_start(self):
        ps = pd.date_range('2016-01-01 ', periods=12, freq='31d').to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, 74.19005356, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879, 56.48779794, 75.46484444, 4.44999789]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.is_quarter_start).sum().to_pandas(),
                           pdf.groupby(pdf.date.dt.is_quarter_start).sum(),
                           check_dtype=False)
        ps = pd.to_datetime(
            ["20190101", np.nan, "20190105", "20170101", "20171231", "20170701", "20170131", "20170401",
             "20180517", "20170102", "20170103", "20170104"]).to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, np.nan, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879, 56.48779794, 75.46484444, 4.44999789]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.is_quarter_start).sum().iloc[1:].to_pandas(),
                           pdf.groupby(pdf.date.dt.is_quarter_start).sum(),
                           check_dtype=False)
        ps = pd.to_datetime(
            ["20190101", "20190101", "20190105", "20170101", "20171231", "20170701", "20170131", "20170401",
             "20180517", "20170102", "20170103", "20170104"]).to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, np.nan, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879, 56.48779794, 75.46484444, 4.44999789]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.is_quarter_start).sum().to_pandas(),
                           pdf.groupby(pdf.date.dt.is_quarter_start).sum(),
                           check_dtype=False)

    def test_topic_series_datetime_groupby_properties_dt_is_quarter_end(self):
        ps = pd.date_range('2016-01-01 ', periods=12, freq='m').to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, 74.19005356, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879, 56.48779794, 75.46484444, 4.44999789]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.is_quarter_end).sum().to_pandas(),
                           pdf.groupby(pdf.date.dt.is_quarter_end).sum(),
                           check_dtype=False)
        ps = pd.to_datetime(
            ["20190101", np.nan, "20190105", "20170101", "20171231", "20170630", "20170131", "20170430",
             "20180517", "20170102", "20170103", "20170104"]).to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, np.nan, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879, 56.48779794, 75.46484444, 4.44999789]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.is_quarter_end).sum().iloc[1:].to_pandas(),
                           pdf.groupby(pdf.date.dt.is_quarter_end).sum(),
                           check_dtype=False)
        ps = pd.to_datetime(
            ["20190101", "20190101", "20190105", "20170101", "20171231", "20170630", "20170131", "20170430",
             "20180517", "20170102", "20170103", "20170104"]).to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, np.nan, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879, 56.48779794, 75.46484444, 4.44999789]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.is_quarter_end).sum().to_pandas(),
                           pdf.groupby(pdf.date.dt.is_quarter_end).sum(),
                           check_dtype=False)

    def test_topic_series_datetime_groupby_properties_dt_is_quarter_start(self):
        ps = pd.date_range('2016-01-01 ', periods=12, freq='qs').to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, 74.19005356, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879, 56.48779794, 75.46484444, 4.44999789]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.is_quarter_start).sum().to_pandas(),
                           pdf.groupby(pdf.date.dt.is_quarter_start).sum(),
                           check_dtype=False)
        ps = pd.to_datetime(
            ["20190101", np.nan, "20190105", "20170101", "20171231", "20170701", "20170131", "20170401",
             "20180517", "20170102", "20170103", "20170104"]).to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, np.nan, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879, 56.48779794, 75.46484444, 4.44999789]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.is_quarter_start).sum().iloc[1:].to_pandas(),
                           pdf.groupby(pdf.date.dt.is_quarter_start).sum(),
                           check_dtype=False)
        ps = pd.to_datetime(
            ["20190101", "20190101", "20190105", "20170101", "20171231", "20170701", "20170131", "20170401",
             "20180517", "20170102", "20170103", "20170104"]).to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, np.nan, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879, 56.48779794, 75.46484444, 4.44999789]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.is_quarter_start).sum().to_pandas(),
                           pdf.groupby(pdf.date.dt.is_quarter_start).sum(),
                           check_dtype=False)

    def test_topic_series_datetime_groupby_properties_dt_is_year_start(self):
        ps = pd.date_range('2016-01-01 ', periods=12, freq='qs').to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, 74.19005356, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879, 56.48779794, 75.46484444, 4.44999789]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.is_year_start).sum().to_pandas(),
                           pdf.groupby(pdf.date.dt.is_year_start).sum(),
                           check_dtype=False)
        ps = pd.to_datetime(
            ["20190101", np.nan, "20190105", "20170101", "20171231", "20170630", "20170131", "20170430",
             "20180101", "20170102", "20170103", "20170104"]).to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, np.nan, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879, 56.48779794, 75.46484444, 4.44999789]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.is_year_start).sum().iloc[1:].to_pandas(),
                           pdf.groupby(pdf.date.dt.is_year_start).sum(),
                           check_dtype=False)
        ps = pd.to_datetime(
            ["20190101", "20190101", "20190105", "20170101", "20171231", "20170630", "20170131", "20170430",
             "20180101", "20170102", "20170103", "20170104"]).to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, np.nan, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879, 56.48779794, 75.46484444, 4.44999789]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.is_year_start).sum().to_pandas(),
                           pdf.groupby(pdf.date.dt.is_year_start).sum(),
                           check_dtype=False)

    def test_topic_series_datetime_groupby_properties_dt_is_year_end(self):
        ps = pd.date_range('2016-01-01 ', periods=12, freq='q').to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, 74.19005356, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879, 56.48779794, 75.46484444, 4.44999789]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.is_year_end).sum().to_pandas(),
                           pdf.groupby(pdf.date.dt.is_year_end).sum(),
                           check_dtype=False)
        ps = pd.to_datetime(
            ["20190101", np.nan, "20190105", "20171231", "20171231", "20170630", "20170131", "20170430",
             "20181231", "20170102", "20170103", "20171231"]).to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, np.nan, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879, 56.48779794, 75.46484444, 4.44999789]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.is_year_end).sum().iloc[1:].to_pandas(),
                           pdf.groupby(pdf.date.dt.is_year_end).sum(),
                           check_dtype=False)
        ps = pd.to_datetime(
            ["20190101", "20190101", "20191231", "20170101", "20171231", "20170630", "20170131", "20170430",
             "20181231", "20170102", "20170103", "20171231"]).to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, np.nan, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879, 56.48779794, 75.46484444, 4.44999789]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.is_year_end).sum().to_pandas(),
                           pdf.groupby(pdf.date.dt.is_year_end).sum(),
                           check_dtype=False)

    def test_topic_series_datetime_groupby_properties_dt_is_leap_year(self):
        ps = pd.date_range('2016-01-01 ', periods=12, freq='y').to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, 74.19005356, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879, 56.48779794, 75.46484444, 4.44999789]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.is_leap_year).sum().to_pandas(),
                           pdf.groupby(pdf.date.dt.is_leap_year).sum(),
                           check_dtype=False)
        ps = pd.to_datetime(
            ["20190101", np.nan, "20190105", "20171231", "20161231", "20170630", "20150131", "20120430",
             "20181231", "20170102", "20170103", "20171231"]).to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, np.nan, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879, 56.48779794, 75.46484444, 4.44999789]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.is_leap_year).sum().iloc[1:].to_pandas(),
                           pdf.groupby(pdf.date.dt.is_leap_year).sum(),
                           check_dtype=False)
        ps = pd.to_datetime(
            ["20190101", "20190101", "20191231", "20170101", "20161231", "20120630", "20150131", "20120430",
             "20181231", "20170102", "20170103", "20171231"]).to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, np.nan, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879, 56.48779794, 75.46484444, 4.44999789]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.is_leap_year).sum().to_pandas(),
                           pdf.groupby(pdf.date.dt.is_leap_year).sum(),
                           check_dtype=False)

    def test_topic_series_datetime_groupby_properties_dt_daysinmonth(self):
        ps = pd.date_range('2016-12-01 ', periods=12, freq='31d').to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, 74.19005356, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879, 56.48779794, 75.46484444, 4.44999789]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.daysinmonth).sum().to_pandas(),
                           pdf.groupby(pdf.date.dt.daysinmonth).sum(),
                           check_dtype=False)
        ps = pd.to_datetime(
            ["20190101", np.nan, "20190105", "20170101", "20171231", "20170615", "20170131", "20170501",
             "20180517", "20170102", "20170103", "20170104"]).to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, np.nan, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879, 56.48779794, 75.46484444, 4.44999789]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.daysinmonth).sum().iloc[1:].to_pandas(),
                           pdf.groupby(pdf.date.dt.daysinmonth).sum(),
                           check_dtype=False, check_index_type=False)
        ps = pd.to_datetime(
            ["20190101", "20190101", "20190105", "20170101", "20171231", "20170615", "20170131", "20170501",
             "20180517", "20170102", "20170103", "20170104"]).to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, np.nan, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879, 56.48779794, 75.46484444, 4.44999789]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.daysinmonth).sum().to_pandas(),
                           pdf.groupby(pdf.date.dt.daysinmonth).sum(),
                           check_dtype=False, check_index_type=False)

    def test_topic_series_datetime_groupby_properties_dt_days_in_month(self):
        ps = pd.date_range('2016-12-01 ', periods=12, freq='31d').to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, 74.19005356, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879, 56.48779794, 75.46484444, 4.44999789]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.days_in_month).sum().to_pandas(),
                           pdf.groupby(pdf.date.dt.days_in_month).sum(),
                           check_dtype=False)
        ps = pd.to_datetime(
            ["20190101", np.nan, "20190105", "20170101", "20171231", "20170615", "20170131", "20170501",
             "20180517", "20170102", "20170103", "20170104"]).to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, np.nan, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879, 56.48779794, 75.46484444, 4.44999789]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.days_in_month).sum().iloc[1:].to_pandas(),
                           pdf.groupby(pdf.date.dt.days_in_month).sum(),
                           check_dtype=False, check_index_type=False)
        ps = pd.to_datetime(
            ["20190101", "20190101", "20190105", "20170101", "20171231", "20170615", "20170131", "20170501",
             "20180517", "20170102", "20170103", "20170104"]).to_series()
        pdf = pd.DataFrame({"date": ps,
                            "value": [50.30134918, np.nan, 7.47490228, 86.98058145, 17.50379441, 2.60501574,
                                      33.78914263, 35.65197614, 93.29494879, 56.48779794, 75.46484444, 4.44999789]})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.groupby(odf.date.dt.days_in_month).sum().to_pandas(),
                           pdf.groupby(pdf.date.dt.days_in_month).sum(),
                           check_dtype=False, check_index_type=False)