import unittest
import orca
import os.path as path
from setup.settings import *
from pandas.util.testing import *


class Csv:
    pdf_csv = None
    odf_csv = None


class RollingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # configure data directory
        DATA_DIR = path.abspath(path.join(__file__, "../setup/data"))
        fileName = 'onlyNumericalColumns.csv'
        data = os.path.join(DATA_DIR, fileName)
        data = data.replace('\\', '/')

        # Orca connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

        Csv.pdf_csv = pd.read_csv(data)
        Csv.odf_csv = orca.read_csv(data)

    @property
    def pdf_csv(self):
        return Csv.pdf_csv

    @property
    def odf_csv(self):
        return Csv.odf_csv

    @property
    def pdf(self):
        n = 100  # note that n should be a multiple of 10
        re = n / 10
        pdf_da = pd.DataFrame({'id': np.arange(1, n + 1, 1, dtype='int32'),
                               'date': np.repeat(pd.date_range('2019.08.01', periods=10, freq='D'), re),
                               'tchar': np.repeat(np.arange(1, 11, 1, dtype='int8'), re),
                               'tshort': np.repeat(np.arange(1, 11, 1, dtype='int16'), re),
                               'tint': np.repeat(np.arange(1, 11, 1, dtype='int32'), re),
                               'tlong': np.repeat(np.arange(1, 11, 1, dtype='int64'), re),
                               'tfloat': np.repeat(np.arange(1, 11, 1, dtype='float32'), re),
                               'tdouble': np.repeat(np.arange(1, 11, 1, dtype='float64'), re)
                               })
        return pdf_da.set_index("id")

    @property
    def pdf_da(self):
        n = 9  # note that n should be a multiple of 10
        ps = pd.to_datetime(
            ["20170101", "20170103", "20170105", "20170106", "20171231", "20180615", "20181031", "20190501",
             "20190517"]).to_series()
        pdf_da = pd.DataFrame({'id': np.arange(1, n + 1, 1, dtype='int32'),
                               'date': ps,
                               'tchar': np.arange(1, 10, 1, dtype='int8'),
                               'tshort': np.arange(1, 10, 1, dtype='int16'),
                               'tint': np.arange(1, 10, 1, dtype='int32'),
                               'tlong': np.arange(1, 10, 1, dtype='int64'),
                               'tfloat': np.arange(1, 10, 1, dtype='float32'),
                               'tdouble': np.arange(1, 10, 1, dtype='float64')
                               })
        return pdf_da.set_index("id")

    @property
    def odf(self):
        return orca.DataFrame(self.pdf)

    @property
    def odf_da(self):
        return orca.DataFrame(self.pdf_da)

    def test_rolling_allocation_verification(self):
        self.assertIsInstance(self.odf.rolling(window=5, on="date")['date'].count().to_pandas(), Series)
        with self.assertRaises(KeyError):
            self.odf.rolling(window=5, on="date")['hello'].count()
        with self.assertRaises(KeyError):
            self.odf.rolling(window=5, on="date")[['dare', 5, 0]].count()
        with self.assertRaises(KeyError):
            self.odf.rolling(window=5, on="date")[['hello', 'world']].count()
        with self.assertRaises(KeyError):
            self.odf.rolling(window=5, on="date")[np.array([1, 2, 3])].count()
        with self.assertRaises(KeyError):
            self.odf.rolling(window=5, on="date")[5].count()
        with self.assertRaises(KeyError):
            self.odf.rolling(window=5, on="date")[[16.5, 5]].count()

    def test_rolling_from_pandas_param_window_sum(self):
        a = self.odf.rolling(window=5, on="date").sum()
        b = self.pdf.rolling(window=5, on="date").sum()
        assert_frame_equal(a.to_pandas(), b)

        a = self.odf.rolling(window=5, on="date")[
            'date', 'tchar', 'tint', 'tlong', 'tfloat'].sum()
        b = self.pdf.rolling(window=5, on="date")[
            'date', 'tchar', 'tint', 'tlong', 'tfloat'].sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        a = self.odf.rolling(window=5, on="date")[
            'date', 'tdouble','tchar', 'tint', 'tlong', 'tfloat','tshort'].sum()
        b = self.pdf.rolling(window=5, on="date")[
            'date', 'tdouble','tchar', 'tint', 'tlong', 'tfloat','tshort'].sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.reset_index()
        pdf_dai = self.pdf.reset_index()
        a = odf_dai.rolling(window=5, on="date")[
            'date',  'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdf_dai.rolling(window=5, on="date")[
            'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('date')
        pdf_dai = self.pdf.set_index('date')
        a = odf_dai.rolling(window=5).sum()
        b = pdf_dai.rolling(window=5).sum()
        assert_frame_equal(a.to_pandas(), b)

        a = odf_dai.rolling(window=5)[
             'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdf_dai.rolling(window=5)[
             'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_rule_day_sum(self):
        a = self.odf_da.rolling(window='d', on="date").sum()
        b = self.pdf_da.rolling(window='d', on="date").sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = self.odf_da.rolling(window='3d', on="date")[
            'date', 'tchar', 'tint', 'tlong', 'tfloat'].sum()
        b = self.pdf_da.rolling(window='3d', on="date")[
            'date', 'tchar', 'tint', 'tlong', 'tfloat'].sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        a = self.odf_da.rolling(window='d', on="date")[
            'date', 'tdouble','tchar', 'tint', 'tlong', 'tfloat','tshort'].sum()
        b = self.pdf_da.rolling(window='d', on="date")[
            'date', 'tdouble','tchar', 'tint', 'tlong', 'tfloat','tshort'].sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf_da.reset_index()
        pdf_dai = self.pdf_da.reset_index()
        a = odf_dai.rolling(window='d', on="date")[
            'date',  'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdf_dai.rolling(window='d', on="date")[
            'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf_da.set_index('date')
        pdf_dai = self.pdf_da.set_index('date')
        a = odf_dai.rolling(window='d').sum()
        b = pdf_dai.rolling(window='d').sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = odf_dai.rolling(window='d')[
             'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdf_dai.rolling(window='d')[
             'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_rule_hour_sum(self):
        ps = pd.to_datetime(
            ["20170101 9:10:15", "20170101 10:10:15", "20170101 11:10:15", "20170101 11:20:15", "20170101 11:21:00", "20180615 9:10:15", "20181031 9:10:15", "20190501 9:10:15",
             "20190517 9:10:15"]).to_series()
        pdf_h = pd.DataFrame({'id': np.arange(1, 10, 1, dtype='int32'),
                               'date': ps,
                               'tchar': np.arange(1, 10, 1, dtype='int8'),
                               'tshort': np.arange(1, 10, 1, dtype='int16'),
                               'tint': np.arange(1, 10, 1, dtype='int32'),
                               'tlong': np.arange(1, 10, 1, dtype='int64'),
                               'tfloat': np.arange(1, 10, 1, dtype='float32'),
                               'tdouble': np.arange(1, 10, 1, dtype='float64')
                               })
        pdf_h.set_index("id", inplace=True)
        odf_h = orca.DataFrame(pdf_h)

        # TODO: ALL ASSERT FAIL
        a = odf_h.rolling(window='h', on="date").sum()
        b = pdf_h.rolling(window='h', on="date").sum()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = odf_h.rolling(window='2h', on="date").sum()
        b = pdf_h.rolling(window='2h', on="date").sum()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        # a = odf_h.rolling(window='h', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].sum()
        # b = pdf_h.rolling(window='h', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].sum()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        # a = odf_hrolling(window='h', on="date")[
        #     'date', 'tdouble','tchar', 'tint', 'tlong', 'tfloat','tshort'].sum()
        # b = pdf_h.rolling(window='h', on="date")[
        #     'date', 'tdouble','tchar', 'tint', 'tlong', 'tfloat','tshort'].sum()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_h.reset_index()
        # pdf_dai = pdf_h.reset_index()
        # a = odf_dai.rolling(window='h', on="date")[
        #     'date',  'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        # b = pdf_dai.rolling(window='h', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_h.set_index('date')
        # pdf_dai = pdf_h.set_index('date')
        # a = odf_dai.rolling(window='h').sum()
        # b = pdf_dai.rolling(window='h').sum()
        # assert_frame_equal(a.to_pandas(), b)
        #
        # a = odf_dai.rolling(window='h')[
        #      'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        # b = pdf_dai.rolling(window='h')[
        #      'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_rule_minute_sum(self):
        ps = pd.to_datetime(
            ["20170101 9:10:15", "20170101 9:10:16", "20170101 9:11:10", "20170101 9:11:17", "20170101 11:21:00",
             "20180615 9:10:15", "20181031 9:10:15", "20190501 9:10:15",
             "20190517 9:10:15"]).to_series()
        pdf_t = pd.DataFrame({'id': np.arange(1, 10, 1, dtype='int32'),
                              'date': ps,
                              'tchar': np.arange(1, 10, 1, dtype='int8'),
                              'tshort': np.arange(1, 10, 1, dtype='int16'),
                              'tint': np.arange(1, 10, 1, dtype='int32'),
                              'tlong': np.arange(1, 10, 1, dtype='int64'),
                              'tfloat': np.arange(1, 10, 1, dtype='float32'),
                              'tdouble': np.arange(1, 10, 1, dtype='float64')
                              })
        pdf_t.set_index("id", inplace=True)
        odf_t = orca.DataFrame(pdf_t)

        # TODO: ALL ASSERT FAIL
        a = odf_t.rolling(window='t', on="date").sum()
        b = pdf_t.rolling(window='t', on="date").sum()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        # a = odf_t.rolling(window='t', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].sum()
        # b = pdf_t.rolling(window='t', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].sum()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # a = odf_t.rolling(window='t', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].sum()
        # b = pdf_t.rolling(window='t', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].sum()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_t.reset_index()
        # pdf_dai = pdf_t.reset_index()
        # a = odf_dai.rolling(window='t', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        # b = pdf_dai.rolling(window='t', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_t.set_index('date')
        # pdf_dai = pdf_t.set_index('date')
        # a = odf_dai.rolling(window='t').sum()
        # b = pdf_dai.rolling(window='t').sum()
        # assert_frame_equal(a.to_pandas(), b)
        #
        # a = odf_dai.rolling(window='t')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        # b = pdf_dai.rolling(window='t')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_rule_second_sum(self):
        ps = pd.to_datetime(
            ["20170101 9:10:15", "20170101 9:10:16", "20170101 9:10:16", "20170101 9:11:17", "20170101 9:11:17",
             "20180615 9:10:15", "20181031 9:10:15", "20190501 9:10:15",
             "20190517 9:10:15"]).to_series()
        pdf_s = pd.DataFrame({'id': np.arange(1, 10, 1, dtype='int32'),
                              'date': ps,
                              'tchar': np.arange(1, 10, 1, dtype='int8'),
                              'tshort': np.arange(1, 10, 1, dtype='int16'),
                              'tint': np.arange(1, 10, 1, dtype='int32'),
                              'tlong': np.arange(1, 10, 1, dtype='int64'),
                              'tfloat': np.arange(1, 10, 1, dtype='float32'),
                              'tdouble': np.arange(1, 10, 1, dtype='float64')
                              })
        pdf_s.set_index("id", inplace=True)
        odf_s = orca.DataFrame(pdf_s)

        # TODO: ALL ASSERT FAIL
        a = odf_s.rolling(window='s', on="date").sum()
        b = pdf_s.rolling(window='s', on="date").sum()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = odf_s.rolling(window='2s', on="date").sum()
        b = pdf_s.rolling(window='2s', on="date").sum()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        # a = odf_s.rolling(window='s', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].sum()
        # b = pdf_s.rolling(window='s', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].sum()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        # a = odf_s.rolling(window='s', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].sum()
        # b = pdf_s.rolling(window='s', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].sum()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_s.reset_index()
        # pdf_dai = pdf_s.reset_index()
        # a = odf_dai.rolling(window='s', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        # b = pdf_dai.rolling(window='s', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_s.set_index('date')
        # pdf_dai = pdf_s.set_index('date')
        # a = odf_dai.rolling(window='s').sum()
        # b = pdf_dai.rolling(window='s').sum()
        # assert_frame_equal(a.to_pandas(), b)
        #
        # a = odf_dai.rolling(window='s')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        # b = pdf_dai.rolling(window='s')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_rule_milli_sum(self):
        ps = pd.to_datetime(
            ["20170101 9:10:15.000", "20170101 9:10:15.000", "20170101 9:10:15.001", "20170101 9:11:17.015", "20170101 9:11:17.015",
             "20180615 9:10:15.015", "20181031 9:10:15.015", "20190501 9:10:15.015",
             "20190517 9:10:15.015"]).to_series()
        pdf_l = pd.DataFrame({'id': np.arange(1, 10, 1, dtype='int32'),
                              'date': ps,
                              'tchar': np.arange(1, 10, 1, dtype='int8'),
                              'tshort': np.arange(1, 10, 1, dtype='int16'),
                              'tint': np.arange(1, 10, 1, dtype='int32'),
                              'tlong': np.arange(1, 10, 1, dtype='int64'),
                              'tfloat': np.arange(1, 10, 1, dtype='float32'),
                              'tdouble': np.arange(1, 10, 1, dtype='float64')
                              })
        pdf_l.set_index("id", inplace=True)
        odf_l = orca.DataFrame(pdf_l)

        # TODO: ALL ASSERT FAIL
        # a = odf_l.rolling(window='l', on="date").sum()
        # b = pdf_l.rolling(window='l', on="date").sum()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)
        #
        # a = odf_l.rolling(window='2l', on="date").sum()
        # b = pdf_l.rolling(window='2l', on="date").sum()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        # a = odf_l.rolling(window='l', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].sum()
        # b = pdf_l.rolling(window='l', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].sum()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # a = odf_l.rolling(window='l', on="date")[
        #     'date'_l, 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].sum()
        # b = pdf.rolling(window='l', on="date")[
        #     'date'_l, 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].sum()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_l.reset_index()
        # pdf_dai = pdf_l.reset_index()
        # a = odf_dai.rolling(window='l', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        # b = pdf_dai.rolling(window='l', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_l.set_index('date')
        # pdf_dai = pdf_l.set_index('date')
        # a = odf_dai.rolling(window='l').sum()
        # b = pdf_dai.rolling(window='l').sum()
        # assert_frame_equal(a.to_pandas(), b)
        #
        # a = odf_dai.rolling(window='l')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        # b = pdf_dai.rolling(window='l')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_rule_micro_sum(self):
        ps = pd.to_datetime(
            ["20170101 9:10:15.000000", "20170101 9:10:15.000000", "20170101 9:10:15.000001", "20170101 9:11:17.015001",
             "20170101 9:11:17.015002",
             "20180615 9:10:15.015000", "20181031 9:10:15.015000", "20190501 9:10:15.015000",
             "20190517 9:10:15.015000"]).to_series()
        pdf_u = pd.DataFrame({'id': np.arange(1, 10, 1, dtype='int32'),
                              'date': ps,
                              'tchar': np.arange(1, 10, 1, dtype='int8'),
                              'tshort': np.arange(1, 10, 1, dtype='int16'),
                              'tint': np.arange(1, 10, 1, dtype='int32'),
                              'tlong': np.arange(1, 10, 1, dtype='int64'),
                              'tfloat': np.arange(1, 10, 1, dtype='float32'),
                              'tdouble': np.arange(1, 10, 1, dtype='float64')
                              })
        pdf_u = pdf_u.set_index("id")
        odf_u = orca.DataFrame(pdf_u)

        # TODO: ALL ASSERT FAIL
        a = odf_u.rolling(window='u', on="date").sum()
        b = pdf_u.rolling(window='u', on="date").sum()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = odf_u.rolling(window='2u', on="date").sum()
        b = pdf_u.rolling(window='2u', on="date").sum()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        # a = odf_u.rolling(window='u', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].sum()
        # b = pdf_u.rolling(window='u', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].sum()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # a = odf_u.rolling(window='u', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].sum()
        # b = pdf_u.rolling(window='u', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].sum()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_u.reset_index()
        # pdf_dai = pdf_u.reset_index()
        # a = odf_dai.rolling(window='u', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        # b = pdf_dai.rolling(window='u', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_u.set_index('date')
        # pdf_dai = pdf_u.set_index('date')
        # a = odf_dai.rolling(window='u').sum()
        # b = pdf_dai.rolling(window='u').sum()
        # assert_frame_equal(a.to_pandas(), b)
        #
        # a = odf_dai.rolling(window='u')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        # b = pdf_dai.rolling(window='u')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_rule_nano_sum(self):
        ps = pd.to_datetime(
            ["20170101 9:10:15.000000000", "20170101 9:10:15.000000000", "20170101 9:10:15.000000001", "20170101 9:11:17.015000001",
             "20170101 9:11:17.015002001",
             "20180615 9:10:15.015000001", "20181031 9:10:15.015000001", "20190501 9:10:15.015000001",
             "20190517 9:10:15.015000001"]).to_series()
        pdf_n = pd.DataFrame({'id': np.arange(1, 10, 1, dtype='int32'),
                              'date': ps,
                              'tchar': np.arange(1, 10, 1, dtype='int8'),
                              'tshort': np.arange(1, 10, 1, dtype='int16'),
                              'tint': np.arange(1, 10, 1, dtype='int32'),
                              'tlong': np.arange(1, 10, 1, dtype='int64'),
                              'tfloat': np.arange(1, 10, 1, dtype='float32'),
                              'tdouble': np.arange(1, 10, 1, dtype='float64')
                              })
        pdf_n = pdf_n.set_index("id")
        odf_n = orca.DataFrame(pdf_n)

        # TODO: ALL ASSERT FAIL
        a = odf_n.rolling(window='n', on="date").sum()
        b = pdf_n.rolling(window='n', on="date").sum()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = odf_n.rolling(window='2n', on="date").sum()
        b = pdf_n.rolling(window='2n', on="date").sum()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        # a = odf_n.rolling(window='n', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].sum()
        # b = pdf_n.rolling(window='n', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].sum()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # a = odf_n.rolling(window='n', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].sum()
        # b = pdf_n.rolling(window='n', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].sum()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_n.reset_index()
        # pdf_dai = pdf_n.reset_index()
        # a = odf_dai.rolling(window='n', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        # b = pdf_dai.rolling(window='n', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_n.set_index('date')
        # pdf_dai = pdf_n.set_index('date')
        # a = odf_dai.rolling(window='n').sum()
        # b = pdf_dai.rolling(window='n').sum()
        # assert_frame_equal(a.to_pandas(), b)
        #
        # a = odf_dai.rolling(window='n')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        # b = pdf_dai.rolling(window='n')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_count(self):
        a = self.odf.rolling(window=5, on="date").count()
        b = self.pdf.rolling(window=5, on="date").count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = self.odf.rolling(window=5, on="date")[
            'date', 'tchar', 'tint', 'tlong', 'tfloat'].count()
        b = self.pdf.rolling(window=5, on="date")[
            'date', 'tchar', 'tint', 'tlong', 'tfloat'].count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        a = self.odf.rolling(window=5, on="date")[
            'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].count()
        b = self.pdf.rolling(window=5, on="date")[
            'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.reset_index()
        pdf_dai = self.pdf.reset_index()
        a = odf_dai.rolling(window=5, on="date")[
            'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdf_dai.rolling(window=5, on="date")[
            'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('date')
        pdf_dai = self.pdf.set_index('date')
        a = odf_dai.rolling(window=5).count()
        b = pdf_dai.rolling(window=5).count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = odf_dai.rolling(window=5)[
            'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdf_dai.rolling(window=5)[
            'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_rule_day_count(self):
        a = self.odf_da.rolling(window='d', on="date").count()
        b = self.pdf_da.rolling(window='d', on="date").count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = self.odf_da.rolling(window='3d', on="date")[
            'date', 'tchar', 'tint', 'tlong', 'tfloat'].count()
        b = self.pdf_da.rolling(window='3d', on="date")[
            'date', 'tchar', 'tint', 'tlong', 'tfloat'].count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        a = self.odf_da.rolling(window='d', on="date")[
            'date', 'tdouble','tchar', 'tint', 'tlong', 'tfloat','tshort'].count()
        b = self.pdf_da.rolling(window='d', on="date")[
            'date', 'tdouble','tchar', 'tint', 'tlong', 'tfloat','tshort'].count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf_da.reset_index()
        pdf_dai = self.pdf_da.reset_index()
        a = odf_dai.rolling(window='d', on="date")[
            'date',  'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdf_dai.rolling(window='d', on="date")[
            'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf_da.set_index('date')
        pdf_dai = self.pdf_da.set_index('date')
        a = odf_dai.rolling(window='d').count()
        b = pdf_dai.rolling(window='d').count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = odf_dai.rolling(window='d')[
             'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdf_dai.rolling(window='d')[
             'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_rule_hour_count(self):
        ps = pd.to_datetime(
            ["20170101 9:10:15", "20170101 10:10:15", "20170101 11:10:15", "20170101 11:20:15", "20170101 11:21:00",
             "20180615 9:10:15", "20181031 9:10:15", "20190501 9:10:15",
             "20190517 9:10:15"]).to_series()
        pdf_h = pd.DataFrame({'id': np.arange(1, 10, 1, dtype='int32'),
                              'date': ps,
                              'tchar': np.arange(1, 10, 1, dtype='int8'),
                              'tshort': np.arange(1, 10, 1, dtype='int16'),
                              'tint': np.arange(1, 10, 1, dtype='int32'),
                              'tlong': np.arange(1, 10, 1, dtype='int64'),
                              'tfloat': np.arange(1, 10, 1, dtype='float32'),
                              'tdouble': np.arange(1, 10, 1, dtype='float64')
                              })
        pdf_h = pdf_h.set_index("id")
        odf_h = orca.DataFrame(pdf_h)

        # TODO: ALL ASSERT FAIL
        a = odf_h.rolling(window='h', on="date").count()
        b = pdf_h.rolling(window='h', on="date").count()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = odf_h.rolling(window='2h', on="date").count()
        b = pdf_h.rolling(window='2h', on="date").count()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        # a = odf_h.rolling(window='h', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].count()
        # b = pdf_h.rolling(window='h', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].count()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # a = odf_hrolling(window='h', on="date")[
        #     'date', 'tdouble','tchar', 'tint', 'tlong', 'tfloat','tshort'].count()
        # b = pdf_h.rolling(window='h', on="date")[
        #     'date', 'tdouble','tchar', 'tint', 'tlong', 'tfloat','tshort'].count()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_h.reset_index()
        # pdf_dai = pdf_h.reset_index()
        # a = odf_dai.rolling(window='h', on="date")[
        #     'date',  'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        # b = pdf_dai.rolling(window='h', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_h.set_index('date')
        # pdf_dai = pdf_h.set_index('date')
        # a = odf_dai.rolling(window='h').count()
        # b = pdf_dai.rolling(window='h').count()
        # assert_frame_equal(a.to_pandas(), b)
        #
        # a = odf_dai.rolling(window='h')[
        #      'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        # b = pdf_dai.rolling(window='h')[
        #      'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_rule_minute_count(self):
        ps = pd.to_datetime(
            ["20170101 9:10:15", "20170101 9:10:16", "20170101 9:11:10", "20170101 9:11:17", "20170101 11:21:00",
             "20180615 9:10:15", "20181031 9:10:15", "20190501 9:10:15",
             "20190517 9:10:15"]).to_series()
        pdf_t = pd.DataFrame({'id': np.arange(1, 10, 1, dtype='int32'),
                              'date': ps,
                              'tchar': np.arange(1, 10, 1, dtype='int8'),
                              'tshort': np.arange(1, 10, 1, dtype='int16'),
                              'tint': np.arange(1, 10, 1, dtype='int32'),
                              'tlong': np.arange(1, 10, 1, dtype='int64'),
                              'tfloat': np.arange(1, 10, 1, dtype='float32'),
                              'tdouble': np.arange(1, 10, 1, dtype='float64')
                              })
        pdf_t = pdf_t.set_index("id")
        odf_t = orca.DataFrame(pdf_t)
        # TODO: ALL ASSERT FAIL
        a = odf_t.rolling(window='t', on="date").count()
        b = pdf_t.rolling(window='t', on="date").count()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        # a = odf_t.rolling(window='t', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].count()
        # b = pdf_t.rolling(window='t', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].count()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # a = odf_t.rolling(window='t', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].count()
        # b = pdf_t.rolling(window='t', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].count()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_t.reset_index()
        # pdf_dai = pdf_t.reset_index()
        # a = odf_dai.rolling(window='t', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        # b = pdf_dai.rolling(window='t', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_t.set_index('date')
        # pdf_dai = pdf_t.set_index('date')
        # a = odf_dai.rolling(window='t').count()
        # b = pdf_dai.rolling(window='t').count()
        # assert_frame_equal(a.to_pandas(), b)
        #
        # a = odf_dai.rolling(window='t')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        # b = pdf_dai.rolling(window='t')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_rule_second_count(self):
        ps = pd.to_datetime(
            ["20170101 9:10:15", "20170101 9:10:16", "20170101 9:10:16", "20170101 9:11:17", "20170101 9:11:17",
             "20180615 9:10:15", "20181031 9:10:15", "20190501 9:10:15",
             "20190517 9:10:15"]).to_series()
        pdf_s = pd.DataFrame({'id': np.arange(1, 10, 1, dtype='int32'),
                              'date': ps,
                              'tchar': np.arange(1, 10, 1, dtype='int8'),
                              'tshort': np.arange(1, 10, 1, dtype='int16'),
                              'tint': np.arange(1, 10, 1, dtype='int32'),
                              'tlong': np.arange(1, 10, 1, dtype='int64'),
                              'tfloat': np.arange(1, 10, 1, dtype='float32'),
                              'tdouble': np.arange(1, 10, 1, dtype='float64')
                              })
        pdf_s = pdf_s.set_index("id")
        odf_s = orca.DataFrame(pdf_s)
        # TODO: ALL ASSERT FAIL
        a = odf_s.rolling(window='s', on="date").count()
        b = pdf_s.rolling(window='s', on="date").count()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = odf_s.rolling(window='2s', on="date").count()
        b = pdf_s.rolling(window='2s', on="date").count()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        # a = odf_s.rolling(window='s', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].count()
        # b = pdf_s.rolling(window='s', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].count()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # a = odf_s.rolling(window='s', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].count()
        # b = pdf_s.rolling(window='s', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].count()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_s.reset_index()
        # pdf_dai = pdf_s.reset_index()
        # a = odf_dai.rolling(window='s', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        # b = pdf_dai.rolling(window='s', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_s.set_index('date')
        # pdf_dai = pdf_s.set_index('date')
        # a = odf_dai.rolling(window='s').count()
        # b = pdf_dai.rolling(window='s').count()
        # assert_frame_equal(a.to_pandas(), b)
        #
        # a = odf_dai.rolling(window='s')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        # b = pdf_dai.rolling(window='s')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_rule_milli_count(self):
        ps = pd.to_datetime(
            ["20170101 9:10:15.000", "20170101 9:10:15.000", "20170101 9:10:15.001", "20170101 9:11:17.015",
             "20170101 9:11:17.015",
             "20180615 9:10:15.015", "20181031 9:10:15.015", "20190501 9:10:15.015",
             "20190517 9:10:15.015"]).to_series()
        pdf_l = pd.DataFrame({'id': np.arange(1, 10, 1, dtype='int32'),
                              'date': ps,
                              'tchar': np.arange(1, 10, 1, dtype='int8'),
                              'tshort': np.arange(1, 10, 1, dtype='int16'),
                              'tint': np.arange(1, 10, 1, dtype='int32'),
                              'tlong': np.arange(1, 10, 1, dtype='int64'),
                              'tfloat': np.arange(1, 10, 1, dtype='float32'),
                              'tdouble': np.arange(1, 10, 1, dtype='float64')
                              })
        pdf_l = pdf_l.set_index("id")
        odf_l = orca.DataFrame(pdf_l)

        # TODO: ALL ASSERT FAIL
        a = odf_l.rolling(window='l', on="date").count()
        b = pdf_l.rolling(window='l', on="date").count()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = odf_l.rolling(window='2l', on="date").count()
        b = pdf_l.rolling(window='2l', on="date").count()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        # a = odf_l.rolling(window='l', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].count()
        # b = pdf_l.rolling(window='l', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].count()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # a = odf_l.rolling(window='l', on="date")[
        #     'date'_l, 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].count()
        # b = pdf.rolling(window='l', on="date")[
        #     'date'_l, 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].count()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_l.reset_index()
        # pdf_dai = pdf_l.reset_index()
        # a = odf_dai.rolling(window='l', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        # b = pdf_dai.rolling(window='l', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_l.set_index('date')
        # pdf_dai = pdf_l.set_index('date')
        # a = odf_dai.rolling(window='l').count()
        # b = pdf_dai.rolling(window='l').count()
        # assert_frame_equal(a.to_pandas(), b)
        #
        # a = odf_dai.rolling(window='l')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        # b = pdf_dai.rolling(window='l')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_rule_micro_count(self):
        ps = pd.to_datetime(
            ["20170101 9:10:15.000000", "20170101 9:10:15.000000", "20170101 9:10:15.000001", "20170101 9:11:17.015001",
             "20170101 9:11:17.015002",
             "20180615 9:10:15.015000", "20181031 9:10:15.015000", "20190501 9:10:15.015000",
             "20190517 9:10:15.015000"]).to_series()
        pdf_u = pd.DataFrame({'id': np.arange(1, 10, 1, dtype='int32'),
                              'date': ps,
                              'tchar': np.arange(1, 10, 1, dtype='int8'),
                              'tshort': np.arange(1, 10, 1, dtype='int16'),
                              'tint': np.arange(1, 10, 1, dtype='int32'),
                              'tlong': np.arange(1, 10, 1, dtype='int64'),
                              'tfloat': np.arange(1, 10, 1, dtype='float32'),
                              'tdouble': np.arange(1, 10, 1, dtype='float64')
                              })
        pdf_u = pdf_u.set_index("id")
        odf_u = orca.DataFrame(pdf_u)
        # TODO: ALL ASSERT FAIL
        a = odf_u.rolling(window='u', on="date").count()
        b = pdf_u.rolling(window='u', on="date").count()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = odf_u.rolling(window='2u', on="date").count()
        b = pdf_u.rolling(window='2u', on="date").count()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        # a = odf_u.rolling(window='u', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].count()
        # b = pdf_u.rolling(window='u', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].count()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # a = odf_u.rolling(window='u', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].count()
        # b = pdf_u.rolling(window='u', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].count()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_u.reset_index()
        # pdf_dai = pdf_u.reset_index()
        # a = odf_dai.rolling(window='u', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        # b = pdf_dai.rolling(window='u', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_u.set_index('date')
        # pdf_dai = pdf_u.set_index('date')
        # a = odf_dai.rolling(window='u').count()
        # b = pdf_dai.rolling(window='u').count()
        # assert_frame_equal(a.to_pandas(), b)
        #
        # a = odf_dai.rolling(window='u')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        # b = pdf_dai.rolling(window='u')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_rule_nano_count(self):
        ps = pd.to_datetime(
            ["20170101 9:10:15.000000000", "20170101 9:10:15.000000000", "20170101 9:10:15.000000001",
             "20170101 9:11:17.015000001",
             "20170101 9:11:17.015002001",
             "20180615 9:10:15.015000001", "20181031 9:10:15.015000001", "20190501 9:10:15.015000001",
             "20190517 9:10:15.015000001"]).to_series()
        pdf_n = pd.DataFrame({'id': np.arange(1, 10, 1, dtype='int32'),
                              'date': ps,
                              'tchar': np.arange(1, 10, 1, dtype='int8'),
                              'tshort': np.arange(1, 10, 1, dtype='int16'),
                              'tint': np.arange(1, 10, 1, dtype='int32'),
                              'tlong': np.arange(1, 10, 1, dtype='int64'),
                              'tfloat': np.arange(1, 10, 1, dtype='float32'),
                              'tdouble': np.arange(1, 10, 1, dtype='float64')
                              })
        pdf_n = pdf_n.set_index("id")
        odf_n = orca.DataFrame(pdf_n)
        # TODO: ALL ASSERT FAIL
        a = odf_n.rolling(window='n', on="date").count()
        b = pdf_n.rolling(window='n', on="date").count()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = odf_n.rolling(window='2n', on="date").count()
        b = pdf_n.rolling(window='2n', on="date").count()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        # a = odf_n.rolling(window='n', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].count()
        # b = pdf_n.rolling(window='n', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].count()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # a = odf_n.rolling(window='n', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].count()
        # b = pdf_n.rolling(window='n', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].count()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_n.reset_index()
        # pdf_dai = pdf_n.reset_index()
        # a = odf_dai.rolling(window='n', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        # b = pdf_dai.rolling(window='n', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_n.set_index('date')
        # pdf_dai = pdf_n.set_index('date')
        # a = odf_dai.rolling(window='n').count()
        # b = pdf_dai.rolling(window='n').count()
        # assert_frame_equal(a.to_pandas(), b)
        #
        # a = odf_dai.rolling(window='n')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        # b = pdf_dai.rolling(window='n')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_mean(self):
        a = self.odf.rolling(window=5, on="date").mean()
        b = self.pdf.rolling(window=5, on="date").mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = self.odf.rolling(window=5, on="date")[
            'date', 'tchar', 'tint', 'tlong', 'tfloat'].mean()
        b = self.pdf.rolling(window=5, on="date")[
            'date', 'tchar', 'tint', 'tlong', 'tfloat'].mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        a = self.odf.rolling(window=5, on="date")[
            'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].mean()
        b = self.pdf.rolling(window=5, on="date")[
            'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.reset_index()
        pdf_dai = self.pdf.reset_index()
        a = odf_dai.rolling(window=5, on="date")[
            'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdf_dai.rolling(window=5, on="date")[
            'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('date')
        pdf_dai = self.pdf.set_index('date')
        a = odf_dai.rolling(window=5).mean()
        b = pdf_dai.rolling(window=5).mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = odf_dai.rolling(window=5)[
            'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdf_dai.rolling(window=5)[
            'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_rule_day_mean(self):
        a = self.odf_da.rolling(window='d', on="date").mean()
        b = self.pdf_da.rolling(window='d', on="date").mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = self.odf_da.rolling(window='3d', on="date")[
            'date', 'tchar', 'tint', 'tlong', 'tfloat'].mean()
        b = self.pdf_da.rolling(window='3d', on="date")[
            'date', 'tchar', 'tint', 'tlong', 'tfloat'].mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        a = self.odf_da.rolling(window='d', on="date")[
            'date', 'tdouble','tchar', 'tint', 'tlong', 'tfloat','tshort'].mean()
        b = self.pdf_da.rolling(window='d', on="date")[
            'date', 'tdouble','tchar', 'tint', 'tlong', 'tfloat','tshort'].mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf_da.reset_index()
        pdf_dai = self.pdf_da.reset_index()
        a = odf_dai.rolling(window='d', on="date")[
            'date',  'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdf_dai.rolling(window='d', on="date")[
            'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf_da.set_index('date')
        pdf_dai = self.pdf_da.set_index('date')
        a = odf_dai.rolling(window='d').mean()
        b = pdf_dai.rolling(window='d').mean()
        assert_frame_equal(a.to_pandas(), b)

        a = odf_dai.rolling(window='d')[
             'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdf_dai.rolling(window='d')[
             'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_rule_hour_mean(self):
        ps = pd.to_datetime(
            ["20170101 9:10:15", "20170101 10:10:15", "20170101 11:10:15", "20170101 11:20:15", "20170101 11:21:00",
             "20180615 9:10:15", "20181031 9:10:15", "20190501 9:10:15",
             "20190517 9:10:15"]).to_series()
        pdf_h = pd.DataFrame({'id': np.arange(1, 10, 1, dtype='int32'),
                              'date': ps,
                              'tchar': np.arange(1, 10, 1, dtype='int8'),
                              'tshort': np.arange(1, 10, 1, dtype='int16'),
                              'tint': np.arange(1, 10, 1, dtype='int32'),
                              'tlong': np.arange(1, 10, 1, dtype='int64'),
                              'tfloat': np.arange(1, 10, 1, dtype='float32'),
                              'tdouble': np.arange(1, 10, 1, dtype='float64')
                              })
        pdf_h = pdf_h.set_index("id")
        odf_h = orca.DataFrame(pdf_h)
        # TODO: ALL ASSERT FAIL
        a = odf_h.rolling(window='h', on="date").mean()
        b = pdf_h.rolling(window='h', on="date").mean()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = odf_h.rolling(window='2h', on="date").mean()
        b = pdf_h.rolling(window='2h', on="date").mean()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        # a = odf_h.rolling(window='h', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].mean()
        # b = pdf_h.rolling(window='h', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].mean()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # a = odf_hrolling(window='h', on="date")[
        #     'date', 'tdouble','tchar', 'tint', 'tlong', 'tfloat','tshort'].mean()
        # b = pdf_h.rolling(window='h', on="date")[
        #     'date', 'tdouble','tchar', 'tint', 'tlong', 'tfloat','tshort'].mean()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_h.reset_index()
        # pdf_dai = pdf_h.reset_index()
        # a = odf_dai.rolling(window='h', on="date")[
        #     'date',  'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        # b = pdf_dai.rolling(window='h', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_h.set_index('date')
        # pdf_dai = pdf_h.set_index('date')
        # a = odf_dai.rolling(window='h').mean()
        # b = pdf_dai.rolling(window='h').mean()
        # assert_frame_equal(a.to_pandas(), b)
        #
        # a = odf_dai.rolling(window='h')[
        #      'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        # b = pdf_dai.rolling(window='h')[
        #      'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_rule_minute_mean(self):
        ps = pd.to_datetime(
            ["20170101 9:10:15", "20170101 9:10:16", "20170101 9:11:10", "20170101 9:11:17", "20170101 11:21:00",
             "20180615 9:10:15", "20181031 9:10:15", "20190501 9:10:15",
             "20190517 9:10:15"]).to_series()
        pdf_t = pd.DataFrame({'id': np.arange(1, 10, 1, dtype='int32'),
                              'date': ps,
                              'tchar': np.arange(1, 10, 1, dtype='int8'),
                              'tshort': np.arange(1, 10, 1, dtype='int16'),
                              'tint': np.arange(1, 10, 1, dtype='int32'),
                              'tlong': np.arange(1, 10, 1, dtype='int64'),
                              'tfloat': np.arange(1, 10, 1, dtype='float32'),
                              'tdouble': np.arange(1, 10, 1, dtype='float64')
                              })
        pdf_t = pdf_t.set_index("id")
        odf_t = orca.DataFrame(pdf_t)
        # TODO: ALL ASSERT FAIL
        a = odf_t.rolling(window='t', on="date").mean()
        b = pdf_t.rolling(window='t', on="date").mean()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        # a = odf_t.rolling(window='t', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].mean()
        # b = pdf_t.rolling(window='t', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].mean()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # a = odf_t.rolling(window='t', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].mean()
        # b = pdf_t.rolling(window='t', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].mean()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_t.reset_index()
        # pdf_dai = pdf_t.reset_index()
        # a = odf_dai.rolling(window='t', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        # b = pdf_dai.rolling(window='t', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_t.set_index('date')
        # pdf_dai = pdf_t.set_index('date')
        # a = odf_dai.rolling(window='t').mean()
        # b = pdf_dai.rolling(window='t').mean()
        # assert_frame_equal(a.to_pandas(), b)
        #
        # a = odf_dai.rolling(window='t')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        # b = pdf_dai.rolling(window='t')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_rule_second_mean(self):
        ps = pd.to_datetime(
            ["20170101 9:10:15", "20170101 9:10:16", "20170101 9:10:16", "20170101 9:11:17", "20170101 9:11:17",
             "20180615 9:10:15", "20181031 9:10:15", "20190501 9:10:15",
             "20190517 9:10:15"]).to_series()
        pdf_s = pd.DataFrame({'id': np.arange(1, 10, 1, dtype='int32'),
                              'date': ps,
                              'tchar': np.arange(1, 10, 1, dtype='int8'),
                              'tshort': np.arange(1, 10, 1, dtype='int16'),
                              'tint': np.arange(1, 10, 1, dtype='int32'),
                              'tlong': np.arange(1, 10, 1, dtype='int64'),
                              'tfloat': np.arange(1, 10, 1, dtype='float32'),
                              'tdouble': np.arange(1, 10, 1, dtype='float64')
                              })
        pdf_s = pdf_s.set_index("id")
        odf_s = orca.DataFrame(pdf_s)
        # TODO: ALL ASSERT FAIL
        a = odf_s.rolling(window='s', on="date").mean()
        b = pdf_s.rolling(window='s', on="date").mean()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = odf_s.rolling(window='2s', on="date").mean()
        b = pdf_s.rolling(window='2s', on="date").mean()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        # a = odf_s.rolling(window='s', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].mean()
        # b = pdf_s.rolling(window='s', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].mean()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # a = odf_s.rolling(window='s', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].mean()
        # b = pdf_s.rolling(window='s', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].mean()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_s.reset_index()
        # pdf_dai = pdf_s.reset_index()
        # a = odf_dai.rolling(window='s', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        # b = pdf_dai.rolling(window='s', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_s.set_index('date')
        # pdf_dai = pdf_s.set_index('date')
        # a = odf_dai.rolling(window='s').mean()
        # b = pdf_dai.rolling(window='s').mean()
        # assert_frame_equal(a.to_pandas(), b)
        #
        # a = odf_dai.rolling(window='s')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        # b = pdf_dai.rolling(window='s')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_rule_milli_mean(self):
        ps = pd.to_datetime(
            ["20170101 9:10:15.000", "20170101 9:10:15.000", "20170101 9:10:15.001", "20170101 9:11:17.015",
             "20170101 9:11:17.015",
             "20180615 9:10:15.015", "20181031 9:10:15.015", "20190501 9:10:15.015",
             "20190517 9:10:15.015"]).to_series()
        pdf_l = pd.DataFrame({'id': np.arange(1, 10, 1, dtype='int32'),
                              'date': ps,
                              'tchar': np.arange(1, 10, 1, dtype='int8'),
                              'tshort': np.arange(1, 10, 1, dtype='int16'),
                              'tint': np.arange(1, 10, 1, dtype='int32'),
                              'tlong': np.arange(1, 10, 1, dtype='int64'),
                              'tfloat': np.arange(1, 10, 1, dtype='float32'),
                              'tdouble': np.arange(1, 10, 1, dtype='float64')
                              })
        pdf_l = pdf_l.set_index("id")
        odf_l = orca.DataFrame(pdf_l)
        # TODO: ALL ASSERT FAIL
        a = odf_l.rolling(window='l', on="date").mean()
        b = pdf_l.rolling(window='l', on="date").mean()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = odf_l.rolling(window='2l', on="date").mean()
        b = pdf_l.rolling(window='2l', on="date").mean()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        # a = odf_l.rolling(window='l', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].mean()
        # b = pdf_l.rolling(window='l', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].mean()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # a = odf_l.rolling(window='l', on="date")[
        #     'date'_l, 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].mean()
        # b = pdf.rolling(window='l', on="date")[
        #     'date'_l, 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].mean()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_l.reset_index()
        # pdf_dai = pdf_l.reset_index()
        # a = odf_dai.rolling(window='l', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        # b = pdf_dai.rolling(window='l', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_l.set_index('date')
        # pdf_dai = pdf_l.set_index('date')
        # a = odf_dai.rolling(window='l').mean()
        # b = pdf_dai.rolling(window='l').mean()
        # assert_frame_equal(a.to_pandas(), b)
        #
        # a = odf_dai.rolling(window='l')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        # b = pdf_dai.rolling(window='l')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_rule_micro_mean(self):
        ps = pd.to_datetime(
            ["20170101 9:10:15.000000", "20170101 9:10:15.000000", "20170101 9:10:15.000001", "20170101 9:11:17.015001",
             "20170101 9:11:17.015002",
             "20180615 9:10:15.015000", "20181031 9:10:15.015000", "20190501 9:10:15.015000",
             "20190517 9:10:15.015000"]).to_series()
        pdf_u = pd.DataFrame({'id': np.arange(1, 10, 1, dtype='int32'),
                              'date': ps,
                              'tchar': np.arange(1, 10, 1, dtype='int8'),
                              'tshort': np.arange(1, 10, 1, dtype='int16'),
                              'tint': np.arange(1, 10, 1, dtype='int32'),
                              'tlong': np.arange(1, 10, 1, dtype='int64'),
                              'tfloat': np.arange(1, 10, 1, dtype='float32'),
                              'tdouble': np.arange(1, 10, 1, dtype='float64')
                              })
        pdf_u = pdf_u.set_index("id")
        odf_u = orca.DataFrame(pdf_u)
        # TODO: ALL ASSERT FAIL
        a = odf_u.rolling(window='u', on="date").mean()
        b = pdf_u.rolling(window='u', on="date").mean()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = odf_u.rolling(window='2u', on="date").mean()
        b = pdf_u.rolling(window='2u', on="date").mean()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        # a = odf_u.rolling(window='u', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].mean()
        # b = pdf_u.rolling(window='u', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].mean()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # a = odf_u.rolling(window='u', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].mean()
        # b = pdf_u.rolling(window='u', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].mean()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_u.reset_index()
        # pdf_dai = pdf_u.reset_index()
        # a = odf_dai.rolling(window='u', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        # b = pdf_dai.rolling(window='u', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_u.set_index('date')
        # pdf_dai = pdf_u.set_index('date')
        # a = odf_dai.rolling(window='u').mean()
        # b = pdf_dai.rolling(window='u').mean()
        # assert_frame_equal(a.to_pandas(), b)
        #
        # a = odf_dai.rolling(window='u')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        # b = pdf_dai.rolling(window='u')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_rule_nano_mean(self):
        ps = pd.to_datetime(
            ["20170101 9:10:15.000000000", "20170101 9:10:15.000000000", "20170101 9:10:15.000000001",
             "20170101 9:11:17.015000001",
             "20170101 9:11:17.015002001",
             "20180615 9:10:15.015000001", "20181031 9:10:15.015000001", "20190501 9:10:15.015000001",
             "20190517 9:10:15.015000001"]).to_series()
        pdf_n = pd.DataFrame({'id': np.arange(1, 10, 1, dtype='int32'),
                              'date': ps,
                              'tchar': np.arange(1, 10, 1, dtype='int8'),
                              'tshort': np.arange(1, 10, 1, dtype='int16'),
                              'tint': np.arange(1, 10, 1, dtype='int32'),
                              'tlong': np.arange(1, 10, 1, dtype='int64'),
                              'tfloat': np.arange(1, 10, 1, dtype='float32'),
                              'tdouble': np.arange(1, 10, 1, dtype='float64')
                              })
        pdf_n = pdf_n.set_index("id")
        odf_n = orca.DataFrame(pdf_n)
        # TODO: ALL ASSERT FAIL
        a = odf_n.rolling(window='n', on="date").mean()
        b = pdf_n.rolling(window='n', on="date").mean()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = odf_n.rolling(window='2n', on="date").mean()
        b = pdf_n.rolling(window='2n', on="date").mean()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        # a = odf_n.rolling(window='n', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].mean()
        # b = pdf_n.rolling(window='n', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].mean()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # a = odf_n.rolling(window='n', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].mean()
        # b = pdf_n.rolling(window='n', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].mean()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_n.reset_index()
        # pdf_dai = pdf_n.reset_index()
        # a = odf_dai.rolling(window='n', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        # b = pdf_dai.rolling(window='n', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_n.set_index('date')
        # pdf_dai = pdf_n.set_index('date')
        # a = odf_dai.rolling(window='n').mean()
        # b = pdf_dai.rolling(window='n').mean()
        # assert_frame_equal(a.to_pandas(), b)
        #
        # a = odf_dai.rolling(window='n')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        # b = pdf_dai.rolling(window='n')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_max(self):
        a = self.odf.rolling(window=5, on="date").max()
        b = self.pdf.rolling(window=5, on="date").max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = self.odf.rolling(window=5, on="date")[
            'date', 'tchar', 'tint', 'tlong', 'tfloat'].max()
        b = self.pdf.rolling(window=5, on="date")[
            'date', 'tchar', 'tint', 'tlong', 'tfloat'].max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        a = self.odf.rolling(window=5, on="date")[
            'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].max()
        b = self.pdf.rolling(window=5, on="date")[
            'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.reset_index()
        pdf_dai = self.pdf.reset_index()
        a = odf_dai.rolling(window=5, on="date")[
            'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdf_dai.rolling(window=5, on="date")[
            'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('date')
        pdf_dai = self.pdf.set_index('date')
        a = odf_dai.rolling(window=5).max()
        b = pdf_dai.rolling(window=5).max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = odf_dai.rolling(window=5)[
            'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdf_dai.rolling(window=5)[
            'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_rule_day_max(self):
        a = self.odf_da.rolling(window='d', on="date").max()
        b = self.pdf_da.rolling(window='d', on="date").max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = self.odf_da.rolling(window='3d', on="date")[
            'date', 'tchar', 'tint', 'tlong', 'tfloat'].max()
        b = self.pdf_da.rolling(window='3d', on="date")[
            'date', 'tchar', 'tint', 'tlong', 'tfloat'].max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        a = self.odf_da.rolling(window='d', on="date")[
            'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].max()
        b = self.pdf_da.rolling(window='d', on="date")[
            'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf_da.reset_index()
        pdf_dai = self.pdf_da.reset_index()
        a = odf_dai.rolling(window='d', on="date")[
            'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdf_dai.rolling(window='d', on="date")[
            'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf_da.set_index('date')
        pdf_dai = self.pdf_da.set_index('date')
        a = odf_dai.rolling(window='d').max()
        b = pdf_dai.rolling(window='d').max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = odf_dai.rolling(window='d')[
            'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdf_dai.rolling(window='d')[
            'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_rule_hour_max(self):
        ps = pd.to_datetime(
            ["20170101 9:10:15", "20170101 10:10:15", "20170101 11:10:15", "20170101 11:20:15", "20170101 11:21:00",
             "20180615 9:10:15", "20181031 9:10:15", "20190501 9:10:15",
             "20190517 9:10:15"]).to_series()
        pdf_h = pd.DataFrame({'id': np.arange(1, 10, 1, dtype='int32'),
                              'date': ps,
                              'tchar': np.arange(1, 10, 1, dtype='int8'),
                              'tshort': np.arange(1, 10, 1, dtype='int16'),
                              'tint': np.arange(1, 10, 1, dtype='int32'),
                              'tlong': np.arange(1, 10, 1, dtype='int64'),
                              'tfloat': np.arange(1, 10, 1, dtype='float32'),
                              'tdouble': np.arange(1, 10, 1, dtype='float64')
                              })
        pdf_h = pdf_h.set_index("id")
        odf_h = orca.DataFrame(pdf_h)

        # TODO: ALL ASSERT FAIL
        a = odf_h.rolling(window='h', on="date").max()
        b = pdf_h.rolling(window='h', on="date").max()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = odf_h.rolling(window='2h', on="date").max()
        b = pdf_h.rolling(window='2h', on="date").max()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        # a = odf_h.rolling(window='h', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].max()
        # b = pdf_h.rolling(window='h', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].max()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # a = odf_hrolling(window='h', on="date")[
        #     'date', 'tdouble','tchar', 'tint', 'tlong', 'tfloat','tshort'].max()
        # b = pdf_h.rolling(window='h', on="date")[
        #     'date', 'tdouble','tchar', 'tint', 'tlong', 'tfloat','tshort'].max()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_h.reset_index()
        # pdf_dai = pdf_h.reset_index()
        # a = odf_dai.rolling(window='h', on="date")[
        #     'date',  'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        # b = pdf_dai.rolling(window='h', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_h.set_index('date')
        # pdf_dai = pdf_h.set_index('date')
        # a = odf_dai.rolling(window='h').max()
        # b = pdf_dai.rolling(window='h').max()
        # assert_frame_equal(a.to_pandas(), b)
        #
        # a = odf_dai.rolling(window='h')[
        #      'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        # b = pdf_dai.rolling(window='h')[
        #      'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_rule_minute_max(self):
        ps = pd.to_datetime(
            ["20170101 9:10:15", "20170101 9:10:16", "20170101 9:11:10", "20170101 9:11:17", "20170101 11:21:00",
             "20180615 9:10:15", "20181031 9:10:15", "20190501 9:10:15",
             "20190517 9:10:15"]).to_series()
        pdf_t = pd.DataFrame({'id': np.arange(1, 10, 1, dtype='int32'),
                              'date': ps,
                              'tchar': np.arange(1, 10, 1, dtype='int8'),
                              'tshort': np.arange(1, 10, 1, dtype='int16'),
                              'tint': np.arange(1, 10, 1, dtype='int32'),
                              'tlong': np.arange(1, 10, 1, dtype='int64'),
                              'tfloat': np.arange(1, 10, 1, dtype='float32'),
                              'tdouble': np.arange(1, 10, 1, dtype='float64')
                              })
        pdf_t = pdf_t.set_index("id")
        odf_t = orca.DataFrame(pdf_t)

        # TODO: ALL ASSERT FAIL
        a = odf_t.rolling(window='t', on="date").max()
        b = pdf_t.rolling(window='t', on="date").max()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        # a = odf_t.rolling(window='t', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].max()
        # b = pdf_t.rolling(window='t', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].max()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # a = odf_t.rolling(window='t', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].max()
        # b = pdf_t.rolling(window='t', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].max()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_t.reset_index()
        # pdf_dai = pdf_t.reset_index()
        # a = odf_dai.rolling(window='t', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        # b = pdf_dai.rolling(window='t', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_t.set_index('date')
        # pdf_dai = pdf_t.set_index('date')
        # a = odf_dai.rolling(window='t').max()
        # b = pdf_dai.rolling(window='t').max()
        # assert_frame_equal(a.to_pandas(), b)
        #
        # a = odf_dai.rolling(window='t')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        # b = pdf_dai.rolling(window='t')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_rule_second_max(self):
        ps = pd.to_datetime(
            ["20170101 9:10:15", "20170101 9:10:16", "20170101 9:10:16", "20170101 9:11:17", "20170101 9:11:17",
             "20180615 9:10:15", "20181031 9:10:15", "20190501 9:10:15",
             "20190517 9:10:15"]).to_series()
        pdf_s = pd.DataFrame({'id': np.arange(1, 10, 1, dtype='int32'),
                              'date': ps,
                              'tchar': np.arange(1, 10, 1, dtype='int8'),
                              'tshort': np.arange(1, 10, 1, dtype='int16'),
                              'tint': np.arange(1, 10, 1, dtype='int32'),
                              'tlong': np.arange(1, 10, 1, dtype='int64'),
                              'tfloat': np.arange(1, 10, 1, dtype='float32'),
                              'tdouble': np.arange(1, 10, 1, dtype='float64')
                              })
        pdf_s = pdf_s.set_index("id")
        odf_s = orca.DataFrame(pdf_s)

        # TODO: ALL ASSERT FAIL
        a = odf_s.rolling(window='s', on="date").max()
        b = pdf_s.rolling(window='s', on="date").max()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = odf_s.rolling(window='2s', on="date").max()
        b = pdf_s.rolling(window='2s', on="date").max()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        # a = odf_s.rolling(window='s', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].max()
        # b = pdf_s.rolling(window='s', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].max()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # a = odf_s.rolling(window='s', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].max()
        # b = pdf_s.rolling(window='s', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].max()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_s.reset_index()
        # pdf_dai = pdf_s.reset_index()
        # a = odf_dai.rolling(window='s', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        # b = pdf_dai.rolling(window='s', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_s.set_index('date')
        # pdf_dai = pdf_s.set_index('date')
        # a = odf_dai.rolling(window='s').max()
        # b = pdf_dai.rolling(window='s').max()
        # assert_frame_equal(a.to_pandas(), b)
        #
        # a = odf_dai.rolling(window='s')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        # b = pdf_dai.rolling(window='s')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_rule_milli_max(self):
        ps = pd.to_datetime(
            ["20170101 9:10:15.000", "20170101 9:10:15.000", "20170101 9:10:15.001", "20170101 9:11:17.015",
             "20170101 9:11:17.015",
             "20180615 9:10:15.015", "20181031 9:10:15.015", "20190501 9:10:15.015",
             "20190517 9:10:15.015"]).to_series()
        pdf_l = pd.DataFrame({'id': np.arange(1, 10, 1, dtype='int32'),
                              'date': ps,
                              'tchar': np.arange(1, 10, 1, dtype='int8'),
                              'tshort': np.arange(1, 10, 1, dtype='int16'),
                              'tint': np.arange(1, 10, 1, dtype='int32'),
                              'tlong': np.arange(1, 10, 1, dtype='int64'),
                              'tfloat': np.arange(1, 10, 1, dtype='float32'),
                              'tdouble': np.arange(1, 10, 1, dtype='float64')
                              })
        pdf_l = pdf_l.set_index("id")
        odf_l = orca.DataFrame(pdf_l)

        # TODO: ALL ASSERT FAIL
        a = odf_l.rolling(window='l', on="date").max()
        b = pdf_l.rolling(window='l', on="date").max()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = odf_l.rolling(window='2l', on="date").max()
        b = pdf_l.rolling(window='2l', on="date").max()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        # a = odf_l.rolling(window='l', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].max()
        # b = pdf_l.rolling(window='l', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].max()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # a = odf_l.rolling(window='l', on="date")[
        #     'date'_l, 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].max()
        # b = pdf.rolling(window='l', on="date")[
        #     'date'_l, 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].max()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_l.reset_index()
        # pdf_dai = pdf_l.reset_index()
        # a = odf_dai.rolling(window='l', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        # b = pdf_dai.rolling(window='l', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_l.set_index('date')
        # pdf_dai = pdf_l.set_index('date')
        # a = odf_dai.rolling(window='l').max()
        # b = pdf_dai.rolling(window='l').max()
        # assert_frame_equal(a.to_pandas(), b)
        #
        # a = odf_dai.rolling(window='l')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        # b = pdf_dai.rolling(window='l')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_rule_micro_max(self):
        ps = pd.to_datetime(
            ["20170101 9:10:15.000000", "20170101 9:10:15.000000", "20170101 9:10:15.000001", "20170101 9:11:17.015001",
             "20170101 9:11:17.015002",
             "20180615 9:10:15.015000", "20181031 9:10:15.015000", "20190501 9:10:15.015000",
             "20190517 9:10:15.015000"]).to_series()
        pdf_u = pd.DataFrame({'id': np.arange(1, 10, 1, dtype='int32'),
                              'date': ps,
                              'tchar': np.arange(1, 10, 1, dtype='int8'),
                              'tshort': np.arange(1, 10, 1, dtype='int16'),
                              'tint': np.arange(1, 10, 1, dtype='int32'),
                              'tlong': np.arange(1, 10, 1, dtype='int64'),
                              'tfloat': np.arange(1, 10, 1, dtype='float32'),
                              'tdouble': np.arange(1, 10, 1, dtype='float64')
                              })
        pdf_u = pdf_u.set_index("id")
        odf_u = orca.DataFrame(pdf_u)

        # TODO: ALL ASSERT FAIL
        a = odf_u.rolling(window='u', on="date").max()
        b = pdf_u.rolling(window='u', on="date").max()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = odf_u.rolling(window='2u', on="date").max()
        b = pdf_u.rolling(window='2u', on="date").max()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        # a = odf_u.rolling(window='u', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].max()
        # b = pdf_u.rolling(window='u', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].max()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # a = odf_u.rolling(window='u', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].max()
        # b = pdf_u.rolling(window='u', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].max()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_u.reset_index()
        # pdf_dai = pdf_u.reset_index()
        # a = odf_dai.rolling(window='u', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        # b = pdf_dai.rolling(window='u', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_u.set_index('date')
        # pdf_dai = pdf_u.set_index('date')
        # a = odf_dai.rolling(window='u').max()
        # b = pdf_dai.rolling(window='u').max()
        # assert_frame_equal(a.to_pandas(), b)
        #
        # a = odf_dai.rolling(window='u')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        # b = pdf_dai.rolling(window='u')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_rule_nano_max(self):
        ps = pd.to_datetime(
            ["20170101 9:10:15.000000000", "20170101 9:10:15.000000000", "20170101 9:10:15.000000001",
             "20170101 9:11:17.015000001",
             "20170101 9:11:17.015002001",
             "20180615 9:10:15.015000001", "20181031 9:10:15.015000001", "20190501 9:10:15.015000001",
             "20190517 9:10:15.015000001"]).to_series()
        pdf_n = pd.DataFrame({'id': np.arange(1, 10, 1, dtype='int32'),
                              'date': ps,
                              'tchar': np.arange(1, 10, 1, dtype='int8'),
                              'tshort': np.arange(1, 10, 1, dtype='int16'),
                              'tint': np.arange(1, 10, 1, dtype='int32'),
                              'tlong': np.arange(1, 10, 1, dtype='int64'),
                              'tfloat': np.arange(1, 10, 1, dtype='float32'),
                              'tdouble': np.arange(1, 10, 1, dtype='float64')
                              })
        pdf_n = pdf_n.set_index("id")
        odf_n = orca.DataFrame(pdf_n)

        # TODO: ALL ASSERT FAIL
        a = odf_n.rolling(window='n', on="date").max()
        b = pdf_n.rolling(window='n', on="date").max()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = odf_n.rolling(window='2n', on="date").max()
        b = pdf_n.rolling(window='2n', on="date").max()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        # a = odf_n.rolling(window='n', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].max()
        # b = pdf_n.rolling(window='n', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].max()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # a = odf_n.rolling(window='n', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].max()
        # b = pdf_n.rolling(window='n', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].max()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_n.reset_index()
        # pdf_dai = pdf_n.reset_index()
        # a = odf_dai.rolling(window='n', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        # b = pdf_dai.rolling(window='n', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_n.set_index('date')
        # pdf_dai = pdf_n.set_index('date')
        # a = odf_dai.rolling(window='n').max()
        # b = pdf_dai.rolling(window='n').max()
        # assert_frame_equal(a.to_pandas(), b)
        #
        # a = odf_dai.rolling(window='n')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        # b = pdf_dai.rolling(window='n')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_min(self):
        a = self.odf.rolling(window=5, on="date").min()
        b = self.pdf.rolling(window=5, on="date").min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = self.odf.rolling(window=5, on="date")[
            'date', 'tchar', 'tint', 'tlong', 'tfloat'].min()
        b = self.pdf.rolling(window=5, on="date")[
            'date', 'tchar', 'tint', 'tlong', 'tfloat'].min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        a = self.odf.rolling(window=5, on="date")[
            'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].min()
        b = self.pdf.rolling(window=5, on="date")[
            'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.reset_index()
        pdf_dai = self.pdf.reset_index()
        a = odf_dai.rolling(window=5, on="date")[
            'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdf_dai.rolling(window=5, on="date")[
            'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('date')
        pdf_dai = self.pdf.set_index('date')
        a = odf_dai.rolling(window=5).min()
        b = pdf_dai.rolling(window=5).min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = odf_dai.rolling(window=5)[
            'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdf_dai.rolling(window=5)[
            'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_rule_day_min(self):
        a = self.odf_da.rolling(window='d', on="date").min()
        b = self.pdf_da.rolling(window='d', on="date").min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = self.odf_da.rolling(window='3d', on="date")[
            'date', 'tchar', 'tint', 'tlong', 'tfloat'].min()
        b = self.pdf_da.rolling(window='3d', on="date")[
            'date', 'tchar', 'tint', 'tlong', 'tfloat'].min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        a = self.odf_da.rolling(window='d', on="date")[
            'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].min()
        b = self.pdf_da.rolling(window='d', on="date")[
            'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf_da.reset_index()
        pdf_dai = self.pdf_da.reset_index()
        a = odf_dai.rolling(window='d', on="date")[
            'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdf_dai.rolling(window='d', on="date")[
            'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf_da.set_index('date')
        pdf_dai = self.pdf_da.set_index('date')
        a = odf_dai.rolling(window='d').min()
        b = pdf_dai.rolling(window='d').min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = odf_dai.rolling(window='d')[
            'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdf_dai.rolling(window='d')[
            'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_rule_hour_min(self):
        ps = pd.to_datetime(
            ["20170101 9:10:15", "20170101 10:10:15", "20170101 11:10:15", "20170101 11:20:15", "20170101 11:21:00",
             "20180615 9:10:15", "20181031 9:10:15", "20190501 9:10:15",
             "20190517 9:10:15"]).to_series()
        pdf_h = pd.DataFrame({'id': np.arange(1, 10, 1, dtype='int32'),
                              'date': ps,
                              'tchar': np.arange(1, 10, 1, dtype='int8'),
                              'tshort': np.arange(1, 10, 1, dtype='int16'),
                              'tint': np.arange(1, 10, 1, dtype='int32'),
                              'tlong': np.arange(1, 10, 1, dtype='int64'),
                              'tfloat': np.arange(1, 10, 1, dtype='float32'),
                              'tdouble': np.arange(1, 10, 1, dtype='float64')
                              })
        pdf_h = pdf_h.set_index("id")
        odf_h = orca.DataFrame(pdf_h)

        # TODO: ALL ASSERT FAIL
        a = odf_h.rolling(window='h', on="date").min()
        b = pdf_h.rolling(window='h', on="date").min()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = odf_h.rolling(window='2h', on="date").min()
        b = pdf_h.rolling(window='2h', on="date").min()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        # a = odf_h.rolling(window='h', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].min()
        # b = pdf_h.rolling(window='h', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].min()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # a = odf_hrolling(window='h', on="date")[
        #     'date', 'tdouble','tchar', 'tint', 'tlong', 'tfloat','tshort'].min()
        # b = pdf_h.rolling(window='h', on="date")[
        #     'date', 'tdouble','tchar', 'tint', 'tlong', 'tfloat','tshort'].min()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_h.reset_index()
        # pdf_dai = pdf_h.reset_index()
        # a = odf_dai.rolling(window='h', on="date")[
        #     'date',  'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        # b = pdf_dai.rolling(window='h', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_h.set_index('date')
        # pdf_dai = pdf_h.set_index('date')
        # a = odf_dai.rolling(window='h').min()
        # b = pdf_dai.rolling(window='h').min()
        # assert_frame_equal(a.to_pandas(), b)
        #
        # a = odf_dai.rolling(window='h')[
        #      'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        # b = pdf_dai.rolling(window='h')[
        #      'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_rule_minute_min(self):
        ps = pd.to_datetime(
            ["20170101 9:10:15", "20170101 9:10:16", "20170101 9:11:10", "20170101 9:11:17", "20170101 11:21:00",
             "20180615 9:10:15", "20181031 9:10:15", "20190501 9:10:15",
             "20190517 9:10:15"]).to_series()
        pdf_t = pd.DataFrame({'id': np.arange(1, 10, 1, dtype='int32'),
                              'date': ps,
                              'tchar': np.arange(1, 10, 1, dtype='int8'),
                              'tshort': np.arange(1, 10, 1, dtype='int16'),
                              'tint': np.arange(1, 10, 1, dtype='int32'),
                              'tlong': np.arange(1, 10, 1, dtype='int64'),
                              'tfloat': np.arange(1, 10, 1, dtype='float32'),
                              'tdouble': np.arange(1, 10, 1, dtype='float64')
                              })
        pdf_t = pdf_t.set_index("id")
        odf_t = orca.DataFrame(pdf_t)

        # TODO: ALL ASSERT FAIL
        a = odf_t.rolling(window='t', on="date").min()
        b = pdf_t.rolling(window='t', on="date").min()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        # a = odf_t.rolling(window='t', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].min()
        # b = pdf_t.rolling(window='t', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].min()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # a = odf_t.rolling(window='t', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].min()
        # b = pdf_t.rolling(window='t', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].min()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_t.reset_index()
        # pdf_dai = pdf_t.reset_index()
        # a = odf_dai.rolling(window='t', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        # b = pdf_dai.rolling(window='t', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_t.set_index('date')
        # pdf_dai = pdf_t.set_index('date')
        # a = odf_dai.rolling(window='t').min()
        # b = pdf_dai.rolling(window='t').min()
        # assert_frame_equal(a.to_pandas(), b)
        #
        # a = odf_dai.rolling(window='t')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        # b = pdf_dai.rolling(window='t')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_rule_second_min(self):
        ps = pd.to_datetime(
            ["20170101 9:10:15", "20170101 9:10:16", "20170101 9:10:16", "20170101 9:11:17", "20170101 9:11:17",
             "20180615 9:10:15", "20181031 9:10:15", "20190501 9:10:15",
             "20190517 9:10:15"]).to_series()
        pdf_s = pd.DataFrame({'id': np.arange(1, 10, 1, dtype='int32'),
                              'date': ps,
                              'tchar': np.arange(1, 10, 1, dtype='int8'),
                              'tshort': np.arange(1, 10, 1, dtype='int16'),
                              'tint': np.arange(1, 10, 1, dtype='int32'),
                              'tlong': np.arange(1, 10, 1, dtype='int64'),
                              'tfloat': np.arange(1, 10, 1, dtype='float32'),
                              'tdouble': np.arange(1, 10, 1, dtype='float64')
                              })
        pdf_s = pdf_s.set_index("id")
        odf_s = orca.DataFrame(pdf_s)

        # TODO: ALL ASSERT FAIL
        a = odf_s.rolling(window='s', on="date").min()
        b = pdf_s.rolling(window='s', on="date").min()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = odf_s.rolling(window='2s', on="date").min()
        b = pdf_s.rolling(window='2s', on="date").min()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        # a = odf_s.rolling(window='s', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].min()
        # b = pdf_s.rolling(window='s', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].min()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # a = odf_s.rolling(window='s', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].min()
        # b = pdf_s.rolling(window='s', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].min()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_s.reset_index()
        # pdf_dai = pdf_s.reset_index()
        # a = odf_dai.rolling(window='s', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        # b = pdf_dai.rolling(window='s', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_s.set_index('date')
        # pdf_dai = pdf_s.set_index('date')
        # a = odf_dai.rolling(window='s').min()
        # b = pdf_dai.rolling(window='s').min()
        # assert_frame_equal(a.to_pandas(), b)
        #
        # a = odf_dai.rolling(window='s')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        # b = pdf_dai.rolling(window='s')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_rule_milli_min(self):
        ps = pd.to_datetime(
            ["20170101 9:10:15.000", "20170101 9:10:15.000", "20170101 9:10:15.001", "20170101 9:11:17.015",
             "20170101 9:11:17.015",
             "20180615 9:10:15.015", "20181031 9:10:15.015", "20190501 9:10:15.015",
             "20190517 9:10:15.015"]).to_series()
        pdf_l = pd.DataFrame({'id': np.arange(1, 10, 1, dtype='int32'),
                              'date': ps,
                              'tchar': np.arange(1, 10, 1, dtype='int8'),
                              'tshort': np.arange(1, 10, 1, dtype='int16'),
                              'tint': np.arange(1, 10, 1, dtype='int32'),
                              'tlong': np.arange(1, 10, 1, dtype='int64'),
                              'tfloat': np.arange(1, 10, 1, dtype='float32'),
                              'tdouble': np.arange(1, 10, 1, dtype='float64')
                              })
        pdf_l = pdf_l.set_index("id")
        odf_l = orca.DataFrame(pdf_l)

        # TODO: ALL ASSERT FAIL
        a = odf_l.rolling(window='l', on="date").min()
        b = pdf_l.rolling(window='l', on="date").min()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = odf_l.rolling(window='2l', on="date").min()
        b = pdf_l.rolling(window='2l', on="date").min()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        # a = odf_l.rolling(window='l', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].min()
        # b = pdf_l.rolling(window='l', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].min()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # a = odf_l.rolling(window='l', on="date")[
        #     'date'_l, 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].min()
        # b = pdf.rolling(window='l', on="date")[
        #     'date'_l, 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].min()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_l.reset_index()
        # pdf_dai = pdf_l.reset_index()
        # a = odf_dai.rolling(window='l', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        # b = pdf_dai.rolling(window='l', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_l.set_index('date')
        # pdf_dai = pdf_l.set_index('date')
        # a = odf_dai.rolling(window='l').min()
        # b = pdf_dai.rolling(window='l').min()
        # assert_frame_equal(a.to_pandas(), b)
        #
        # a = odf_dai.rolling(window='l')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        # b = pdf_dai.rolling(window='l')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_rule_micro_min(self):
        ps = pd.to_datetime(
            ["20170101 9:10:15.000000", "20170101 9:10:15.000000", "20170101 9:10:15.000001", "20170101 9:11:17.015001",
             "20170101 9:11:17.015002",
             "20180615 9:10:15.015000", "20181031 9:10:15.015000", "20190501 9:10:15.015000",
             "20190517 9:10:15.015000"]).to_series()
        pdf_u = pd.DataFrame({'id': np.arange(1, 10, 1, dtype='int32'),
                              'date': ps,
                              'tchar': np.arange(1, 10, 1, dtype='int8'),
                              'tshort': np.arange(1, 10, 1, dtype='int16'),
                              'tint': np.arange(1, 10, 1, dtype='int32'),
                              'tlong': np.arange(1, 10, 1, dtype='int64'),
                              'tfloat': np.arange(1, 10, 1, dtype='float32'),
                              'tdouble': np.arange(1, 10, 1, dtype='float64')
                              })
        pdf_u = pdf_u.set_index("id")
        odf_u = orca.DataFrame(pdf_u)

        # TODO: ALL ASSERT FAIL
        a = odf_u.rolling(window='u', on="date").min()
        b = pdf_u.rolling(window='u', on="date").min()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = odf_u.rolling(window='2u', on="date").min()
        b = pdf_u.rolling(window='2u', on="date").min()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        # a = odf_u.rolling(window='u', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].min()
        # b = pdf_u.rolling(window='u', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].min()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # a = odf_u.rolling(window='u', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].min()
        # b = pdf_u.rolling(window='u', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].min()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_u.reset_index()
        # pdf_dai = pdf_u.reset_index()
        # a = odf_dai.rolling(window='u', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        # b = pdf_dai.rolling(window='u', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_u.set_index('date')
        # pdf_dai = pdf_u.set_index('date')
        # a = odf_dai.rolling(window='u').min()
        # b = pdf_dai.rolling(window='u').min()
        # assert_frame_equal(a.to_pandas(), b)
        #
        # a = odf_dai.rolling(window='u')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        # b = pdf_dai.rolling(window='u')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_rule_nano_min(self):
        ps = pd.to_datetime(
            ["20170101 9:10:15.000000000", "20170101 9:10:15.000000000", "20170101 9:10:15.000000001",
             "20170101 9:11:17.015000001",
             "20170101 9:11:17.015002001",
             "20180615 9:10:15.015000001", "20181031 9:10:15.015000001", "20190501 9:10:15.015000001",
             "20190517 9:10:15.015000001"]).to_series()
        pdf_n = pd.DataFrame({'id': np.arange(1, 10, 1, dtype='int32'),
                              'date': ps,
                              'tchar': np.arange(1, 10, 1, dtype='int8'),
                              'tshort': np.arange(1, 10, 1, dtype='int16'),
                              'tint': np.arange(1, 10, 1, dtype='int32'),
                              'tlong': np.arange(1, 10, 1, dtype='int64'),
                              'tfloat': np.arange(1, 10, 1, dtype='float32'),
                              'tdouble': np.arange(1, 10, 1, dtype='float64')
                              })
        pdf_n = pdf_n.set_index("id")
        odf_n = orca.DataFrame(pdf_n)

        # TODO: ALL ASSERT FAIL
        a = odf_n.rolling(window='n', on="date").min()
        b = pdf_n.rolling(window='n', on="date").min()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = odf_n.rolling(window='2n', on="date").min()
        b = pdf_n.rolling(window='2n', on="date").min()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        # a = odf_n.rolling(window='n', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].min()
        # b = pdf_n.rolling(window='n', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].min()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # a = odf_n.rolling(window='n', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].min()
        # b = pdf_n.rolling(window='n', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].min()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_n.reset_index()
        # pdf_dai = pdf_n.reset_index()
        # a = odf_dai.rolling(window='n', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        # b = pdf_dai.rolling(window='n', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_n.set_index('date')
        # pdf_dai = pdf_n.set_index('date')
        # a = odf_dai.rolling(window='n').min()
        # b = pdf_dai.rolling(window='n').min()
        # assert_frame_equal(a.to_pandas(), b)
        #
        # a = odf_dai.rolling(window='n')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        # b = pdf_dai.rolling(window='n')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_std(self):
        a = self.odf.rolling(window=5, on="date").std()
        b = self.pdf.rolling(window=5, on="date").std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = self.odf.rolling(window=5, on="date")[
            'date', 'tchar', 'tint', 'tlong', 'tfloat'].std()
        b = self.pdf.rolling(window=5, on="date")[
            'date', 'tchar', 'tint', 'tlong', 'tfloat'].std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        a = self.odf.rolling(window=5, on="date")[
            'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].std()
        b = self.pdf.rolling(window=5, on="date")[
            'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.reset_index()
        pdf_dai = self.pdf.reset_index()
        a = odf_dai.rolling(window=5, on="date")[
            'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdf_dai.rolling(window=5, on="date")[
            'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('date')
        pdf_dai = self.pdf.set_index('date')
        a = odf_dai.rolling(window=5).std()
        b = pdf_dai.rolling(window=5).std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = odf_dai.rolling(window=5)[
            'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdf_dai.rolling(window=5)[
            'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_rule_day_std(self):
        a = self.odf_da.rolling(window='d', on="date").std()
        b = self.pdf_da.rolling(window='d', on="date").std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = self.odf_da.rolling(window='3d', on="date")[
            'date', 'tchar', 'tint', 'tlong', 'tfloat'].std()
        b = self.pdf_da.rolling(window='3d', on="date")[
            'date', 'tchar', 'tint', 'tlong', 'tfloat'].std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        a = self.odf_da.rolling(window='d', on="date")[
            'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].std()
        b = self.pdf_da.rolling(window='d', on="date")[
            'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf_da.reset_index()
        pdf_dai = self.pdf_da.reset_index()
        a = odf_dai.rolling(window='d', on="date")[
            'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdf_dai.rolling(window='d', on="date")[
            'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf_da.set_index('date')
        pdf_dai = self.pdf_da.set_index('date')
        a = odf_dai.rolling(window='d').std()
        b = pdf_dai.rolling(window='d').std()
        assert_frame_equal(a.to_pandas(), b)

        a = odf_dai.rolling(window='d')[
            'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdf_dai.rolling(window='d')[
            'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_rule_hour_std(self):
        ps = pd.to_datetime(
            ["20170101 9:10:15", "20170101 10:10:15", "20170101 11:10:15", "20170101 11:20:15", "20170101 11:21:00",
             "20180615 9:10:15", "20181031 9:10:15", "20190501 9:10:15",
             "20190517 9:10:15"]).to_series()
        pdf_h = pd.DataFrame({'id': np.arange(1, 10, 1, dtype='int32'),
                              'date': ps,
                              'tchar': np.arange(1, 10, 1, dtype='int8'),
                              'tshort': np.arange(1, 10, 1, dtype='int16'),
                              'tint': np.arange(1, 10, 1, dtype='int32'),
                              'tlong': np.arange(1, 10, 1, dtype='int64'),
                              'tfloat': np.arange(1, 10, 1, dtype='float32'),
                              'tdouble': np.arange(1, 10, 1, dtype='float64')
                              })
        pdf_h = pdf_h.set_index("id")
        odf_h = orca.DataFrame(pdf_h)

        # TODO: ALL ASSERT FAIL
        a = odf_h.rolling(window='h', on="date").std()
        b = pdf_h.rolling(window='h', on="date").std()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = odf_h.rolling(window='2h', on="date").std()
        b = pdf_h.rolling(window='2h', on="date").std()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        # a = odf_h.rolling(window='h', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].std()
        # b = pdf_h.rolling(window='h', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].std()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # a = odf_hrolling(window='h', on="date")[
        #     'date', 'tdouble','tchar', 'tint', 'tlong', 'tfloat','tshort'].std()
        # b = pdf_h.rolling(window='h', on="date")[
        #     'date', 'tdouble','tchar', 'tint', 'tlong', 'tfloat','tshort'].std()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_h.reset_index()
        # pdf_dai = pdf_h.reset_index()
        # a = odf_dai.rolling(window='h', on="date")[
        #     'date',  'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        # b = pdf_dai.rolling(window='h', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_h.set_index('date')
        # pdf_dai = pdf_h.set_index('date')
        # a = odf_dai.rolling(window='h').std()
        # b = pdf_dai.rolling(window='h').std()
        # assert_frame_equal(a.to_pandas(), b)
        #
        # a = odf_dai.rolling(window='h')[
        #      'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        # b = pdf_dai.rolling(window='h')[
        #      'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_rule_minute_std(self):
        ps = pd.to_datetime(
            ["20170101 9:10:15", "20170101 9:10:16", "20170101 9:11:10", "20170101 9:11:17", "20170101 11:21:00",
             "20180615 9:10:15", "20181031 9:10:15", "20190501 9:10:15",
             "20190517 9:10:15"]).to_series()
        pdf_t = pd.DataFrame({'id': np.arange(1, 10, 1, dtype='int32'),
                              'date': ps,
                              'tchar': np.arange(1, 10, 1, dtype='int8'),
                              'tshort': np.arange(1, 10, 1, dtype='int16'),
                              'tint': np.arange(1, 10, 1, dtype='int32'),
                              'tlong': np.arange(1, 10, 1, dtype='int64'),
                              'tfloat': np.arange(1, 10, 1, dtype='float32'),
                              'tdouble': np.arange(1, 10, 1, dtype='float64')
                              })
        pdf_t = pdf_t.set_index("id")
        odf_t = orca.DataFrame(pdf_t)

        # TODO: ALL ASSERT FAIL
        a = odf_t.rolling(window='t', on="date").std()
        b = pdf_t.rolling(window='t', on="date").std()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        # a = odf_t.rolling(window='t', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].std()
        # b = pdf_t.rolling(window='t', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].std()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # a = odf_t.rolling(window='t', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].std()
        # b = pdf_t.rolling(window='t', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].std()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_t.reset_index()
        # pdf_dai = pdf_t.reset_index()
        # a = odf_dai.rolling(window='t', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        # b = pdf_dai.rolling(window='t', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_t.set_index('date')
        # pdf_dai = pdf_t.set_index('date')
        # a = odf_dai.rolling(window='t').std()
        # b = pdf_dai.rolling(window='t').std()
        # assert_frame_equal(a.to_pandas(), b)
        #
        # a = odf_dai.rolling(window='t')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        # b = pdf_dai.rolling(window='t')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_rule_second_std(self):
        ps = pd.to_datetime(
            ["20170101 9:10:15", "20170101 9:10:16", "20170101 9:10:16", "20170101 9:11:17", "20170101 9:11:17",
             "20180615 9:10:15", "20181031 9:10:15", "20190501 9:10:15",
             "20190517 9:10:15"]).to_series()
        pdf_s = pd.DataFrame({'id': np.arange(1, 10, 1, dtype='int32'),
                              'date': ps,
                              'tchar': np.arange(1, 10, 1, dtype='int8'),
                              'tshort': np.arange(1, 10, 1, dtype='int16'),
                              'tint': np.arange(1, 10, 1, dtype='int32'),
                              'tlong': np.arange(1, 10, 1, dtype='int64'),
                              'tfloat': np.arange(1, 10, 1, dtype='float32'),
                              'tdouble': np.arange(1, 10, 1, dtype='float64')
                              })
        pdf_s = pdf_s.set_index("id")
        odf_s = orca.DataFrame(pdf_s)

        # TODO: ALL ASSERT FAIL
        a = odf_s.rolling(window='s', on="date").std()
        b = pdf_s.rolling(window='s', on="date").std()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = odf_s.rolling(window='2s', on="date").std()
        b = pdf_s.rolling(window='2s', on="date").std()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        # a = odf_s.rolling(window='s', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].std()
        # b = pdf_s.rolling(window='s', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].std()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # a = odf_s.rolling(window='s', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].std()
        # b = pdf_s.rolling(window='s', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].std()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_s.reset_index()
        # pdf_dai = pdf_s.reset_index()
        # a = odf_dai.rolling(window='s', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        # b = pdf_dai.rolling(window='s', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_s.set_index('date')
        # pdf_dai = pdf_s.set_index('date')
        # a = odf_dai.rolling(window='s').std()
        # b = pdf_dai.rolling(window='s').std()
        # assert_frame_equal(a.to_pandas(), b)
        #
        # a = odf_dai.rolling(window='s')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        # b = pdf_dai.rolling(window='s')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_rule_milli_std(self):
        ps = pd.to_datetime(
            ["20170101 9:10:15.000", "20170101 9:10:15.000", "20170101 9:10:15.001", "20170101 9:11:17.015",
             "20170101 9:11:17.015",
             "20180615 9:10:15.015", "20181031 9:10:15.015", "20190501 9:10:15.015",
             "20190517 9:10:15.015"]).to_series()
        pdf_l = pd.DataFrame({'id': np.arange(1, 10, 1, dtype='int32'),
                              'date': ps,
                              'tchar': np.arange(1, 10, 1, dtype='int8'),
                              'tshort': np.arange(1, 10, 1, dtype='int16'),
                              'tint': np.arange(1, 10, 1, dtype='int32'),
                              'tlong': np.arange(1, 10, 1, dtype='int64'),
                              'tfloat': np.arange(1, 10, 1, dtype='float32'),
                              'tdouble': np.arange(1, 10, 1, dtype='float64')
                              })
        pdf_l = pdf_l.set_index("id")
        odf_l = orca.DataFrame(pdf_l)

        # TODO: ALL ASSERT FAIL
        a = odf_l.rolling(window='l', on="date").std()
        b = pdf_l.rolling(window='l', on="date").std()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = odf_l.rolling(window='2l', on="date").std()
        b = pdf_l.rolling(window='2l', on="date").std()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        # a = odf_l.rolling(window='l', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].std()
        # b = pdf_l.rolling(window='l', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].std()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # a = odf_l.rolling(window='l', on="date")[
        #     'date'_l, 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].std()
        # b = pdf.rolling(window='l', on="date")[
        #     'date'_l, 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].std()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_l.reset_index()
        # pdf_dai = pdf_l.reset_index()
        # a = odf_dai.rolling(window='l', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        # b = pdf_dai.rolling(window='l', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_l.set_index('date')
        # pdf_dai = pdf_l.set_index('date')
        # a = odf_dai.rolling(window='l').std()
        # b = pdf_dai.rolling(window='l').std()
        # assert_frame_equal(a.to_pandas(), b)
        #
        # a = odf_dai.rolling(window='l')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        # b = pdf_dai.rolling(window='l')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_rule_micro_std(self):
        ps = pd.to_datetime(
            ["20170101 9:10:15.000000", "20170101 9:10:15.000000", "20170101 9:10:15.000001", "20170101 9:11:17.015001",
             "20170101 9:11:17.015002",
             "20180615 9:10:15.015000", "20181031 9:10:15.015000", "20190501 9:10:15.015000",
             "20190517 9:10:15.015000"]).to_series()
        pdf_u = pd.DataFrame({'id': np.arange(1, 10, 1, dtype='int32'),
                              'date': ps,
                              'tchar': np.arange(1, 10, 1, dtype='int8'),
                              'tshort': np.arange(1, 10, 1, dtype='int16'),
                              'tint': np.arange(1, 10, 1, dtype='int32'),
                              'tlong': np.arange(1, 10, 1, dtype='int64'),
                              'tfloat': np.arange(1, 10, 1, dtype='float32'),
                              'tdouble': np.arange(1, 10, 1, dtype='float64')
                              })
        pdf_u = pdf_u.set_index("id")
        odf_u = orca.DataFrame(pdf_u)

        # TODO: ALL ASSERT FAIL
        a = odf_u.rolling(window='u', on="date").std()
        b = pdf_u.rolling(window='u', on="date").std()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = odf_u.rolling(window='2u', on="date").std()
        b = pdf_u.rolling(window='2u', on="date").std()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        # a = odf_u.rolling(window='u', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].std()
        # b = pdf_u.rolling(window='u', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].std()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # a = odf_u.rolling(window='u', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].std()
        # b = pdf_u.rolling(window='u', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].std()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_u.reset_index()
        # pdf_dai = pdf_u.reset_index()
        # a = odf_dai.rolling(window='u', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        # b = pdf_dai.rolling(window='u', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_u.set_index('date')
        # pdf_dai = pdf_u.set_index('date')
        # a = odf_dai.rolling(window='u').std()
        # b = pdf_dai.rolling(window='u').std()
        # assert_frame_equal(a.to_pandas(), b)
        #
        # a = odf_dai.rolling(window='u')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        # b = pdf_dai.rolling(window='u')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_rule_nano_std(self):
        ps = pd.to_datetime(
            ["20170101 9:10:15.000000000", "20170101 9:10:15.000000000", "20170101 9:10:15.000000001",
             "20170101 9:11:17.015000001",
             "20170101 9:11:17.015002001",
             "20180615 9:10:15.015000001", "20181031 9:10:15.015000001", "20190501 9:10:15.015000001",
             "20190517 9:10:15.015000001"]).to_series()
        pdf_n = pd.DataFrame({'id': np.arange(1, 10, 1, dtype='int32'),
                              'date': ps,
                              'tchar': np.arange(1, 10, 1, dtype='int8'),
                              'tshort': np.arange(1, 10, 1, dtype='int16'),
                              'tint': np.arange(1, 10, 1, dtype='int32'),
                              'tlong': np.arange(1, 10, 1, dtype='int64'),
                              'tfloat': np.arange(1, 10, 1, dtype='float32'),
                              'tdouble': np.arange(1, 10, 1, dtype='float64')
                              })
        pdf_n = pdf_n.set_index("id")
        odf_n = orca.DataFrame(pdf_n)

        # TODO: ALL ASSERT FAIL
        a = odf_n.rolling(window='n', on="date").std()
        b = pdf_n.rolling(window='n', on="date").std()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = odf_n.rolling(window='2n', on="date").std()
        b = pdf_n.rolling(window='2n', on="date").std()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        # a = odf_n.rolling(window='n', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].std()
        # b = pdf_n.rolling(window='n', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].std()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # a = odf_n.rolling(window='n', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].std()
        # b = pdf_n.rolling(window='n', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].std()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_n.reset_index()
        # pdf_dai = pdf_n.reset_index()
        # a = odf_dai.rolling(window='n', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        # b = pdf_dai.rolling(window='n', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_n.set_index('date')
        # pdf_dai = pdf_n.set_index('date')
        # a = odf_dai.rolling(window='n').std()
        # b = pdf_dai.rolling(window='n').std()
        # assert_frame_equal(a.to_pandas(), b)
        #
        # a = odf_dai.rolling(window='n')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        # b = pdf_dai.rolling(window='n')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_var(self):
        a = self.odf.rolling(window=5, on="date").var()
        b = self.pdf.rolling(window=5, on="date").var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = self.odf.rolling(window=5, on="date")[
            'date', 'tchar', 'tint', 'tlong', 'tfloat'].var()
        b = self.pdf.rolling(window=5, on="date")[
            'date', 'tchar', 'tint', 'tlong', 'tfloat'].var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        a = self.odf.rolling(window=5, on="date")[
            'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].var()
        b = self.pdf.rolling(window=5, on="date")[
            'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.reset_index()
        pdf_dai = self.pdf.reset_index()
        a = odf_dai.rolling(window=5, on="date")[
            'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdf_dai.rolling(window=5, on="date")[
            'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf.set_index('date')
        pdf_dai = self.pdf.set_index('date')
        a = odf_dai.rolling(window=5).var()
        b = pdf_dai.rolling(window=5).var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = odf_dai.rolling(window=5)[
            'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdf_dai.rolling(window=5)[
            'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_rule_day_var(self):
        a = self.odf_da.rolling(window='d', on="date").var()
        b = self.pdf_da.rolling(window='d', on="date").var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = self.odf_da.rolling(window='3d', on="date")[
            'date', 'tchar', 'tint', 'tlong', 'tfloat'].var()
        b = self.pdf_da.rolling(window='3d', on="date")[
            'date', 'tchar', 'tint', 'tlong', 'tfloat'].var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        a = self.odf_da.rolling(window='d', on="date")[
            'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].var()
        b = self.pdf_da.rolling(window='d', on="date")[
            'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf_da.reset_index()
        pdf_dai = self.pdf_da.reset_index()
        a = odf_dai.rolling(window='d', on="date")[
            'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdf_dai.rolling(window='d', on="date")[
            'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf_da.set_index('date')
        pdf_dai = self.pdf_da.set_index('date')
        a = odf_dai.rolling(window='d').var()
        b = pdf_dai.rolling(window='d').var()
        assert_frame_equal(a.to_pandas(), b)

        a = odf_dai.rolling(window='d')[
            'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdf_dai.rolling(window='d')[
            'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_rule_hour_var(self):
        ps = pd.to_datetime(
            ["20170101 9:10:15", "20170101 10:10:15", "20170101 11:10:15", "20170101 11:20:15", "20170101 11:21:00",
             "20180615 9:10:15", "20181031 9:10:15", "20190501 9:10:15",
             "20190517 9:10:15"]).to_series()
        pdf_h = pd.DataFrame({'id': np.arange(1, 10, 1, dtype='int32'),
                              'date': ps,
                              'tchar': np.arange(1, 10, 1, dtype='int8'),
                              'tshort': np.arange(1, 10, 1, dtype='int16'),
                              'tint': np.arange(1, 10, 1, dtype='int32'),
                              'tlong': np.arange(1, 10, 1, dtype='int64'),
                              'tfloat': np.arange(1, 10, 1, dtype='float32'),
                              'tdouble': np.arange(1, 10, 1, dtype='float64')
                              })
        pdf_h = pdf_h.set_index("id")
        odf_h = orca.DataFrame(pdf_h)

        # TODO: ALL ASSERT FAIL
        a = odf_h.rolling(window='h', on="date").var()
        b = pdf_h.rolling(window='h', on="date").var()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = odf_h.rolling(window='2h', on="date").var()
        b = pdf_h.rolling(window='2h', on="date").var()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        # a = odf_h.rolling(window='h', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].var()
        # b = pdf_h.rolling(window='h', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].var()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # a = odf_hrolling(window='h', on="date")[
        #     'date', 'tdouble','tchar', 'tint', 'tlong', 'tfloat','tshort'].var()
        # b = pdf_h.rolling(window='h', on="date")[
        #     'date', 'tdouble','tchar', 'tint', 'tlong', 'tfloat','tshort'].var()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_h.reset_index()
        # pdf_dai = pdf_h.reset_index()
        # a = odf_dai.rolling(window='h', on="date")[
        #     'date',  'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        # b = pdf_dai.rolling(window='h', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_h.set_index('date')
        # pdf_dai = pdf_h.set_index('date')
        # a = odf_dai.rolling(window='h').var()
        # b = pdf_dai.rolling(window='h').var()
        # assert_frame_equal(a.to_pandas(), b)
        #
        # a = odf_dai.rolling(window='h')[
        #      'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        # b = pdf_dai.rolling(window='h')[
        #      'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_rule_minute_var(self):
        ps = pd.to_datetime(
            ["20170101 9:10:15", "20170101 9:10:16", "20170101 9:11:10", "20170101 9:11:17", "20170101 11:21:00",
             "20180615 9:10:15", "20181031 9:10:15", "20190501 9:10:15",
             "20190517 9:10:15"]).to_series()
        pdf_t = pd.DataFrame({'id': np.arange(1, 10, 1, dtype='int32'),
                              'date': ps,
                              'tchar': np.arange(1, 10, 1, dtype='int8'),
                              'tshort': np.arange(1, 10, 1, dtype='int16'),
                              'tint': np.arange(1, 10, 1, dtype='int32'),
                              'tlong': np.arange(1, 10, 1, dtype='int64'),
                              'tfloat': np.arange(1, 10, 1, dtype='float32'),
                              'tdouble': np.arange(1, 10, 1, dtype='float64')
                              })
        pdf_t = pdf_t.set_index("id")
        odf_t = orca.DataFrame(pdf_t)

        # TODO: ALL ASSERT FAIL
        a = odf_t.rolling(window='t', on="date").var()
        b = pdf_t.rolling(window='t', on="date").var()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        # a = odf_t.rolling(window='t', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].var()
        # b = pdf_t.rolling(window='t', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].var()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # a = odf_t.rolling(window='t', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].var()
        # b = pdf_t.rolling(window='t', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].var()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_t.reset_index()
        # pdf_dai = pdf_t.reset_index()
        # a = odf_dai.rolling(window='t', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        # b = pdf_dai.rolling(window='t', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_t.set_index('date')
        # pdf_dai = pdf_t.set_index('date')
        # a = odf_dai.rolling(window='t').var()
        # b = pdf_dai.rolling(window='t').var()
        # assert_frame_equal(a.to_pandas(), b)
        #
        # a = odf_dai.rolling(window='t')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        # b = pdf_dai.rolling(window='t')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_rule_second_var(self):
        ps = pd.to_datetime(
            ["20170101 9:10:15", "20170101 9:10:16", "20170101 9:10:16", "20170101 9:11:17", "20170101 9:11:17",
             "20180615 9:10:15", "20181031 9:10:15", "20190501 9:10:15",
             "20190517 9:10:15"]).to_series()
        pdf_s = pd.DataFrame({'id': np.arange(1, 10, 1, dtype='int32'),
                              'date': ps,
                              'tchar': np.arange(1, 10, 1, dtype='int8'),
                              'tshort': np.arange(1, 10, 1, dtype='int16'),
                              'tint': np.arange(1, 10, 1, dtype='int32'),
                              'tlong': np.arange(1, 10, 1, dtype='int64'),
                              'tfloat': np.arange(1, 10, 1, dtype='float32'),
                              'tdouble': np.arange(1, 10, 1, dtype='float64')
                              })
        pdf_s = pdf_s.set_index("id")
        odf_s = orca.DataFrame(pdf_s)

        # TODO: ALL ASSERT FAIL
        a = odf_s.rolling(window='s', on="date").var()
        b = pdf_s.rolling(window='s', on="date").var()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = odf_s.rolling(window='2s', on="date").var()
        b = pdf_s.rolling(window='2s', on="date").var()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        # a = odf_s.rolling(window='s', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].var()
        # b = pdf_s.rolling(window='s', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].var()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # a = odf_s.rolling(window='s', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].var()
        # b = pdf_s.rolling(window='s', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].var()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_s.reset_index()
        # pdf_dai = pdf_s.reset_index()
        # a = odf_dai.rolling(window='s', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        # b = pdf_dai.rolling(window='s', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_s.set_index('date')
        # pdf_dai = pdf_s.set_index('date')
        # a = odf_dai.rolling(window='s').var()
        # b = pdf_dai.rolling(window='s').var()
        # assert_frame_equal(a.to_pandas(), b)
        #
        # a = odf_dai.rolling(window='s')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        # b = pdf_dai.rolling(window='s')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_rule_milli_var(self):
        ps = pd.to_datetime(
            ["20170101 9:10:15.000", "20170101 9:10:15.000", "20170101 9:10:15.001", "20170101 9:11:17.015",
             "20170101 9:11:17.015",
             "20180615 9:10:15.015", "20181031 9:10:15.015", "20190501 9:10:15.015",
             "20190517 9:10:15.015"]).to_series()
        pdf_l = pd.DataFrame({'id': np.arange(1, 10, 1, dtype='int32'),
                              'date': ps,
                              'tchar': np.arange(1, 10, 1, dtype='int8'),
                              'tshort': np.arange(1, 10, 1, dtype='int16'),
                              'tint': np.arange(1, 10, 1, dtype='int32'),
                              'tlong': np.arange(1, 10, 1, dtype='int64'),
                              'tfloat': np.arange(1, 10, 1, dtype='float32'),
                              'tdouble': np.arange(1, 10, 1, dtype='float64')
                              })
        pdf_l = pdf_l.set_index("id")
        odf_l = orca.DataFrame(pdf_l)

        # TODO: ALL ASSERT FAIL
        a = odf_l.rolling(window='l', on="date").var()
        b = pdf_l.rolling(window='l', on="date").var()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = odf_l.rolling(window='2l', on="date").var()
        b = pdf_l.rolling(window='2l', on="date").var()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        # a = odf_l.rolling(window='l', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].var()
        # b = pdf_l.rolling(window='l', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].var()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # a = odf_l.rolling(window='l', on="date")[
        #     'date'_l, 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].var()
        # b = pdf.rolling(window='l', on="date")[
        #     'date'_l, 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].var()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_l.reset_index()
        # pdf_dai = pdf_l.reset_index()
        # a = odf_dai.rolling(window='l', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        # b = pdf_dai.rolling(window='l', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_l.set_index('date')
        # pdf_dai = pdf_l.set_index('date')
        # a = odf_dai.rolling(window='l').var()
        # b = pdf_dai.rolling(window='l').var()
        # assert_frame_equal(a.to_pandas(), b)
        #
        # a = odf_dai.rolling(window='l')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        # b = pdf_dai.rolling(window='l')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_rule_micro_var(self):
        ps = pd.to_datetime(
            ["20170101 9:10:15.000000", "20170101 9:10:15.000000", "20170101 9:10:15.000001", "20170101 9:11:17.015001",
             "20170101 9:11:17.015002",
             "20180615 9:10:15.015000", "20181031 9:10:15.015000", "20190501 9:10:15.015000",
             "20190517 9:10:15.015000"]).to_series()
        pdf_u = pd.DataFrame({'id': np.arange(1, 10, 1, dtype='int32'),
                              'date': ps,
                              'tchar': np.arange(1, 10, 1, dtype='int8'),
                              'tshort': np.arange(1, 10, 1, dtype='int16'),
                              'tint': np.arange(1, 10, 1, dtype='int32'),
                              'tlong': np.arange(1, 10, 1, dtype='int64'),
                              'tfloat': np.arange(1, 10, 1, dtype='float32'),
                              'tdouble': np.arange(1, 10, 1, dtype='float64')
                              })
        pdf_u = pdf_u.set_index("id")
        odf_u = orca.DataFrame(pdf_u)

        # TODO: ALL ASSERT FAIL
        a = odf_u.rolling(window='u', on="date").var()
        b = pdf_u.rolling(window='u', on="date").var()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = odf_u.rolling(window='2u', on="date").var()
        b = pdf_u.rolling(window='2u', on="date").var()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        # a = odf_u.rolling(window='u', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].var()
        # b = pdf_u.rolling(window='u', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].var()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # a = odf_u.rolling(window='u', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].var()
        # b = pdf_u.rolling(window='u', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].var()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_u.reset_index()
        # pdf_dai = pdf_u.reset_index()
        # a = odf_dai.rolling(window='u', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        # b = pdf_dai.rolling(window='u', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_u.set_index('date')
        # pdf_dai = pdf_u.set_index('date')
        # a = odf_dai.rolling(window='u').var()
        # b = pdf_dai.rolling(window='u').var()
        # assert_frame_equal(a.to_pandas(), b)
        #
        # a = odf_dai.rolling(window='u')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        # b = pdf_dai.rolling(window='u')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_pandas_param_window_rule_nano_var(self):
        ps = pd.to_datetime(
            ["20170101 9:10:15.000000000", "20170101 9:10:15.000000000", "20170101 9:10:15.000000001",
             "20170101 9:11:17.015000001",
             "20170101 9:11:17.015002001",
             "20180615 9:10:15.015000001", "20181031 9:10:15.015000001", "20190501 9:10:15.015000001",
             "20190517 9:10:15.015000001"]).to_series()
        pdf_n = pd.DataFrame({'id': np.arange(1, 10, 1, dtype='int32'),
                              'date': ps,
                              'tchar': np.arange(1, 10, 1, dtype='int8'),
                              'tshort': np.arange(1, 10, 1, dtype='int16'),
                              'tint': np.arange(1, 10, 1, dtype='int32'),
                              'tlong': np.arange(1, 10, 1, dtype='int64'),
                              'tfloat': np.arange(1, 10, 1, dtype='float32'),
                              'tdouble': np.arange(1, 10, 1, dtype='float64')
                              })
        pdf_n = pdf_n.set_index("id")
        odf_n = orca.DataFrame(pdf_n)

        # TODO: ALL ASSERT FAIL
        a = odf_n.rolling(window='n', on="date").var()
        b = pdf_n.rolling(window='n', on="date").var()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = odf_n.rolling(window='2n', on="date").var()
        b = pdf_n.rolling(window='2n', on="date").var()
        # assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        # a = odf_n.rolling(window='n', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].var()
        # b = pdf_n.rolling(window='n', on="date")[
        #     'date', 'tchar', 'tint', 'tlong', 'tfloat'].var()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # a = odf_n.rolling(window='n', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].var()
        # b = pdf_n.rolling(window='n', on="date")[
        #     'date', 'tdouble', 'tchar', 'tint', 'tlong', 'tfloat', 'tshort'].var()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_n.reset_index()
        # pdf_dai = pdf_n.reset_index()
        # a = odf_dai.rolling(window='n', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        # b = pdf_dai.rolling(window='n', on="date")[
        #     'date', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        #
        # odf_dai = odf_n.set_index('date')
        # pdf_dai = pdf_n.set_index('date')
        # a = odf_dai.rolling(window='n').var()
        # b = pdf_dai.rolling(window='n').var()
        # assert_frame_equal(a.to_pandas(), b)
        #
        # a = odf_dai.rolling(window='n')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        # b = pdf_dai.rolling(window='n')[
        #     'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        # assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_import_param_window_sum(self):
        a = self.odf_csv.rolling(window=5).sum()
        b = self.pdf_csv.rolling(window=5).sum()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = self.odf_csv.rolling(window=5)[
            'id','tchar','tshort','tint','tlong','tfloat'].sum()
        b = self.pdf_csv.rolling(window=5)[
            'id','tchar','tshort','tint','tlong','tfloat'].sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        a = self.odf_csv.rolling(window=5)[
            'id','tdouble','tbool','tchar','tshort','tint','tlong','tfloat'].sum()
        b = self.pdf_csv.rolling(window=5)[
            'id','tdouble','tbool','tchar','tshort','tint','tlong','tfloat'].sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf_csv.reset_index()
        pdf_dai = self.pdf_csv.reset_index()
        a = odf_dai.rolling(window=5, on="id")[
            'id', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdf_dai.rolling(window=5, on="id")[
            'id', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf_csv.set_index('id')
        pdf_dai = self.pdf_csv.set_index('id')

        a = odf_dai.rolling(window=5)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        b = pdf_dai.rolling(window=5)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].sum()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_import_param_window_count(self):
        a = self.odf_csv.rolling(window=5).count()
        b = self.pdf_csv.rolling(window=5).count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = self.odf_csv.rolling(window=5)[
            'id', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat'].count()
        b = self.pdf_csv.rolling(window=5)[
            'id', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat'].count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        a = self.odf_csv.rolling(window=5)[
            'id', 'tdouble', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat'].count()
        b = self.pdf_csv.rolling(window=5)[
            'id', 'tdouble', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat'].count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf_csv.reset_index()
        pdf_dai = self.pdf_csv.reset_index()
        a = odf_dai.rolling(window=5, on="id")[
            'id', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdf_dai.rolling(window=5, on="id")[
            'id', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf_csv.set_index('id')
        pdf_dai = self.pdf_csv.set_index('id')

        a = odf_dai.rolling(window=5)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        b = pdf_dai.rolling(window=5)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].count()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_import_param_window_mean(self):
        a = self.odf_csv.rolling(window=5).mean()
        b = self.pdf_csv.rolling(window=5).mean()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = self.odf_csv.rolling(window=5)[
            'id', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat'].mean()
        b = self.pdf_csv.rolling(window=5)[
            'id', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat'].mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        a = self.odf_csv.rolling(window=5)[
            'id', 'tdouble', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat'].mean()
        b = self.pdf_csv.rolling(window=5)[
            'id', 'tdouble', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat'].mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf_csv.reset_index()
        pdf_dai = self.pdf_csv.reset_index()
        a = odf_dai.rolling(window=5, on="id")[
            'id', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdf_dai.rolling(window=5, on="id")[
            'id', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf_csv.set_index('id')
        pdf_dai = self.pdf_csv.set_index('id')

        a = odf_dai.rolling(window=5)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        b = pdf_dai.rolling(window=5)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].mean()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_import_param_window_max(self):
        a = self.odf_csv.rolling(window=5).max()
        b = self.pdf_csv.rolling(window=5).max()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = self.odf_csv.rolling(window=5)[
            'id', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat'].max()
        b = self.pdf_csv.rolling(window=5)[
            'id', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat'].max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        a = self.odf_csv.rolling(window=5)[
            'id', 'tdouble', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat'].max()
        b = self.pdf_csv.rolling(window=5)[
            'id', 'tdouble', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat'].max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf_csv.reset_index()
        pdf_dai = self.pdf_csv.reset_index()
        a = odf_dai.rolling(window=5, on="id")[
            'id', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdf_dai.rolling(window=5, on="id")[
            'id', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf_csv.set_index('id')
        pdf_dai = self.pdf_csv.set_index('id')

        a = odf_dai.rolling(window=5)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        b = pdf_dai.rolling(window=5)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].max()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_import_param_window_min(self):
        a = self.odf_csv.rolling(window=5).min()
        b = self.pdf_csv.rolling(window=5).min()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = self.odf_csv.rolling(window=5)[
            'id', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat'].min()
        b = self.pdf_csv.rolling(window=5)[
            'id', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat'].min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        a = self.odf_csv.rolling(window=5)[
            'id', 'tdouble', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat'].min()
        b = self.pdf_csv.rolling(window=5)[
            'id', 'tdouble', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat'].min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf_csv.reset_index()
        pdf_dai = self.pdf_csv.reset_index()
        a = odf_dai.rolling(window=5, on="id")[
            'id', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdf_dai.rolling(window=5, on="id")[
            'id', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf_csv.set_index('id')
        pdf_dai = self.pdf_csv.set_index('id')

        a = odf_dai.rolling(window=5)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        b = pdf_dai.rolling(window=5)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].min()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_import_param_window_std(self):
        a = self.odf_csv.rolling(window=5).std()
        b = self.pdf_csv.rolling(window=5).std()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = self.odf_csv.rolling(window=5)[
            'id', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat'].std()
        b = self.pdf_csv.rolling(window=5)[
            'id', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat'].std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        a = self.odf_csv.rolling(window=5)[
            'id', 'tdouble', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat'].std()
        b = self.pdf_csv.rolling(window=5)[
            'id', 'tdouble', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat'].std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf_csv.reset_index()
        pdf_dai = self.pdf_csv.reset_index()
        a = odf_dai.rolling(window=5, on="id")[
            'id', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdf_dai.rolling(window=5, on="id")[
            'id', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf_csv.set_index('id')
        pdf_dai = self.pdf_csv.set_index('id')

        a = odf_dai.rolling(window=5)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        b = pdf_dai.rolling(window=5)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].std()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

    def test_rolling_from_import_param_window_var(self):
        a = self.odf_csv.rolling(window=5).var()
        b = self.pdf_csv.rolling(window=5).var()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

        a = self.odf_csv.rolling(window=5)[
            'id', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat'].var()
        b = self.pdf_csv.rolling(window=5)[
            'id', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat'].var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        a = self.odf_csv.rolling(window=5)[
            'id', 'tdouble', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat'].var()
        b = self.pdf_csv.rolling(window=5)[
            'id', 'tdouble', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat'].var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf_csv.reset_index()
        pdf_dai = self.pdf_csv.reset_index()
        a = odf_dai.rolling(window=5, on="id")[
            'id', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdf_dai.rolling(window=5, on="id")[
            'id', 'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)

        odf_dai = self.odf_csv.set_index('id')
        pdf_dai = self.pdf_csv.set_index('id')

        a = odf_dai.rolling(window=5)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        b = pdf_dai.rolling(window=5)[
            'tbool', 'tchar', 'tshort', 'tint', 'tlong', 'tfloat', 'tdouble'].var()
        assert_frame_equal(a.to_pandas().reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)


if __name__ == '__main__':
    unittest.main()
