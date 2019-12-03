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
                               'tsymbol': np.repeat(['a', 'b', 'c', 'd', 'e', 'QWW', 'FEA', 'FFW', 'DER', 'POD'], re),
                               'tchar': np.repeat(np.arange(1, 11, 1, dtype='int8'), re),
                               'tshort': np.repeat(np.arange(1, 11, 1, dtype='int16'), re),
                               'tint': np.repeat(np.arange(1, 11, 1, dtype='int32'), re),
                               'tlong': np.repeat(np.arange(1, 11, 1, dtype='int64'), re),
                               'tfloat': np.repeat(np.arange(1, 11, 1, dtype='float32'), re),
                               'tdouble': np.repeat(np.arange(1, 11, 1, dtype='float64'), re)
                               })
        return pdf_da.set_index("id")

    @property
    def odf(self):
        return orca.DataFrame(self.pdf)

    def test_rolling_from_pandas_param_window_sum(self):
        a = self.odf.rolling(window=5, on="date").sum()
        b = self.pdf.rolling(window=5, on="date").sum()
        assert_frame_equal(a.to_pandas(), b)

    def test_rolling_from_pandas_param_window_count(self):
        a = self.odf.rolling(window=5, on="date").count()
        b = self.pdf.rolling(window=5, on="date").count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_rolling_from_pandas_param_window_mean(self):
        a = self.odf.rolling(window=5, on="date").mean()
        b = self.pdf.rolling(window=5, on="date").mean()
        assert_frame_equal(a.to_pandas(), b)

    def test_rolling_from_pandas_param_window_max(self):
        a = self.odf.rolling(window=5, on="date").max()
        b = self.pdf.rolling(window=5, on="date").max()
        assert_frame_equal(a.to_pandas(), b)

    def test_rolling_from_pandas_param_window_min(self):
        a = self.odf.rolling(window=5, on="date").min()
        b = self.pdf.rolling(window=5, on="date").min()
        assert_frame_equal(a.to_pandas(), b)

    def test_rolling_from_pandas_param_window_std(self):
        a = self.odf.rolling(window=5, on="date").std()
        b = self.pdf.rolling(window=5, on="date").std()
        assert_frame_equal(a.to_pandas(), b)

    def test_rolling_from_pandas_param_window_var(self):
        a = self.odf.rolling(window=5, on="date").var()
        b = self.pdf.rolling(window=5, on="date").var()
        assert_frame_equal(a.to_pandas(), b)

    def test_rolling_from_import_param_window_sum(self):
        a = self.odf_csv.rolling(window=5).sum()
        b = self.pdf_csv.rolling(window=5).sum()
        assert_frame_equal(a.to_pandas(), b)

    def test_rolling_from_import_param_window_count(self):
        a = self.odf_csv.rolling(window=5).count()
        b = self.pdf_csv.rolling(window=5).count()
        assert_frame_equal(a.to_pandas(), b, check_dtype=False)

    def test_rolling_from_import_param_window_mean(self):
        a = self.odf_csv.rolling(window=5).mean()
        b = self.pdf_csv.rolling(window=5).mean()
        assert_frame_equal(a.to_pandas(), b)

    def test_rolling_from_import_param_window_max(self):
        a = self.odf_csv.rolling(window=5).max()
        b = self.pdf_csv.rolling(window=5).max()
        assert_frame_equal(a.to_pandas(), b)

    def test_rolling_from_import_param_window_min(self):
        a = self.odf_csv.rolling(window=5).min()
        b = self.pdf_csv.rolling(window=5).min()
        assert_frame_equal(a.to_pandas(), b)

    def test_rolling_from_import_param_window_std(self):
        a = self.odf_csv.rolling(window=5).std()
        b = self.pdf_csv.rolling(window=5).std()
        assert_frame_equal(a.to_pandas(), b)

    def test_rolling_from_import_param_window_var(self):
        a = self.odf_csv.rolling(window=5).var()
        b = self.pdf_csv.rolling(window=5).var()
        assert_frame_equal(a.to_pandas(), b)


if __name__ == '__main__':
    unittest.main()
