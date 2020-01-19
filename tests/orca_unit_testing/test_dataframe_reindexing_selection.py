import unittest
import orca
import os.path as path
from setup.settings import *
from pandas.util.testing import *


class Csv:
    pdf_csv = None
    odf_csv = None


class DataFrameReindexingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # configure data directory
        DATA_DIR = path.abspath(path.join(__file__, "../setup/data"))
        fileName = 'USPricesSample.csv'
        data = os.path.join(DATA_DIR, fileName)
        data = data.replace('\\', '/')

        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

        Csv.odf_csv = orca.read_csv(data, dtype={"DLSTCD": np.float32, "DLPRC": np.float32})
        # pdf from import
        Csv.pdf_csv = pd.read_csv(data, parse_dates=[1], dtype={"DLSTCD": np.float32, "DLPRC": np.float32})
        Csv.odf_csv = Csv.odf_csv.drop(columns=['DLRET'])
        Csv.pdf_csv.drop(columns=['DLRET'], inplace=True)

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

    def test_dataframe_reindexing_selection_label_mainpulation_between_time(self):
        idx = pd.date_range('2018-04-09', periods=4, freq='1D20min')
        pdf = pd.DataFrame({'A': [1, 2, 3, 4]}, index=idx)
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.between_time('0:15', '0:45').to_pandas(), pdf.between_time('0:15', '0:45'))
        assert_frame_equal(odf.between_time('0:45', '0:15').to_pandas(), pdf.between_time('0:45', '0:15'))

    def test_dataframe_reindexing_selection_label_mainpulation_take(self):
        n = np.array([0, 1, 4])
        assert_frame_equal(self.odf.take(n).to_pandas(), self.pdf.take(n))
        assert_frame_equal(self.odf.take([]).to_pandas(), self.pdf.take([]))
        assert_frame_equal(self.odf.take([0, 1], axis=1).to_pandas(), self.pdf.take([0, 1], axis=1))
        assert_frame_equal(self.odf.take([-1, -2], axis=0).to_pandas(), self.pdf.take([-1, -2], axis=0))
        n = np.random.randint(0, 2999, 100)
        assert_frame_equal(self.odf_csv.take(n).to_pandas(), self.pdf_csv.take(n), check_dtype=False)
        assert_frame_equal(self.odf_csv.take([0, 1, 5, 7, 11, 15], axis=1).to_pandas(), self.pdf_csv.take([0, 1, 5, 7, 11, 15], axis=1), check_dtype=False)

    def test_dataframe_reindexing_selection_label_mainpulation_equals(self):
        pdf = pd.DataFrame({1: [10], 2: [20]})
        p_exactly_equal = pd.DataFrame({1: [10], 2: [20]})
        p_different_column_type = pd.DataFrame({1.0: [10], 2.0: [20]})
        p_different_data_type = pd.DataFrame({1: [10.0], 2: [20.0]})
        odf = orca.DataFrame(pdf)
        o_exactly_equal = orca.DataFrame(p_exactly_equal)
        o_different_column_type = orca.DataFrame(p_different_column_type)
        o_different_data_type = orca.DataFrame(p_different_data_type)
        self.assertEqual(odf.equals(o_exactly_equal), pdf.equals(p_exactly_equal))
        self.assertEqual(odf.equals(o_different_column_type), pdf.equals(p_different_column_type))
        self.assertEqual(odf.equals(o_different_data_type), pdf.equals(p_different_data_type))


if __name__ == '__main__':
    unittest.main()
