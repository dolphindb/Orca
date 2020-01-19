import unittest
import orca
import os.path as path
from setup.settings import *
from pandas.util.testing import *


class Csv:
    pdf_csv = None
    odf_csv = None


class DataFrameFuncApplicationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # configure data directory
        DATA_DIR = path.abspath(path.join(__file__, "../setup/data"))
        fileName = 'USPricesSample.csv'
        data = os.path.join(DATA_DIR, fileName)
        data = data.replace('\\', '/')

        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")

        # Csv.odf_csv = orca.read_csv(data, dtype={"DLSTCD": np.float32, "DLPRC": np.float32})
        Csv.odf_csv = orca.read_csv(data, dtype={"PERMNO": np.int32, "date": 'DATE', "TRDSTAT": 'SYMBOL',
                                                           "DLSTCD": np.float32, "DLPRC": np.float32, "VOL": np.float32,
                                                           "SHROUT": np.float32, "CFACPR":np.float32, "CFACSHR": np.float32})
        # pdf from import
        Csv.pdf_csv = pd.read_csv(data, parse_dates=[1], dtype={"PERMNO": np.int32, "SHRCD": np.int32, "HEXCD": np.int32,
                                                                "DLSTCD": np.float32, "DLPRC": np.float32,
                                                                "VOL": np.float32, "SHROUT": np.float32})
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


    def test_dataframe_function_application_apply(self):
        pdf = pd.DataFrame([[4, 9]] * 3, columns=['A', 'B'])
        odf = orca.DataFrame([[4, 9]] * 3, columns=['A', 'B'])
        assert_frame_equal(odf.apply(np.sqrt).to_pandas(), pdf.apply(np.sqrt))
        assert_frame_equal(odf.apply(np.sum, axis=0).to_pandas(), pdf.apply(np.sum, axis=0).to_frame())

    def test_dataframe_function_application_applymap(self):
        pdf = pd.DataFrame([[4, 9]] * 3, columns=['A', 'B'])
        odf = orca.DataFrame([[4, 9]] * 3, columns=['A', 'B'])
        assert_frame_equal(odf.applymap(np.sqrt).to_pandas(), pdf.applymap(np.sqrt))

    def test_dataframe_function_application_pipe(self):
        # TODO NOT IMPLEMENTED ERROR
        pass

    def test_dataframe_function_application_agg(self):
        pdf = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9], [np.nan, np.nan, np.nan]], columns=['A', 'B', 'C'])
        odf = orca.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9], [np.nan, np.nan, np.nan]], columns=['A', 'B', 'C'])
        assert_frame_equal(odf.agg(['sum', 'min']).to_pandas(), pdf.agg(['sum', 'min']))

    def test_dataframe_function_application_aggregate(self):
        pdf = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9], [np.nan, np.nan, np.nan]], columns=['A', 'B', 'C'])
        odf = orca.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9], [np.nan, np.nan, np.nan]], columns=['A', 'B', 'C'])
        assert_frame_equal(odf.aggregate(['sum', 'min']).to_pandas(), pdf.aggregate(['sum', 'min']))
        assert_frame_equal(odf.aggregate({'A': ['sum', 'min'], 'B': ['min', 'max']}).to_pandas(),
                           pdf.aggregate({'A': ['sum', 'min'], 'B': ['min', 'max']}))
        # TODO:DIFFS
        # assert_frame_equal(odf.aggregate("mean", axis="columns").to_pandas(), pdf.aggregate("mean", axis="columns"))

    def test_dataframe_function_application_transform(self):
        pdf = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9], [np.nan, np.nan, np.nan]], columns=['A', 'B', 'C'])
        odf = orca.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9], [np.nan, np.nan, np.nan]], columns=['A', 'B', 'C'])
        assert_frame_equal(odf.transform([np.sqrt, np.exp]).to_pandas(), pdf.transform([np.sqrt, np.exp]))

    def test_dataframe_function_application_expanding(self):
        # pdf = pd.DataFrame({'B': [0, 1, 2, np.nan, 4]})
        # odf = orca.DataFrame({'B': [0, 1, 2, np.nan, 4]})
        # assert_frame_equal(pdf.expanding(2).sum(), odf.expanding(2).sum().to_pandas())
        # TODO NOT IMPLEMENTED ERROR
        pass


if __name__ == '__main__':
    unittest.main()
