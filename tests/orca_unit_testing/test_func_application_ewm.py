import unittest
import orca
import os.path as path
from setup.settings import *
from pandas.util.testing import *


class Csv:
    pdf_csv = None
    odf_csv = None


class EwmTest(unittest.TestCase):
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
        }, index=[0, 1, 3, 5, 3, 8, 9, 9, 9])

    @property
    def odf(self):
        return orca.DataFrame(self.pdf)

    @property
    def pdf_other(self):
        return pd.DataFrame({
            'a': [1, 2, 0, 4, 1, 9, 7, 2, 9],
            'b': [4, 9, 8, 1, 4, 2, 3, 6, 0],
        }, index=[0, 1, 3, 3, 6, 6, 10, 9, 7])

    @property
    def odf_other(self):
        return orca.DataFrame(self.pdf_other)

    def test_dataframe_function_application_ewm(self):
        ewmp = self.pdf.ewm(com=0.5)
        ewmo = self.odf.ewm(com=0.5)
        assert_frame_equal(ewmo.mean().to_pandas(), ewmp.mean())
        assert_frame_equal(ewmo.std().to_pandas(), ewmp.std())
        assert_frame_equal(ewmo.var().to_pandas(), ewmp.var())
        # TODO: pairwise
        # assert_frame_equal(ewmo.corr(self.odf_other).to_pandas(), ewmp.corr(self.pdf_other))
        # assert_frame_equal(ewmo.cov(self.odf_other).to_pandas(), ewmp.cov(self.pdf_other))

        ewmp = self.pdf.ewm(span=5)
        ewmo = self.odf.ewm(span=5)
        assert_frame_equal(ewmo.mean().to_pandas(), ewmp.mean())
        assert_frame_equal(ewmo.std().to_pandas(), ewmp.std())
        assert_frame_equal(ewmo.var().to_pandas(), ewmp.var())

        ewmp = self.pdf.ewm(halflife=7)
        ewmo = self.odf.ewm(halflife=7)
        assert_frame_equal(ewmo.mean().to_pandas(), ewmp.mean())
        assert_frame_equal(ewmo.std().to_pandas(), ewmp.std())
        assert_frame_equal(ewmo.var().to_pandas(), ewmp.var())

        ewmp = self.pdf.ewm(alpha=0.2)
        ewmo = self.odf.ewm(alpha=0.2)
        assert_frame_equal(ewmo.mean().to_pandas(), ewmp.mean())
        assert_frame_equal(ewmo.std().to_pandas(), ewmp.std())
        assert_frame_equal(ewmo.var().to_pandas(), ewmp.var())

        ewmp = self.pdf.ewm(alpha=0.7, min_periods=2, adjust=False, ignore_na=True)
        ewmo = self.odf.ewm(alpha=0.7, min_periods=2, adjust=False, ignore_na=True)
        assert_frame_equal(ewmo.mean().to_pandas(), ewmp.mean())
        assert_frame_equal(ewmo.std().to_pandas(), ewmp.std())
        assert_frame_equal(ewmo.var().to_pandas(), ewmp.var())

        pdf = self.pdf
        odf = self.odf

        # condition filter
        assert_frame_equal(odf[odf['a'] > 4].ewm(com=0.5).mean().to_pandas(), pdf[pdf['a'] > 4].ewm(com=0.5).mean())
        assert_frame_equal(odf[odf['a'] > 4].ewm(com=0.5).std().to_pandas(), pdf[pdf['a'] > 4].ewm(com=0.5).std())
        assert_frame_equal(odf[odf['a'] > 4].ewm(com=0.5).var().to_pandas(), pdf[pdf['a'] > 4].ewm(com=0.5).var())

        # expressions

        # TODO；unverified pandas error
        # csvp = self.pdf_csv.ewm(com=0.5)
        # csvo = self.odf_csv.ewm(com=0.5)
        # csvp.mean()
        # assert_frame_equal(csvo.mean().to_pandas(), csvp.mean())
        # assert_frame_equal(csvo.std().to_pandas(), csvp.std())
        # assert_frame_equal(csvo.var().to_pandas(), csvp.var())
        #
        # # TODO: pairwise
        # # assert_frame_equal(ewmo.corr().to_pandas(), ewmp.corr())
        # # assert_frame_equal(ewmo.cov().to_pandas(), ewmp.cov())
        #
        # csvp = self.pdf_csv.ewm(span=5)
        # csvo = self.odf_csv.ewm(span=5)
        # assert_frame_equal(csvo.mean().to_pandas(), csvp.mean())
        # assert_frame_equal(csvo.std().to_pandas(), csvp.std())
        # assert_frame_equal(csvo.var().to_pandas(), csvp.var())
        #
        # csvp = self.pdf_csv.ewm(halflife=7)
        # csvo = self.odf_csv.ewm(halflife=7)
        # assert_frame_equal(csvo.mean().to_pandas(), csvp.mean())
        # assert_frame_equal(csvo.std().to_pandas(), csvp.std())
        # assert_frame_equal(csvo.var().to_pandas(), csvp.var())
        #
        # csvp = self.pdf_csv.ewm(alpha=0.2)
        # csvo = self.odf_csv.ewm(alpha=0.2)
        # assert_frame_equal(csvo.mean().to_pandas(), csvp.mean())
        # assert_frame_equal(csvo.std().to_pandas(), csvp.std())
        # assert_frame_equal(csvo.var().to_pandas(), csvp.var())
        #
        # csvp = self.pdf_csv.ewm(alpha=0.7, min_periods=2, adjust=False, ignore_na=True)
        # csvo = self.odf_csv.ewm(alpha=0.7, min_periods=2, adjust=False, ignore_na=True)
        # assert_frame_equal(csvo.mean().to_pandas(), csvp.mean())
        # assert_frame_equal(csvo.std().to_pandas(), csvp.std())
        # assert_frame_equal(csvo.var().to_pandas(), csvp.var())

if __name__ == '__main__':
    unittest.main()
