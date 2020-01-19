import unittest
import orca
import os.path as path
from setup.settings import *
from pandas.util.testing import *


class Csv:
    pdf_csv = None
    odf_csv = None


class DataFrameComputationsTest(unittest.TestCase):
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
        n = 100  # note that n should be a multiple of 10
        re = n / 10
        return pd.DataFrame({
                               # 'date': np.repeat(pd.date_range('2019.08.01', periods=10, freq='D'), re),
                               # 'tsymbol': np.repeat(['a', 'b', 'c', 'd', 'e', 'QWW', 'FEA', 'FFW', 'DER', 'POD'], re),
                               # 'tbool': np.repeat(np.repeat(np.arange(2, dtype='bool'), 5), re),
                               'tchar': np.repeat(np.arange(1, 11, 1, dtype='int8'), re),
                               'tshort': np.repeat(np.arange(1, 11, 1, dtype='int16'), re),
                               'tint': np.repeat(np.arange(1, 11, 1, dtype='int32'), re),
                               'tlong': np.repeat(np.arange(1, 11, 1, dtype='int64'), re),
                               # 'tfloat': np.repeat(np.arange(1, 11, 1, dtype='float32'), re),
                               # 'tdouble': np.repeat(np.arange(1, 11, 1, dtype='float64'), re)
                               }, index=pd.Index(np.arange(1, n + 1, 1, dtype='int32'), name="id"))

    @property
    def odf(self):
        return orca.DataFrame(self.pdf)

    def test_dataframe_computations(self):
        # from csv
        assert_series_equal(self.odf_csv.drop(columns=['date', 'TICKER', 'TRDSTAT', 'CUSIP']).all().to_pandas(),
                            self.pdf_csv.drop(columns=['date', 'TICKER', 'TRDSTAT', 'CUSIP']).all())
        assert_series_equal(self.odf_csv.drop(columns=['date', 'TICKER', 'TRDSTAT', 'CUSIP']).any().to_pandas(),
                            self.pdf_csv.drop(columns=['date', 'TICKER', 'TRDSTAT', 'CUSIP']).any())
        assert_frame_equal(self.odf_csv.corr().to_pandas(), self.pdf_csv.corr(), check_dtype=False)
        assert_series_equal(self.odf_csv.count().to_pandas(), self.pdf_csv.count(), check_dtype=False)
        assert_frame_equal(self.odf_csv.cov().to_pandas(), self.pdf_csv.cov())
        assert_series_equal(
            self.odf_csv.drop(columns=['date', 'TICKER', 'TRDSTAT', 'CUSIP']).kurt().to_pandas().fillna(0.0),
            self.pdf_csv.drop(columns=['date', 'TICKER', 'TRDSTAT', 'CUSIP']).kurt().fillna(0.0), check_dtype=False,
            check_less_precise=True)
        assert_series_equal(
            self.odf_csv.drop(columns=['date', 'TICKER', 'TRDSTAT', 'CUSIP']).kurtosis().to_pandas().fillna(0.0),
            self.pdf_csv.kurtosis().fillna(0.0), check_dtype=False, check_less_precise=True, check_exact=False)
        assert_series_equal(self.odf_csv.mean().to_pandas(), self.pdf_csv.mean())
        assert_series_equal(self.odf_csv.median().to_pandas(), self.pdf_csv.median())
        # TODO：diffs
        # assert_series_equal(self.odf_csv.drop(columns=['date', 'RET']).min().to_pandas(),
        #                     self.pdf_csv.drop(columns=['date', 'RET']).min(), check_dtype=False, check_exact=False)
        # assert_series_equal(self.odf_csv.max().to_pandas(), self.pdf_csv.max(),
        #                     check_dtype=False, check_less_precise=True)
        assert_series_equal(self.odf_csv.mode().to_pandas().drop('DLSTCD'),
                            self.pdf_csv.mode().drop(columns=['date', 'TICKER', 'TRDSTAT', 'CUSIP', 'DLSTCD']).iloc[0],
                            check_dtype=False, check_names=False)
        assert_series_equal(self.odf_csv.prod().to_pandas(), self.pdf_csv.prod())
        assert_series_equal(self.odf_csv.product().to_pandas(), self.pdf_csv.product())
        assert_frame_equal(self.odf_csv.quantile([0.3, 0.5]).to_pandas(), self.pdf_csv.quantile([0.3, 0.5]), check_dtype=False)
        assert_series_equal(
            self.odf_csv.drop(columns=['date', 'TICKER', 'TRDSTAT', 'CUSIP']).skew().to_pandas().fillna(0.0),
            self.pdf_csv.skew().fillna(0.0), check_dtype=False, check_less_precise=True, check_exact=False)
        assert_series_equal(self.odf_csv.sum().to_pandas(),
                            self.pdf_csv.drop(columns=['TICKER', 'TRDSTAT', 'CUSIP']).sum())
        assert_series_equal(self.odf_csv.std().to_pandas(), self.pdf_csv.std())
        assert_series_equal(self.odf_csv.var().to_pandas(), self.pdf_csv.var())

        # from construction
        assert_series_equal(self.odf.all().to_pandas(), self.pdf.all())
        assert_series_equal(self.odf.any().to_pandas(), self.pdf.any())
        assert_frame_equal(self.odf.corr().to_pandas(), self.pdf.corr())
        assert_series_equal(self.odf.count().to_pandas(), self.pdf.count(), check_dtype=False)
        assert_frame_equal(self.odf.cov().to_pandas(), self.pdf.cov())
        assert_series_equal(self.odf.kurt().to_pandas(), self.pdf.kurt())
        assert_series_equal(self.odf.kurtosis().to_pandas(), self.pdf.kurtosis())
        assert_series_equal(self.odf.mean().to_pandas(), self.pdf.mean())
        assert_series_equal(self.odf.median().to_pandas(), self.pdf.median())
        assert_series_equal(self.odf.min().to_pandas(), self.pdf.min())
        assert_series_equal(self.odf.max().to_pandas(), self.pdf.max())
        pdf = pd.DataFrame({
            'a': [1, 2, 3, 4, 5, 6, 7, 9, 9],
            'b': [4, 5, 6, 3, 2, 1, 0, 0, 0],
        }, index=[0, 1, 3, 5, 6, 8, 9, 9, 9])
        odf = orca.DataFrame(pdf)
        assert_series_equal(odf.mode().to_pandas(), pdf.mode().iloc[0], check_dtype=False, check_names=False)
        assert_series_equal(self.odf.prod().to_pandas(), self.pdf.prod())
        assert_series_equal(self.odf.product().to_pandas(), self.pdf.product())
        pdf = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": ['a', 'b', 'c']})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.quantile([0.3, 0.5]).to_pandas(), pdf.quantile([0.3, 0.5]))
        assert_frame_equal(self.odf.quantile([0.3, 0.5]).to_pandas(), self.pdf.quantile([0.3, 0.5]))
        assert_series_equal(self.odf.skew().to_pandas(), self.pdf.skew())
        assert_series_equal(self.odf.sum().to_pandas(), self.pdf.sum())
        assert_series_equal(self.odf.std().to_pandas(), self.pdf.std())
        assert_series_equal(self.odf.var().to_pandas(), self.pdf.var())

        assert_frame_equal(self.odf.pct_change().to_pandas(), self.pdf.pct_change())
        assert_series_equal(self.odf.mad().to_pandas(), self.pdf.mad())
        assert_series_equal(self.odf.sem().to_pandas(), self.pdf.sem())
        assert_frame_equal(self.odf.cumsum().to_pandas(), self.pdf.cumsum())
        assert_frame_equal(self.odf.cummax().to_pandas(), self.pdf.cummax(), check_dtype=False)
        assert_frame_equal(self.odf.cummin().to_pandas(), self.pdf.cummin(), check_dtype=False)
        assert_frame_equal(self.odf.cumprod().to_pandas(), self.pdf.cumprod())
        assert_frame_equal(self.odf.describe(), self.pdf.describe())

    def test_dataframe_computations_describe(self):
        pdf = pd.DataFrame(
            {'categorical': pd.Categorical(['d', 'e', 'f']), 'numeric': [1, 2, 3], 'object': ['a', 'b', 'c']})
        odf = orca.DataFrame(pdf)
        assert_frame_equal(odf.describe(), pdf.describe())
        # assert_frame_equal(odf.describe(include='all'), pdf.describe(include='all'))
        assert_series_equal(odf.numeric.describe(), pdf.numeric.describe(), check_names=False)
        assert_frame_equal(odf.describe(include=[np.number]), pdf.describe(include=[np.number]))
        # assert_frame_equal(odf.describe(include=[np.object]), pdf.describe(include=[np.object]))
        # assert_frame_equal(odf.describe(include=['category']), pdf.describe(include=['category']))
        # assert_frame_equal(odf.describe(exclude=[np.number]), pdf.describe(exclude=[np.number]))
        # assert_frame_equal(odf.describe(exclude=[np.object]), pdf.describe(exclude=[np.object]))

    def test_dataframe_computations_param_axis(self):
        assert_series_equal(self.odf_csv.sum(axis=1).to_pandas(), self.pdf_csv.sum(axis=1))
        assert_series_equal(self.odf_csv.std(axis=1).to_pandas(), self.pdf_csv.std(axis=1))
        assert_series_equal(self.odf_csv.var(axis=1).to_pandas(), self.pdf_csv.var(axis=1))
        assert_series_equal(self.odf_csv.mean(axis=1).to_pandas(), self.pdf_csv.mean(axis=1))
        assert_series_equal(self.odf_csv.min(axis=1).to_pandas(), self.pdf_csv.min(axis=1))
        assert_series_equal(self.odf_csv.max(axis=1).to_pandas(), self.pdf_csv.max(axis=1))

        assert_series_equal(self.odf.sum(axis=1).to_pandas(), self.pdf.sum(axis=1))
        assert_series_equal(self.odf.std(axis=1).to_pandas(), self.pdf.std(axis=1))
        assert_series_equal(self.odf.var(axis=1).to_pandas(), self.pdf.var(axis=1))
        assert_series_equal(self.odf.mean(axis=1).to_pandas(), self.pdf.mean(axis=1))
        assert_series_equal(self.odf.min(axis=1).to_pandas(), self.pdf.min(axis=1))
        assert_series_equal(self.odf.max(axis=1).to_pandas(), self.pdf.max(axis=1))

    def test_dataframe_computations_param_level(self):
        idx = pd.MultiIndex.from_arrays([[1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
                                         ['dog', 'falcon', 'dog', 'falcon', 'dog', 'falcon', 'dog', 'falcon', 'spider',
                                          'fish', 'spider', 'fish', 'spider', 'spider']],
                                        names=['id', 'animal'])
        pdf = pd.DataFrame({'weight': [2, 5, 4, 3, 2, 3, 2, 3, 4, 2, 2, 4, 5, 3]}, index=idx)
        odf = orca.DataFrame(pdf)
        # TODO: SORT_INDEX() HAS BEEN ADDED TO DIFFS
        assert_frame_equal(odf.all(level=1).to_pandas(), pdf.all(level=1).sort_index())
        assert_frame_equal(odf.any(level=1).to_pandas(), pdf.any(level=1).sort_index())
        assert_frame_equal(odf.count(level=1).to_pandas().sort_index(), pdf.count(level=1), check_dtype=False)
        assert_frame_equal(odf.kurt(level=1).to_pandas(), pdf.kurt(level=1).sort_index())
        assert_frame_equal(odf.kurtosis(level=1).to_pandas(), pdf.kurtosis(level=1).sort_index())
        assert_frame_equal(odf.mean(level=1).to_pandas(), pdf.mean(level=1), check_dtype=False)
        assert_frame_equal(odf.median(level=1).to_pandas(), pdf.median(level=1).sort_index(), check_dtype=False)
        assert_frame_equal(odf.min(level=1).to_pandas(), pdf.min(level=1), check_dtype=False)
        assert_frame_equal(odf.max(level=1).to_pandas(), pdf.max(level=1), check_dtype=False)
        # assert_frame_equal(odf.mode(level=1).to_pandas())
        assert_frame_equal(odf.prod(level=1).to_pandas(), pdf.prod(level=1).sort_index())
        assert_frame_equal(odf.product(level=1).to_pandas(), pdf.product(level=1).sort_index())
        assert_frame_equal(odf.skew(level=1).to_pandas(), pdf.skew(level=1).sort_index())
        assert_frame_equal(odf.sum(level=1).to_pandas(), pdf.sum(level=1))
        assert_frame_equal(odf.std(level=1).to_pandas(), pdf.std(level=1).sort_index())
        assert_frame_equal(odf.var(level=1).to_pandas(), pdf.var(level=1).sort_index())

    def test_dataframe_computations_topic_groupby_param_level(self):
        pdf = pd.DataFrame({'id': np.arange(1, 11, 1, dtype='int32'),
                            'date': pd.date_range('2019.08.01', periods=10, freq='D'),
                            'tsymbol': ['a', 'b', 'c', 'd', 'e', 'QWW', 'FEA', 'FFW', 'DER', 'POD'],
                            'tbool': np.repeat(np.arange(2, dtype='bool'), 5),
                            'tchar': np.arange(1, 11, 1, dtype='int8'),
                            'tshort': np.arange(1, 11, 1, dtype='int16'),
                            'tint': np.arange(1, 11, 1, dtype='int32'),
                            'tlong': np.arange(1, 11, 1, dtype='int64'),
                            'tfloat': np.arange(1, 11, 1, dtype='float32'),
                            'tdouble': np.arange(1, 11, 1, dtype='float64'),
                            })
        odf = orca.DataFrame(pdf)
        # TODO:ORCA LEVEL BUG
        # assert_frame_equal(pdf.groupby(['tsymbol','tbool']).min(level=1), odf.groupby(['tsymbol','tbool']).min(level=1).to_pandas(), check_dtype=False)
        # assert_frame_equal(pdf.groupby(['tsymbol','tbool']).max(level=1), odf.groupby(['tsymbol','tbool']).max(level=1).to_pandas(), check_dtype=False)
        # assert_frame_equal(pdf.groupby(['tsymbol','tbool']).prod(level=1), odf.groupby(['tsymbol','tbool']).prod(level=1).to_pandas())
        # assert_frame_equal(pdf.groupby(['tsymbol','tbool']).skew(level=1), odf.groupby(['tsymbol','tbool']).skew(level=1).to_pandas())
        # assert_frame_equal(pdf.groupby(['tsymbol','tbool']).sum(level=1), odf.groupby(['tsymbol','tbool']).sum(level=1).to_pandas())
        #
        # assert_frame_equal(pdf.groupby(['tsymbol', 'tbool']).min(level="tsymbol"),
        #                    odf.groupby(['tsymbol', 'tbool']).min(level="tsymbol").to_pandas(), check_dtype=False)
        # assert_frame_equal(pdf.groupby(['tsymbol', 'tbool']).max(level="tsymbol"),
        #                    odf.groupby(['tsymbol', 'tbool']).max(level="tsymbol").to_pandas(), check_dtype=False)
        # assert_frame_equal(pdf.groupby(['tsymbol', 'tbool']).prod(level="tsymbol"),
        #                    odf.groupby(['tsymbol', 'tbool']).prod(level="tsymbol").to_pandas())
        # assert_frame_equal(pdf.groupby(['tsymbol','tbool']).skew(level=1), odf.groupby(['tsymbol','tbool']).skew(level=1).to_pandas())
        # assert_frame_equal(pdf.groupby(['tsymbol', 'tbool']).sum(level="tsymbol"),
        #                    odf.groupby(['tsymbol', 'tbool']).sum(level="tsymbol").to_pandas())


if __name__ == '__main__':
    unittest.main()