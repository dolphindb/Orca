import unittest
import orca
import os.path as path
from setup.settings import *
from pandas.util.testing import *


def _create_odf_csv(datal, datar):
    dfsDatabase = "dfs://testMergeAsofDB"
    s = orca.default_session()
    dolphindb_script = """
                        login('admin', '123456')
                        if(existsDatabase('{dbPath}'))
                           dropDatabase('{dbPath}')
                        db=database('{dbPath}', VALUE, 2010.01M..2010.05M)
                        stb1=extractTextSchema('{data1}')
                        update stb1 set type="SYMBOL" where name="type"
                        stb2=extractTextSchema('{data2}')
                        update stb2 set type="SYMBOL" where name="ticker"
                        loadTextEx(db,`tickers,`date, '{data1}',,stb1)
                        loadTextEx(db,`values,`date, '{data2}',,stb2)
                        """.format(dbPath=dfsDatabase, data1=datal, data2=datar)
    s.run(dolphindb_script)


class Csv:
    odfs_csv_left = None
    odfs_csv_right = None
    pdf_csv_left = None
    pdf_csv_right = None


class DfsMergeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # configure data directory
        DATA_DIR = path.abspath(path.join(__file__, "../setup/data"))
        left_fileName = 'test_merge_asof_left_table.csv'
        right_fileName = 'test_merge_asof_right_table.csv'
        datal = os.path.join(DATA_DIR, left_fileName)
        datal= datal.replace('\\', '/')
        datar = os.path.join(DATA_DIR, right_fileName)
        datar = datar.replace('\\', '/')
        dfsDatabase = "dfs://testMergeAsofDB"

        # connect to a DolphinDB server
        orca.connect(HOST, PORT, "admin", "123456")
        _create_odf_csv(datal, datar)

        # import
        Csv.odfs_csv_left = orca.read_table(dfsDatabase, 'tickers')
        Csv.pdf_csv_left = pd.read_csv(datal, parse_dates=[0])
        Csv.odfs_csv_right = orca.read_table(dfsDatabase, 'values')
        Csv.pdf_csv_right = pd.read_csv(datar, parse_dates=[0])

    @property
    def odfs_csv_left(self):
        return Csv.odfs_csv_left

    @property
    def odfs_csv_right(self):
        return Csv.odfs_csv_right

    @property
    def pdf_csv_left(self):
        return Csv.pdf_csv_left

    @property
    def pdf_csv_right(self):
        return Csv.pdf_csv_right

    @property
    def odfs_csv_left_index(self):
        return Csv.odfs_csv_left.set_index("date")

    @property
    def odfs_csv_right_index(self):
        return Csv.odfs_csv_right.set_index("date")

    @property
    def pdf_csv_left_index(self):
        return Csv.pdf_csv_left.set_index("date")

    @property
    def pdf_csv_right_index(self):
        return Csv.pdf_csv_right.set_index("date")

    @property
    def odfs_bid_csv_left(self):
        return self.odfs_csv_left.sort_values(by=['bid', 'date']).reset_index(drop=True)

    @property
    def odfs_bid_csv_right(self):
        return self.odfs_csv_right.sort_values(by=['bid', 'date']).reset_index(drop=True)

    @property
    def pdf_bid_csv_left(self):
        return self.pdf_csv_left.sort_values(by=['bid', 'date']).reset_index(drop=True)

    @property
    def pdf_bid_csv_right(self):
        return self.pdf_csv_right.sort_values(by=['bid', 'date']).reset_index(drop=True)

    @property
    def odfs_bid_csv_left_index(self):
        return self.odfs_csv_left.sort_values(by=['bid', 'date']).set_index('bid')

    @property
    def odfs_bid_csv_right_index(self):
        return self.odfs_csv_right.sort_values(by=['bid', 'date']).set_index('bid')

    @property
    def pdf_bid_csv_left_index(self):
        return self.pdf_csv_left.sort_values(by=['bid', 'date']).set_index('bid')

    @property
    def pdf_bid_csv_right_index(self):
        return self.pdf_csv_right.sort_values(by=['bid', 'date']).set_index('bid')

    def test_assert_original_dataframe_equal(self):
        assert_frame_equal(self.odfs_csv_left.to_pandas(), self.pdf_csv_left, check_dtype=False)
        assert_frame_equal(self.odfs_csv_right.to_pandas(), self.pdf_csv_right, check_dtype=False)
        assert_frame_equal(self.odfs_csv_left_index.to_pandas(), self.pdf_csv_left_index, check_dtype=False)
        assert_frame_equal(self.odfs_csv_right_index.to_pandas(), self.pdf_csv_right_index, check_dtype=False)
        assert_frame_equal(self.odfs_bid_csv_left.to_pandas(), self.pdf_bid_csv_left, check_dtype=False)
        assert_frame_equal(self.odfs_bid_csv_right.to_pandas(), self.pdf_bid_csv_right, check_dtype=False)
        assert_frame_equal(self.odfs_bid_csv_left_index.to_pandas(), self.pdf_bid_csv_left_index, check_dtype=False)
        assert_frame_equal(self.odfs_bid_csv_right_index.to_pandas(), self.pdf_bid_csv_right_index, check_dtype=False)

    def test_merge_asof_from_dfs_param_on(self):
        pdf = pd.merge_asof(self.pdf_csv_left, self.pdf_csv_right, on='date')
        odf = orca.merge_asof(self.odfs_csv_left, self.odfs_csv_right, on='date')
        assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

        pdf = pd.merge_asof(self.pdf_bid_csv_left, self.pdf_bid_csv_right, on='bid')
        odf = orca.merge_asof(self.odfs_bid_csv_left, self.odfs_bid_csv_right, on='bid')
        assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False, check_like=False)

    def test_merge_asof_from_dfs_param_leftonrighton(self):
        pdf = pd.merge_asof(self.pdf_csv_left, self.pdf_csv_right, left_on='date', right_on='date')
        odf = orca.merge_asof(self.odfs_csv_left, self.odfs_csv_right, left_on='date', right_on='date')
        assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

        pdf = pd.merge_asof(self.pdf_bid_csv_left, self.pdf_bid_csv_right, left_on='bid', right_on='bid')
        odf = orca.merge_asof(self.odfs_bid_csv_left, self.odfs_bid_csv_right, left_on='bid', right_on='bid')
        assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False, check_like=False)

    def test_merge_asof_from_dfs_param_index(self):
        pdf = pd.merge_asof(self.pdf_csv_left_index, self.pdf_csv_right_index, left_index=True, right_index=True)
        odf = orca.merge_asof(self.odfs_csv_left_index, self.odfs_csv_right_index, left_index=True, right_index=True)
        assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

        pdf = pd.merge_asof(self.pdf_bid_csv_left_index, self.pdf_bid_csv_right_index, left_index=True, right_index=True)
        odf = orca.merge_asof(self.odfs_bid_csv_left_index, self.odfs_bid_csv_right_index, left_index=True, right_index=True)
        assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False, check_like=False)

    def test_merge_asof_from_dfs_param_by(self):
        pdf = pd.merge_asof(self.pdf_csv_left, self.pdf_csv_right, on='date', by='ticker')
        odf = orca.merge_asof(self.odfs_csv_left, self.odfs_csv_right, on='date', by='ticker')
        # TODO:ORCA by bug
        # assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

        pdf = pd.merge_asof(self.pdf_bid_csv_left, self.pdf_bid_csv_right, on='bid', by='ticker')
        odf = orca.merge_asof(self.odfs_bid_csv_left, self.odfs_bid_csv_right, on='bid', by='ticker')
        # TODO:ORCA by bug
        # assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False, check_like=False)

    def test_merge_asof_from_dfs_param_leftbyrightby(self):
        pdf = pd.merge_asof(self.pdf_csv_left, self.pdf_csv_right, on='date', left_by='ticker', right_by='ticker')
        # TODO:ORCA by bug
        # odf = orca.merge_asof(self.odfs_csv_left, self.odfs_csv_right, on='date', left_by='ticker', right_by='ticker')
        # TODO:ORCA by bug
        # assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

        pdf = pd.merge_asof(self.pdf_bid_csv_left, self.pdf_bid_csv_right, on='bid', left_by='ticker',
                            right_by='ticker')
        # TODO:ORCA by bug
        # odf = orca.merge_asof(self.odfs_bid_csv_left, self.odfs_bid_csv_right, on='bid', left_by='ticker', right_by='ticker')
        # TODO:ORCA by bug
        # assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False, check_like=False)

    def test_merge_asof_from_dfs_param_suffixes(self):
        pdf = pd.merge_asof(self.pdf_csv_left, self.pdf_csv_right, on='date', suffixes=('_left', '_right'))
        odf = orca.merge_asof(self.odfs_csv_left, self.odfs_csv_right, on='date', suffixes=('_left', '_right'))
        assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

        pdf = pd.merge_asof(self.pdf_bid_csv_left, self.pdf_bid_csv_right, on='bid', suffixes=('_left', '_right'))
        odf = orca.merge_asof(self.odfs_bid_csv_left, self.odfs_bid_csv_right, on='bid', suffixes=('_left', '_right'))
        assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False, check_like=False)

    def test_merge_asof_from_dfs_param_leftonrighton_param_suffixes(self):
        pdf = pd.merge_asof(self.pdf_csv_left, self.pdf_csv_right, left_on='date', right_on='date',
                            suffixes=('_left', '_right'))
        odf = orca.merge_asof(self.odfs_csv_left, self.odfs_csv_right, left_on='date', right_on='date',
                              suffixes=('_left', '_right'))
        assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

        pdf = pd.merge_asof(self.pdf_bid_csv_left, self.pdf_bid_csv_right, left_on='bid', right_on='bid',
                            suffixes=('_left', '_right'))
        odf = orca.merge_asof(self.odfs_bid_csv_left, self.odfs_bid_csv_right, left_on='bid', right_on='bid',
                              suffixes=('_left', '_right'))
        assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False, check_like=False)

    def test_merge_asof_from_dfs_param_leftonrighton_param_by(self):
        pdf = pd.merge_asof(self.pdf_csv_left, self.pdf_csv_right, left_on='date', right_on='date', by='ticker')
        odf = orca.merge_asof(self.odfs_csv_left, self.odfs_csv_right, left_on='date', right_on='date', by='ticker')
        # TODO:ORCA by bug
        # assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

        pdf = pd.merge_asof(self.pdf_bid_csv_left, self.pdf_bid_csv_right, left_on='bid', right_on='bid', by='ticker')
        odf = orca.merge_asof(self.odfs_bid_csv_left, self.odfs_bid_csv_right, left_on='bid', right_on='bid', by='ticker')
        # TODO:ORCA by bug
        # assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False, check_like=False)

    def test_merge_asof_from_dfs_param_leftonrighton_param_leftbyrightby(self):
        pdf = pd.merge_asof(self.pdf_csv_left, self.pdf_csv_right, on='date', left_by='ticker', right_by='ticker')
        # TODO:ORCA by bug
        # odf = orca.merge_asof(self.odfs_csv_left, self.odfs_csv_right, on='date', left_by='ticker', right_by='ticker')
        # assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

        pdf = pd.merge_asof(self.pdf_bid_csv_left, self.pdf_bid_csv_right, on='bid', left_by='ticker',
                            right_by='ticker')
        # TODO:ORCA by bug
        # odf = orca.merge_asof(self.odfs_bid_csv_left, self.odfs_bid_csv_right, on='bid', left_by='ticker', right_by='ticker')
        # assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False, check_like=False)

    def test_merge_asof_from_dfs_param_index_param_by(self):
        pdf = pd.merge_asof(self.pdf_csv_left_index, self.pdf_csv_right_index, left_index=True, right_index=True,
                            by='ticker')
        odf = orca.merge_asof(self.odfs_csv_left_index, self.odfs_csv_right_index, left_index=True, right_index=True,
                              by='ticker')
        # TODO:ORCA by bug
        # assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

        pdf = pd.merge_asof(self.pdf_bid_csv_left_index, self.pdf_bid_csv_right_index, left_index=True,
                            right_index=True, by='ticker')
        odf = orca.merge_asof(self.odfs_bid_csv_left_index, self.odfs_bid_csv_right_index, left_index=True,
                              right_index=True, by='ticker')
        # TODO:ORCA by bug
        # assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False, check_like=False)

    def test_merge_asof_from_dfs_param_index_param_leftbyrightby(self):
        pdf = pd.merge_asof(self.pdf_csv_left_index, self.pdf_csv_right_index, left_index=True, right_index=True,
                            left_by='ticker', right_by='ticker')
        odf = orca.merge_asof(self.odfs_csv_left_index, self.odfs_csv_right_index, left_index=True, right_index=True,
                              left_by='ticker', right_by='ticker')
        # TODO:ORCA by bug
        # assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

        pdf = pd.merge_asof(self.pdf_bid_csv_left_index, self.pdf_bid_csv_right_index, left_index=True,
                            right_index=True, left_by='ticker',
                            right_by='ticker')
        odf = orca.merge_asof(self.odfs_bid_csv_left_index, self.odfs_bid_csv_right_index, left_index=True,
                              right_index=True, left_by='ticker',
                              right_by='ticker')
        # TODO:ORCA by bug
        # assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False, check_like=False)

    def test_merge_asof_from_dfs_param_index_param_suffixes(self):
        pdf = pd.merge_asof(self.pdf_csv_left_index, self.pdf_csv_right_index, left_index=True, right_index=True,
                            suffixes=('_left', '_right'))
        odf = orca.merge_asof(self.odfs_csv_left_index, self.odfs_csv_right_index, left_index=True, right_index=True,
                              suffixes=('_left', '_right'))
        assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

        pdf = pd.merge_asof(self.pdf_bid_csv_left_index, self.pdf_bid_csv_right_index, left_index=True,
                            right_index=True, suffixes=('_left', '_right'))
        odf = orca.merge_asof(self.odfs_bid_csv_left_index, self.odfs_bid_csv_right_index, left_index=True,
                              right_index=True,
                              suffixes=('_left', '_right'))
        assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False, check_like=False)

    def test_merge_asof_from_dfs_param_on_param_by_suffixes(self):
        pdf = pd.merge_asof(self.pdf_csv_left, self.pdf_csv_right, on='date', by='ticker', suffixes=('_left', '_right'))
        odf = orca.merge_asof(self.odfs_csv_left, self.odfs_csv_right, on='date', by='ticker',
                              suffixes=('_left', '_right'))
        # TODO:ORCA by bug
        # assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

        pdf = pd.merge_asof(self.pdf_bid_csv_left, self.pdf_bid_csv_right, on='bid', by='ticker',
                            suffixes=('_left', '_right'))
        odf = orca.merge_asof(self.odfs_bid_csv_left, self.odfs_bid_csv_right, on='bid', by='ticker',
                              suffixes=('_left', '_right'))
        # TODO:ORCA by bug
        # assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False, check_like=False)

    def test_merge_asof_from_dfs_param_on_param_leftbyrightby_suffixes(self):
        pdf = pd.merge_asof(self.pdf_csv_left, self.pdf_csv_right, on='date',
                            left_by='ticker', right_by='ticker', suffixes=('_left', '_right'))
        # TODO:ORCA by bug
        # odf = orca.merge_asof(self.odfs_csv_left, self.odfs_csv_right, on='date',
        #                       left_by='ticker', right_by='ticker', suffixes=('_left', '_right'))
        # assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

        pdf = pd.merge_asof(self.pdf_bid_csv_left, self.pdf_bid_csv_right, on='bid', left_by='ticker',
                            right_by='ticker', suffixes=('_left', '_right'))
        # TODO:ORCA by bug
        # odf = orca.merge_asof(self.odfs_bid_csv_left, self.odfs_bid_csv_right, on='bid', left_by='ticker', right_by='ticker', suffixes=('_left', '_right'))
        # assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False, check_like=False)

    def test_merge_asof_from_dfs_param_leftonrighton_param_by_suffixes(self):
        pdf = pd.merge_asof(self.pdf_csv_left, self.pdf_csv_right, left_on='date', right_on='date',
                            by='ticker', suffixes=('_left', '_right'))
        odf = orca.merge_asof(self.odfs_csv_left, self.odfs_csv_right, left_on='date', right_on='date',
                              by='ticker', suffixes=('_left', '_right'))
        # TODO:ORCA by bug
        # assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

        pdf = pd.merge_asof(self.pdf_bid_csv_left, self.pdf_bid_csv_right, left_on='bid', right_on='bid', by='ticker',
                            suffixes=('_left', '_right'))
        pdf.fillna("", inplace=True)
        odf = orca.merge_asof(self.odfs_bid_csv_left, self.odfs_bid_csv_right, left_on='bid', right_on='bid', by='ticker',
                              suffixes=('_left', '_right'))
        # TODO:ORCA by bug
        # assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False, check_like=False)

    def test_merge_asof_from_dfs_param_leftonrighton_param_leftbyrightby_suffixes(self):
        pdf = pd.merge_asof(self.pdf_csv_left, self.pdf_csv_right, left_on='date', right_on='date',
                            left_by='ticker', right_by='ticker', suffixes=('_left', '_right'))
        # TODO:ORCA by bug
        # odf = orca.merge_asof(self.odfs_csv_left, self.odfs_csv_right, left_on='date', right_on='date',
        #                       left_by='ticker', right_by='ticker', suffixes=('_left', '_right'))
        # assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

        pdf = pd.merge_asof(self.pdf_bid_csv_left, self.pdf_bid_csv_right, left_on='bid', right_on='bid',
                            left_by='ticker', right_by='ticker', suffixes=('_left', '_right'))
        # TODO:ORCA by bug
        # odf = orca.merge_asof(self.odfs_bid_csv_left, self.odfs_bid_csv_right, left_on='bid', right_on='bid',
        #                       left_by='ticker', right_by='ticker', suffixes=('_left', '_right'))
        # assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False, check_like=False)

    def test_merge_asof_from_dfs_param_index_param_by_suffixes(self):
        pdf = pd.merge_asof(self.pdf_csv_left_index, self.pdf_csv_right_index, left_index=True, right_index=True,
                            by='ticker', suffixes=('_left', '_right'))
        odf = orca.merge_asof(self.odfs_csv_left_index, self.odfs_csv_right_index, left_index=True, right_index=True,
                              by='ticker', suffixes=('_left', '_right'))
        # TODO:ORCA by bug
        # assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

        pdf = pd.merge_asof(self.pdf_bid_csv_left_index, self.pdf_bid_csv_right_index, left_index=True,
                            right_index=True,
                            by='ticker', suffixes=('_left', '_right'))
        odf = orca.merge_asof(self.odfs_bid_csv_left_index, self.odfs_bid_csv_right_index, left_index=True,
                              right_index=True,
                              by='ticker', suffixes=('_left', '_right'))
        # TODO:ORCA by bug
        # assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False, check_like=False)

    def test_merge_asof_from_dfs_param_index_param_leftbyrightby_suffixes(self):
        pdf = pd.merge_asof(self.pdf_csv_left_index, self.pdf_csv_right_index, left_index=True, right_index=True,
                            left_by='ticker', right_by='ticker', suffixes=('_left', '_right'))
        # TODO:ORCA by bug
        # odf = orca.merge_asof(self.odfs_csv_left_index, self.odfs_csv_right_index, left_index=True, right_index=True,
        #                       left_by='ticker', right_by='ticker', suffixes=('_left', '_right'))
        # assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

        pdf = pd.merge_asof(self.pdf_bid_csv_left_index, self.pdf_bid_csv_right_index, left_index=True,
                            right_index=True,
                            left_by='ticker', right_by='ticker', suffixes=('_left', '_right'))
        # TODO:ORCA by bug
        # odf = orca.merge_asof(self.odfs_bid_csv_left_index, self.odfs_bid_csv_right_index, left_index=True, right_index=True,
        #                       left_by='ticker', right_by='ticker', suffixes=('_left', '_right'))
        # assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False, check_like=False)

    def test_merge_asof_from_dfs_param_on_param_index(self):
        pdf = pd.merge_asof(self.pdf_csv_left_index, self.pdf_csv_right, left_index=True, right_on='date')
        # TODO:ORCA error left_index, right_on not supported
        # odf = orca.merge_asof(self.odfs_csv_left_index, self.odfs_csv_right_index, left_index=True, right_on='date')
        # assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

        pdf = pd.merge_asof(self.pdf_csv_left, self.pdf_csv_right_index, right_index=True, left_on='date')
        # TODO:ORCA error left_index, right_on not supported
        # odf = orca.merge_asof(self.odfs_csv_left, self.odfs_csv_right_index, right_index=True, left_on='date')
        # assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

        pdf = pd.merge_asof(self.pdf_bid_csv_left_index, self.pdf_bid_csv_right, left_index=True, right_on='bid')
        # TODO:ORCA error left_index, right_on not supported
        # odf = orca.merge_asof(self.odfs_bid_csv_left_index, self.odfs_bid_csv_right, left_index=True, right_on='bid')
        # assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False, check_like=False)

        pdf = pd.merge_asof(self.pdf_bid_csv_left, self.pdf_bid_csv_right_index, right_index=True, left_on='bid')
        # TODO:ORCA error left_index, right_on not supported
        # odf = orca.merge_asof(self.odfs_bid_csv_left, self.odfs_bid_csv_right_index, right_index=True, left_on='bid')
        # assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False, check_like=False)

    def test_merge_asof_from_dfs_param_on_param_index_param_by(self):
        pdf = pd.merge_asof(self.pdf_csv_left_index, self.pdf_csv_right, left_index=True, right_on='date', by='ticker')
        # TODO:ORCA by bug
        # odf = orca.merge_asof(self.odfs_csv_left_index, self.odfs_csv_right_index, left_index=True, right_on='date', by='ticker')
        # assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

        pdf = pd.merge_asof(self.pdf_csv_left, self.pdf_csv_right_index, right_index=True, left_on='date', by='ticker')
        # TODO:ORCA by bug
        # odf = orca.merge_asof(self.odfs_csv_left_index, self.odfs_csv_right_index, right_index=True, left_on='date', by='ticker')
        # assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

        pdf = pd.merge_asof(self.pdf_bid_csv_left_index, self.pdf_bid_csv_right, left_index=True, right_on='bid',
                            by='ticker')
        # TODO:ORCA by bug
        # odf = orca.merge_asof(self.odfs_bid_csv_left_index, self.odfs_bid_csv_right, left_index=True, right_on='bid', by='ticker')
        # assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False, check_like=False)

        pdf = pd.merge_asof(self.pdf_bid_csv_left, self.pdf_bid_csv_right_index, right_index=True, left_on='bid',
                            by='ticker')
        # TODO:ORCA by bug
        # odf = orca.merge_asof(self.odfs_bid_csv_left, self.odfs_bid_csv_right_index, right_index=True, left_on='bid', by='ticker')
        # assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False, check_like=False)

    def test_merge_asof_from_dfs_param_on_param_index_param_by_param_suffixes(self):
        pdf = pd.merge_asof(self.pdf_csv_left_index, self.pdf_csv_right, left_index=True, right_on='date',
                            by='ticker', suffixes=('_left', '_right'))
        # TODO:ORCA by bug
        # odf = orca.merge_asof(self.odfs_csv_left_index, self.odfs_csv_right_index, left_index=True, right_on='date',
        #                       by='ticker', suffixes=('_left', '_right'))
        # assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

        pdf = pd.merge_asof(self.pdf_csv_left, self.pdf_csv_right_index, right_index=True, left_on='date',
                            by='ticker', suffixes=('_left', '_right'))
        # TODO:ORCA by bug
        # odf = orca.merge_asof(self.odfs_csv_left_index, self.odfs_csv_right_index, right_index=True, left_on='date',
        #                       by='ticker', suffixes=('_left', '_right'))
        # assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False)

        pdf = pd.merge_asof(self.pdf_bid_csv_left_index, self.pdf_bid_csv_right, left_index=True, right_on='bid',
                            by='ticker', suffixes=('_left', '_right'))
        # TODO:ORCA by bug
        # odf = orca.merge_asof(self.odfs_bid_csv_left_index, self.odfs_bid_csv_right, left_index=True, right_on='bid',
        #                       by='ticker', suffixes=('_left', '_right'))
        # assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False, check_like=False)

        pdf = pd.merge_asof(self.pdf_bid_csv_left, self.pdf_bid_csv_right_index, right_index=True, left_on='bid',
                            by='ticker', suffixes=('_left', '_right'))
        # TODO:ORCA by bug
        # odf = orca.merge_asof(self.odfs_bid_csv_left, self.odfs_bid_csv_right_index, right_index=True, left_on='bid',
        #                       by='ticker', suffixes=('_left', '_right'))
        # assert_frame_equal(odf.to_pandas().fillna(""), pdf.fillna(""), check_dtype=False, check_like=False)


if __name__ == '__main__':
    unittest.main()
